import argparse
import asyncio
import aiohttp
import time
import statistics
import json
from collections import Counter, defaultdict


COUNTER_METRICS = {
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
}


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((len(xs) - 1) * p / 100))
    return xs[k]


def stat(xs):
    if not xs:
        return {
            "mean": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    return {
        "mean": statistics.mean(xs),
        "max": max(xs),
        "p50": pct(xs, 50),
        "p90": pct(xs, 90),
        "p99": pct(xs, 99),
    }


def metrics_url_from_chat_url(url):
    if url.endswith("/metrics"):
        return url
    if "/v1/" in url:
        return url.split("/v1/")[0] + "/metrics"
    return url.rstrip("/") + "/metrics"


def parse_prometheus_metrics(text):
    wanted = {
        "vllm:kv_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:prompt_tokens_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:generation_tokens_total",
    }

    out = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        if metric_name not in wanted:
            continue

        try:
            value = float(line.rsplit(" ", 1)[-1])
        except Exception:
            continue

        out[metric_name] = value

    return out


async def fetch_one_metrics(session, chat_url):
    murl = metrics_url_from_chat_url(chat_url)

    try:
        async with session.get(
            murl,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return {
                    "ok": False,
                    "url": chat_url,
                    "metrics_url": murl,
                    "status": resp.status,
                    "error": await resp.text(),
                }

            text = await resp.text()
            parsed = parse_prometheus_metrics(text)
            return {
                "ok": True,
                "url": chat_url,
                "metrics_url": murl,
                "metrics": parsed,
            }

    except Exception as e:
        return {
            "ok": False,
            "url": chat_url,
            "metrics_url": murl,
            "error": repr(e),
        }


async def fetch_metrics_snapshot(session, urls):
    results = await asyncio.gather(
        *[fetch_one_metrics(session, url) for url in urls],
        return_exceptions=True,
    )

    out = []
    for r in results:
        if isinstance(r, Exception):
            out.append({"ok": False, "error": repr(r)})
        else:
            out.append(r)
    return out


async def metrics_sampler(session, urls, interval_s, stop_event, samples):
    while not stop_event.is_set():
        ts = time.time()
        per_url = await fetch_metrics_snapshot(session, urls)

        samples.append({
            "timestamp": ts,
            "per_url": per_url,
        })

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def one_request(session, url, model, prompt, max_tokens, idx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t0 = time.perf_counter()
    first = None
    last = None
    chunks = 0
    text = ""

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "ok": False,
                    "idx": idx,
                    "url": url,
                    "status": resp.status,
                    "error": body[:500],
                }

            async for raw in resp.content:
                now = time.perf_counter()
                line = raw.decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                for part in line.splitlines():
                    part = part.strip()

                    if not part.startswith("data:"):
                        continue

                    data = part[5:].strip()

                    if data == "[DONE]":
                        last = now
                        break

                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue

                    if first is None:
                        first = now

                    chunks += 1
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    text += delta.get("content", "") or ""

            t1 = time.perf_counter()

    except Exception as e:
        return {
            "ok": False,
            "idx": idx,
            "url": url,
            "error": repr(e),
        }

    if first is None:
        return {
            "ok": False,
            "idx": idx,
            "url": url,
            "error": "no first token",
        }

    last = last or t1
    e2e = t1 - t0
    ttft = first - t0
    decode_time = max(0.0, last - first)
    itl = decode_time / max(chunks - 1, 1)

    return {
        "ok": True,
        "idx": idx,
        "url": url,
        "ttft": ttft,
        "e2e": e2e,
        "decode_time": decode_time,
        "itl": itl,
        "chunks": chunks,
        "chars": len(text),
    }


def summarize_metrics(samples):
    by_metric = defaultdict(list)
    by_url_metric = defaultdict(lambda: defaultdict(list))

    for sample in samples:
        for item in sample["per_url"]:
            if not item.get("ok"):
                continue

            url = item["url"]
            metrics = item.get("metrics", {})

            for k, v in metrics.items():
                by_metric[k].append(v)
                by_url_metric[url][k].append(v)

    overall = {k: stat(vs) for k, vs in by_metric.items()}
    per_url = {
        url: {k: stat(vs) for k, vs in metric_map.items()}
        for url, metric_map in by_url_metric.items()
    }

    return {
        "num_samples": len(samples),
        "overall": overall,
        "per_url": per_url,
    }


def summarize_counter_deltas(before_snapshot, after_snapshot):
    before_by_url = {
        item["url"]: item.get("metrics", {})
        for item in before_snapshot
        if item.get("ok")
    }
    after_by_url = {
        item["url"]: item.get("metrics", {})
        for item in after_snapshot
        if item.get("ok")
    }

    per_url_deltas = {}
    aggregate = defaultdict(float)
    aggregate_found = defaultdict(bool)

    for url, after_metrics in after_by_url.items():
        before_metrics = before_by_url.get(url, {})
        per_url_deltas[url] = {}

        for metric_name in COUNTER_METRICS:
            before_val = before_metrics.get(metric_name)
            after_val = after_metrics.get(metric_name)

            if not isinstance(before_val, (int, float)) or not isinstance(after_val, (int, float)):
                per_url_deltas[url][metric_name] = None
                continue

            delta = after_val - before_val

            if delta < 0:
                per_url_deltas[url][metric_name] = None
                continue

            per_url_deltas[url][metric_name] = delta
            aggregate[metric_name] += delta
            aggregate_found[metric_name] = True

    overall_deltas = {
        metric_name: aggregate[metric_name] if aggregate_found[metric_name] else None
        for metric_name in COUNTER_METRICS
    }

    pt = overall_deltas.get("vllm:prompt_tokens_total")
    gen = overall_deltas.get("vllm:generation_tokens_total")

    overall_deltas["total_tokens_delta_total"] = (
        pt + gen
        if isinstance(pt, (int, float)) and isinstance(gen, (int, float))
        else None
    )

    derived = {
        "prefix_cache_hit_rate_total": None,
        "external_prefix_cache_hit_rate_total": None,
        "prompt_cache_hit_rate_total": None,
    }

    q = overall_deltas.get("vllm:prefix_cache_queries_total")
    h = overall_deltas.get("vllm:prefix_cache_hits_total")
    if q and q > 0:
        derived["prefix_cache_hit_rate_total"] = h / q

    eq = overall_deltas.get("vllm:external_prefix_cache_queries_total")
    eh = overall_deltas.get("vllm:external_prefix_cache_hits_total")
    if eq and eq > 0:
        derived["external_prefix_cache_hit_rate_total"] = eh / eq

    prompt_total = overall_deltas.get("vllm:prompt_tokens_total")
    prompt_cached = overall_deltas.get("vllm:prompt_tokens_cached_total")
    if prompt_total and prompt_total > 0:
        derived["prompt_cache_hit_rate_total"] = prompt_cached / prompt_total

    return {
        "overall_deltas": overall_deltas,
        "per_url_deltas": per_url_deltas,
        "derived": derived,
    }


def build_token_throughput(counter_summary, total_time_s):
    deltas = counter_summary.get("overall_deltas", {})

    prompt_tokens_total = deltas.get("vllm:prompt_tokens_total")
    generation_tokens_total = deltas.get("vllm:generation_tokens_total")
    total_tokens_total = deltas.get("total_tokens_delta_total")

    def tok_s(x):
        if isinstance(x, (int, float)) and total_time_s > 0:
            return x / total_time_s
        return None

    return {
        "prompt_tokens_total": prompt_tokens_total,
        "generation_tokens_total": generation_tokens_total,
        "total_tokens_total": total_tokens_total,
        "prompt_tokens_per_second": tok_s(prompt_tokens_total),
        "generation_tokens_per_second": tok_s(generation_tokens_total),
        "total_tokens_per_second": tok_s(total_tokens_total),
    }


def build_token_throughput_by_url(counter_summary, total_time_s):
    out = {}

    for url, deltas in counter_summary.get("per_url_deltas", {}).items():
        prompt_tokens_total = deltas.get("vllm:prompt_tokens_total")
        generation_tokens_total = deltas.get("vllm:generation_tokens_total")

        total_tokens_total = (
            prompt_tokens_total + generation_tokens_total
            if isinstance(prompt_tokens_total, (int, float))
            and isinstance(generation_tokens_total, (int, float))
            else None
        )

        def tok_s(x):
            if isinstance(x, (int, float)) and total_time_s > 0:
                return x / total_time_s
            return None

        out[url] = {
            "prompt_tokens_total": prompt_tokens_total,
            "generation_tokens_total": generation_tokens_total,
            "total_tokens_total": total_tokens_total,
            "prompt_tokens_per_second": tok_s(prompt_tokens_total),
            "generation_tokens_per_second": tok_s(generation_tokens_total),
            "total_tokens_per_second": tok_s(total_tokens_total),
        }

    return out


def build_pd_token_throughput(token_throughput_by_url):
    """Extract prefill/decode token throughput for 4P4D.

    Assumption used by current deployment:
      - Prefill metrics endpoint: port 20005
      - Decode metrics endpoint: port 40005
    """
    prefill_url = None
    decode_url = None

    for url in token_throughput_by_url:
        if ":20005" in url:
            prefill_url = url
        elif ":40005" in url:
            decode_url = url

    prefill = token_throughput_by_url.get(prefill_url, {}) if prefill_url else {}
    decode = token_throughput_by_url.get(decode_url, {}) if decode_url else {}

    return {
        "prefill_metrics_url": prefill_url,
        "decode_metrics_url": decode_url,

        "prefill_prompt_tokens_total": prefill.get("prompt_tokens_total"),
        "prefill_generation_tokens_total": prefill.get("generation_tokens_total"),
        "prefill_prompt_tokens_per_second": prefill.get("prompt_tokens_per_second"),
        "prefill_generation_tokens_per_second": prefill.get("generation_tokens_per_second"),

        "decode_prompt_tokens_total": decode.get("prompt_tokens_total"),
        "decode_generation_tokens_total": decode.get("generation_tokens_total"),
        "decode_prompt_tokens_per_second": decode.get("prompt_tokens_per_second"),
        "decode_generation_tokens_per_second": decode.get("generation_tokens_per_second"),
    }


async def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--url", default=None)
    ap.add_argument("--urls", nargs="+", default=None)

    ap.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--input-chars", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--name", default="run")

    ap.add_argument(
        "--metrics-interval",
        type=float,
        default=0.2,
        help="Sampling interval for /metrics in seconds. Set <=0 to disable metrics sampling.",
    )
    ap.add_argument("--metrics-urls", nargs="+", default=None)
    args = ap.parse_args()

    if args.urls:
        urls = args.urls
    elif args.url:
        urls = [args.url]
    else:
        raise ValueError("Please provide either --url or --urls.")

    metrics_urls = args.metrics_urls if args.metrics_urls else urls

    base = "Please read the following text and answer briefly.\n\n"
    filler = "This is a synthetic long-prefill benchmark sentence. " * 10000
    prompt = (base + filler)[: args.input_chars]

    sem = asyncio.Semaphore(args.concurrency)
    results = []
    metrics_samples = []
    stop_event = asyncio.Event()

    async with aiohttp.ClientSession() as session:
        sampler_task = None

        # Important:
        # Use explicit before/after snapshots for Prometheus counters.
        # The periodic sampler is not reliable for counter deltas under high concurrency
        # because the first sample may occur after many requests have already completed.
        metrics_before = await fetch_metrics_snapshot(session, metrics_urls)

        if args.metrics_interval > 0:
            sampler_task = asyncio.create_task(
                metrics_sampler(
                    session=session,
                    urls=metrics_urls,
                    interval_s=args.metrics_interval,
                    stop_event=stop_event,
                    samples=metrics_samples,
                )
            )

        async def wrapped(i):
            url = urls[i % len(urls)]

            async with sem:
                return await one_request(
                    session=session,
                    url=url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    idx=i,
                )

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(wrapped(i)) for i in range(args.num_requests)]

        for t in asyncio.as_completed(tasks):
            r = await t
            results.append(r)
            print(f"done {len(results)}/{args.num_requests}: {r}")

        total = time.perf_counter() - t0

        metrics_after = await fetch_metrics_snapshot(session, metrics_urls)

        if sampler_task:
            stop_event.set()
            await sampler_task

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]

    def summary(key):
        xs = [r[key] for r in ok]
        return {
            "mean": statistics.mean(xs) if xs else None,
            "p50": pct(xs, 50),
            "p90": pct(xs, 90),
            "p99": pct(xs, 99),
        }

    request_count_by_url = Counter(r["url"] for r in results)
    completed_count_by_url = Counter(r["url"] for r in ok)
    failed_count_by_url = Counter(r["url"] for r in bad)

    metrics_summary = summarize_metrics(metrics_samples)
    counter_summary = summarize_counter_deltas(metrics_before, metrics_after)

    metrics_summary["counter_deltas_from_snapshots"] = counter_summary
    metrics_summary["derived"] = counter_summary.get("derived", {})

    token_throughput = build_token_throughput(counter_summary, total)
    token_throughput_by_url = build_token_throughput_by_url(counter_summary, total)
    pd_token_throughput = build_pd_token_throughput(token_throughput_by_url)

    out = {
        "name": args.name,
        "mode": "multi_url_round_robin" if len(urls) > 1 else "single_url",
        "urls": urls,
        "metrics_urls": metrics_urls,
        "num_urls": len(urls),
        "model": args.model,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "input_chars": args.input_chars,
        "max_tokens": args.max_tokens,
        "completed": len(ok),
        "failed": len(bad),
        "total_time_s": total,
        "request_throughput_req_s": len(ok) / total if total > 0 else None,
        "request_count_by_url": dict(request_count_by_url),
        "completed_count_by_url": dict(completed_count_by_url),
        "failed_count_by_url": dict(failed_count_by_url),
        "ttft_s": summary("ttft"),
        "itl_s": summary("itl"),
        "decode_time_s": summary("decode_time"),
        "e2e_s": summary("e2e"),
        "token_throughput": token_throughput,
        "token_throughput_by_url": token_throughput_by_url,
        "pd_token_throughput": pd_token_throughput,
        "metrics_summary": metrics_summary,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metrics_raw": metrics_samples,
        "raw": results,
    }

    print("\n===== SUMMARY =====")
    print(json.dumps(out, indent=2))

    with open(f"{args.name}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())