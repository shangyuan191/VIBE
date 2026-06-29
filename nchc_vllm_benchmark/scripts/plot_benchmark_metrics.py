#!/usr/bin/env python3
import glob
import json
import sys
import argparse

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False


def load_results(patterns):
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))

    items = []
    for p in set(paths):
        try:
            with open(p) as f:
                d = json.load(f)
            c = int(d.get("concurrency", 0))
            items.append((c, d, p))
        except Exception:
            continue

    items.sort(key=lambda x: x[0])
    return items


def ratio_to_percent(x):
    return x * 100 if isinstance(x, (int, float)) else None


def annotate_points(ax, xs, ys, fmt="{:.2f}", y_offset=5):
    for x, y in zip(xs, ys):
        if y is None:
            continue
        ax.annotate(
            fmt.format(y),
            (x, y),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            fontsize=8,
        )


def fallback_generation_tok_s(d):
    tok = d.get("token_throughput", {})
    measured = tok.get("generation_tokens_per_second")
    if isinstance(measured, (int, float)):
        return measured

    req_s = d.get("request_throughput_req_s")
    max_tokens = d.get("max_tokens")
    if isinstance(req_s, (int, float)) and isinstance(max_tokens, (int, float)):
        return req_s * max_tokens

    return None


def extract_series(items):
    xs = []

    request_throughput = []

    prompt_token_throughput = []
    generation_token_throughput = []
    total_token_throughput = []

    prefill_prompt_token_throughput = []
    decode_generation_token_throughput = []

    ttft_mean = []
    ttft_p99 = []
    e2e_mean = []
    e2e_p99 = []
    kv_mean = []
    kv_peak = []
    running_peak = []
    waiting_peak = []
    prefix_hit = []
    prompt_hit = []

    for c, d, p in items:
        xs.append(c)

        request_throughput.append(d.get("request_throughput_req_s"))

        tok = d.get("token_throughput", {})
        prompt_token_throughput.append(tok.get("prompt_tokens_per_second"))
        generation_token_throughput.append(fallback_generation_tok_s(d))
        total_token_throughput.append(tok.get("total_tokens_per_second"))

        pd_tok = d.get("pd_token_throughput", {})
        prefill_prompt_token_throughput.append(
            pd_tok.get("prefill_prompt_tokens_per_second")
        )
        decode_generation_token_throughput.append(
            pd_tok.get("decode_generation_tokens_per_second")
        )

        ttft_mean.append(d.get("ttft_s", {}).get("mean"))
        ttft_p99.append(d.get("ttft_s", {}).get("p99"))
        e2e_mean.append(d.get("e2e_s", {}).get("mean"))
        e2e_p99.append(d.get("e2e_s", {}).get("p99"))

        m = d.get("metrics_summary", {}).get("overall", {})
        kv = m.get("vllm:kv_cache_usage_perc", {})
        running = m.get("vllm:num_requests_running", {})
        waiting = m.get("vllm:num_requests_waiting", {})

        kv_mean.append(ratio_to_percent(kv.get("mean")))
        kv_peak.append(ratio_to_percent(kv.get("max")))
        running_peak.append(running.get("max"))
        waiting_peak.append(waiting.get("max"))

        derived = d.get("metrics_summary", {}).get("derived", {})
        prefix_hit.append(ratio_to_percent(derived.get("prefix_cache_hit_rate_total")))
        prompt_hit.append(ratio_to_percent(derived.get("prompt_cache_hit_rate_total")))

    has_physical_pd = any(x is not None for x in prefill_prompt_token_throughput) or any(
        x is not None for x in decode_generation_token_throughput
    )

    # For independent serving, there is no physical prefill/decode separation.
    # We still visualize prompt/generation token throughput as prefill/decode workload proxies.
    if not has_physical_pd:
        prefill_prompt_token_throughput = prompt_token_throughput
        decode_generation_token_throughput = generation_token_throughput

    has_pd_panel = True

    return {
        "x": xs,
        "has_pd": has_pd_panel,
        "has_physical_pd": has_physical_pd,
        "request_throughput": request_throughput,
        "prompt_token_throughput": prompt_token_throughput,
        "generation_token_throughput": generation_token_throughput,
        "total_token_throughput": total_token_throughput,
        "prefill_prompt_token_throughput": prefill_prompt_token_throughput,
        "decode_generation_token_throughput": decode_generation_token_throughput,
        "ttft_mean": ttft_mean,
        "ttft_p99": ttft_p99,
        "e2e_mean": e2e_mean,
        "e2e_p99": e2e_p99,
        "kv_mean": kv_mean,
        "kv_peak": kv_peak,
        "running_peak": running_peak,
        "waiting_peak": waiting_peak,
        "prefix_hit": prefix_hit,
        "prompt_hit": prompt_hit,
    }


def write_plotly_report(series, out_path, title, embed_plotly=False):
    x = series["x"]
    rows = 8 if series["has_pd"] else 7

    subplot_titles = [
        "Request Throughput (req/s)",
        "Overall Token Throughput (tok/s)",
    ]

    if series["has_physical_pd"]:
        subplot_titles.append("4P4D Physical Prefill / Decode Token Throughput (tok/s)")
    else:
        subplot_titles.append("Prefill-like / Decode-like Token Throughput Proxy (tok/s)")

    subplot_titles += [
        "TTFT (mean & p99, s)",
        "E2E (mean & p99, s)",
        "KV Cache (mean & peak, %)",
        "Running / Waiting Peak (requests)",
        "Prefix / Prompt Cache Hit Rates",
    ]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=subplot_titles)

    r = 1

    fig.add_trace(go.Scatter(x=x, y=series["request_throughput"], mode="lines+markers", name="request_throughput"), row=r, col=1)
    r += 1

    fig.add_trace(go.Scatter(x=x, y=series["prompt_token_throughput"], mode="lines+markers", name="overall_prompt_tok_s"), row=r, col=1)
    fig.add_trace(go.Scatter(x=x, y=series["generation_token_throughput"], mode="lines+markers", name="overall_generation_tok_s"), row=r, col=1)
    r += 1

    if series["has_pd"]:
        fig.add_trace(go.Scatter(x=x, y=series["prefill_prompt_token_throughput"], mode="lines+markers", name="prefill_prompt_tok_s"), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=series["decode_generation_token_throughput"], mode="lines+markers", name="decode_generation_tok_s"), row=r, col=1)
        r += 1

    fig.add_trace(go.Scatter(x=x, y=series["ttft_mean"], mode="lines+markers", name="ttft_mean"), row=r, col=1)
    fig.add_trace(go.Scatter(x=x, y=series["ttft_p99"], mode="lines+markers", name="ttft_p99"), row=r, col=1)
    r += 1

    fig.add_trace(go.Scatter(x=x, y=series["e2e_mean"], mode="lines+markers", name="e2e_mean"), row=r, col=1)
    fig.add_trace(go.Scatter(x=x, y=series["e2e_p99"], mode="lines+markers", name="e2e_p99"), row=r, col=1)
    r += 1

    fig.add_trace(go.Scatter(x=x, y=series["kv_mean"], mode="lines+markers", name="kv_mean"), row=r, col=1)
    fig.add_trace(go.Scatter(x=x, y=series["kv_peak"], mode="lines+markers", name="kv_peak"), row=r, col=1)
    r += 1

    fig.add_trace(go.Scatter(x=x, y=series["running_peak"], mode="lines+markers", name="running_peak"), row=r, col=1)
    fig.add_trace(go.Scatter(x=x, y=series["waiting_peak"], mode="lines+markers", name="waiting_peak"), row=r, col=1)
    r += 1

    fig.add_trace(go.Scatter(x=x, y=series["prefix_hit"], mode="lines+markers", name="prefix_hit_rate"), row=r, col=1)
    fig.add_trace(go.Scatter(x=x, y=series["prompt_hit"], mode="lines+markers", name="prompt_cache_hit_rate"), row=r, col=1)

    fig.update_layout(height=1900 if series["has_pd"] else 1650, width=1000, title_text=title)
    fig.update_xaxes(title_text="Concurrency", row=rows, col=1)

    include_plotlyjs = True if embed_plotly else "cdn"
    fig.write_html(out_path, include_plotlyjs=include_plotlyjs)
    print(f"Wrote interactive report: {out_path}")


def write_matplotlib_report(series, out_path, title):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; cannot produce fallback PNG.")
        return

    x = series["x"]
    nrows = 8 if series["has_pd"] else 7
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 3.2 * nrows), sharex=True)

    r = 0

    axes[r].plot(x, series["request_throughput"], marker="o")
    axes[r].set_ylabel("Throughput\n(req/s)")
    axes[r].set_title("Request Throughput")
    annotate_points(axes[r], x, series["request_throughput"], "{:.2f}")
    r += 1

    axes[r].plot(x, series["prompt_token_throughput"], marker="o", label="overall prompt tok/s")
    axes[r].plot(x, series["generation_token_throughput"], marker="o", label="overall generation tok/s")
    axes[r].legend()
    axes[r].set_ylabel("Overall Token\n(tok/s)")
    axes[r].set_title("Overall Token Throughput")
    annotate_points(axes[r], x, series["generation_token_throughput"], "{:.0f}")
    r += 1

    if series["has_pd"]:
        axes[r].plot(x, series["prefill_prompt_token_throughput"], marker="o", label="prefill / prompt tok/s")
        axes[r].plot(x, series["decode_generation_token_throughput"], marker="o", label="decode / generation tok/s")
        axes[r].legend()
        axes[r].set_ylabel("PD Token\n(tok/s)")
        if series["has_physical_pd"]:
            axes[r].set_title("4P4D Physical Prefill / Decode Token Throughput")
        else:
            axes[r].set_title("Prefill-like / Decode-like Token Throughput Proxy")
        annotate_points(axes[r], x, series["prefill_prompt_token_throughput"], "{:.0f}")
        annotate_points(axes[r], x, series["decode_generation_token_throughput"], "{:.0f}", y_offset=-12)
        r += 1

    axes[r].plot(x, series["ttft_mean"], marker="o", label="mean")
    axes[r].plot(x, series["ttft_p99"], marker="o", label="p99")
    axes[r].legend()
    axes[r].set_ylabel("TTFT\n(seconds)")
    axes[r].set_title("Time To First Token")
    annotate_points(axes[r], x, series["ttft_mean"], "{:.2f}")
    annotate_points(axes[r], x, series["ttft_p99"], "{:.2f}", y_offset=-12)
    r += 1

    axes[r].plot(x, series["e2e_mean"], marker="o", label="mean")
    axes[r].plot(x, series["e2e_p99"], marker="o", label="p99")
    axes[r].legend()
    axes[r].set_ylabel("E2E\n(seconds)")
    axes[r].set_title("End-to-End Latency")
    annotate_points(axes[r], x, series["e2e_mean"], "{:.2f}")
    annotate_points(axes[r], x, series["e2e_p99"], "{:.2f}", y_offset=-12)
    r += 1

    axes[r].plot(x, series["kv_mean"], marker="o", label="kv_mean")
    axes[r].plot(x, series["kv_peak"], marker="o", label="kv_peak")
    axes[r].legend()
    axes[r].set_ylabel("KV Cache\n(%)")
    axes[r].set_title("KV Cache Usage")
    annotate_points(axes[r], x, series["kv_mean"], "{:.2f}%")
    annotate_points(axes[r], x, series["kv_peak"], "{:.2f}%", y_offset=-12)
    r += 1

    axes[r].plot(x, series["running_peak"], marker="o", label="running")
    axes[r].plot(x, series["waiting_peak"], marker="o", label="waiting")
    axes[r].legend()
    axes[r].set_ylabel("Requests")
    axes[r].set_title("Running / Waiting Requests Peak")
    annotate_points(axes[r], x, series["running_peak"], "{:.0f}")
    annotate_points(axes[r], x, series["waiting_peak"], "{:.0f}", y_offset=-12)
    r += 1

    axes[r].plot(x, series["prefix_hit"], marker="o", label="prefix_hit")
    axes[r].plot(x, series["prompt_hit"], marker="o", label="prompt_hit")
    axes[r].legend()
    axes[r].set_ylabel("Hit Rate\n(%)")
    axes[r].set_title("Prefix / Prompt Cache Hit Rate")
    axes[r].set_xlabel("Concurrency")
    annotate_points(axes[r], x, series["prefix_hit"], "{:.2f}%")
    annotate_points(axes[r], x, series["prompt_hit"], "{:.2f}%", y_offset=-12)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=160)
    print(f"Wrote static PNG report: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", "-p", nargs="+", default=["results/*_c*_metrics.json"])
    ap.add_argument("--out-html", default=None)
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--title", default="vLLM Benchmark Report")
    ap.add_argument("--embed-plotly", action="store_true")
    ap.add_argument("--only-png", action="store_true")
    args = ap.parse_args()

    items = load_results(args.pattern)
    if not items:
        print(f"No results found for patterns: {args.pattern}")
        sys.exit(2)

    series = extract_series(items)

    out_html = args.out_html or "results/benchmark_report.html"
    out_png = args.out_png or "results/benchmark_report.png"

    if args.only_png:
        write_matplotlib_report(series, out_png, args.title)
        return

    if HAVE_PLOTLY:
        try:
            write_plotly_report(series, out_html, args.title, embed_plotly=args.embed_plotly)
            return
        except Exception as e:
            print("Plotly plotting failed:", e)

    write_matplotlib_report(series, out_png, args.title)


if __name__ == "__main__":
    main()