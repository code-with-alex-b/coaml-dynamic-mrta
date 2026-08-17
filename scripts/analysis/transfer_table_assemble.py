"""Phase 3 timing summary and Phase 4 assembly of the zero-shot transfer table.

Reads ``phase0_inventory.json``, ``phase2_statistics.json`` and the per-scale
``timing_*.csv`` files, and writes the assembled table as markdown and LaTeX
plus a per-stage timing summary.

The table groups the three ratio-3.0 rows as one size-scaling series and keeps
the ratio-6.0 density row apart, so the two are not read as a single series.
Only the R=6, T=18 row has an offline anticipative MILP stored per record, so
it is the only row carrying a gap closure; every other row carries cost ratios
against the two Hungarian variants and no closure.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STAGES = [
    ("t_features_s", "feature construction"),
    ("t_graph_build_s", "graph build"),
    ("t_gnn_forward_s", "GNN forward"),
    ("t_hungarian_s", "Hungarian solve"),
    ("t_stages_sum_s", "sum of stages"),
    ("t_total_s", "whole decision"),
]


def proc_translated_in_process() -> str:
    """Read sysctl.proc_translated from inside this process via libc.

    A subprocess would report its own translation state, so the value is read
    with sysctlbyname in-process to be certain it describes the interpreter
    that produced the timings.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        out = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(out))
        rc = libc.sysctlbyname(
            b"sysctl.proc_translated", ctypes.byref(out),
            ctypes.byref(size), None, ctypes.c_size_t(0),
        )
        if rc != 0:
            return "unavailable (sysctl returned nonzero, name not present)"
        return str(out.value)
    except Exception as e:  # pragma: no cover
        return f"unavailable ({e})"


def timing_summary(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open()))
    seeds = {r["seed"] for r in rows}
    out = {
        "n_decisions": len(rows),
        "n_instances": len(seeds),
        "mean_decisions_per_instance": len(rows) / len(seeds),
        "stages": {},
    }
    for key, label in STAGES:
        a = np.array([float(r[key]) for r in rows]) * 1e6
        out["stages"][label] = {
            "median_us": float(np.median(a)),
            "mean_us": float(a.mean()),
            "p5_us": float(np.percentile(a, 5)),
            "p25_us": float(np.percentile(a, 25)),
            "p75_us": float(np.percentile(a, 75)),
            "p95_us": float(np.percentile(a, 95)),
        }
    tot = np.array([float(r["t_total_s"]) for r in rows])
    out["per_instance_total_s"] = float(tot.mean() * out["mean_decisions_per_instance"])
    d = np.array([float(r["t_discrepancy_s"]) for r in rows]) * 1e6
    out["discrepancy_us"] = {"median": float(np.median(d)), "mean": float(d.mean())}
    return out


def _iv(pair, prec=4):
    lo, hi = pair
    return f"[{lo:.{prec}f}, {hi:.{prec}f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = REPO_ROOT / args.dir

    inv = json.load((d / "phase0_inventory.json").open())
    stats = json.load((d / "phase2_statistics.json").open())["results"]
    by_key = {r["scale"]["key"]: r for r in stats}

    timings = {}
    for s in inv["scales"]:
        p = d / f"timing_{s['key']}.csv"
        if p.exists():
            timings[s["key"]] = timing_summary(p)

    env = {
        "proc_translated_in_process": proc_translated_in_process(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu (map_location='cpu', no .to(device) in the rollout path)",
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    }
    for name, key in [("chip", "machdep.cpu.brand_string"), ("logical_cores", "hw.ncpu")]:
        env[name] = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()

    out = {"environment": env, "timings": timings}
    tp = d / "phase3_timing_summary.json"
    if tp.exists():
        raise SystemExit(f"REFUSING to overwrite {tp}")
    json.dump(out, tp.open("w"), indent=2)

    order = [("size", "Ratio-matched size scaling (tasks per robot = 3.0)"),
             ("density", "Density change (tasks per robot = 6.0)")]
    md = []
    md.append("# Zero-shot transfer table\n")
    md.append(f"Checkpoint `{inv['checkpoint']}`, sha256 "
              f"`{inv['checkpoint_sha256_start']}`, unchanged at end "
              f"(`{inv['checkpoint_sha256_end']}`). Hard decode, epsilon 0, "
              f"inference only, no retraining.\n")
    md.append("Cells carry the point estimate and a 95 per cent bootstrap "
              "interval over 10,000 paired resamples, seed 20260731. The cost "
              "ratio is policy cost over baseline cost, so below 1.0 favours "
              "the policy.\n")

    hdr = ("| Scale | n | Serve-all | Cost ratio vs distance-only Hungarian | "
           "Cost ratio vs kappa-weighted Hungarian | Gap closure vs MILP | "
           "Median per-decision latency |")
    sep = "|---|---|---|---|---|---|---|"

    for group, title in order:
        md.append(f"\n**{title}**\n")
        md.append(hdr)
        md.append(sep)
        for s in inv["scales"]:
            if s["group"] != group:
                continue
            r = by_key[s["key"]]
            vb = r["vs_baselines"]
            g = vb["distance-only Hungarian"]
            k = vb["kappa-weighted Hungarian"]
            gc = r["gap_closure"]
            if gc is None:
                gcs = "not defined (no benchmark)"
            else:
                m = gc["vs_milp"]
                gcs = (f"{m['ratio_of_means']:.4f} "
                       f"{_iv(m['ratio_of_means_ci95'])} (n={m['n']})")
            t = timings.get(s["key"])
            ts = (f"{t['stages']['whole decision']['median_us']:.0f} us"
                  if t else "not measured")
            md.append(
                f"| R={s['R']}, T={s['T']} | "
                f"{r['n_instances_in_statistics']}/{r['n_instances_total']} | "
                f"{s['n_serve_all']}/{s['n_instances']} | "
                f"{g['ratio_of_means']:.4f} {_iv(g['ratio_of_means_ci95'])} | "
                f"{k['ratio_of_means']:.4f} {_iv(k['ratio_of_means_ci95'])} | "
                f"{gcs} | {ts} |"
            )

    md.append(
        "\nCaption. Zero-shot transfer of the checkpoint trained at R=6, T=18. "
        "The first block holds the tasks-per-robot ratio fixed at 3.0 and "
        "varies size alone; the R=10, T=60 row changes size and density "
        "together and is a separate comparison, not a fourth point on the "
        "same series. Only the R=6, T=18 row has an offline anticipative MILP "
        "benchmark stored per record, so it is the only row for which gap "
        "closure is defined; every other row is a cost ratio against the two "
        "Hungarian variants and no closure is reported for it. n is the "
        "instances entering the statistics over the instances evaluated; an "
        "instance is excluded only when the policy or a baseline failed to "
        "serve every task, and every exclusion is named in the report.\n")

    mp = d / "transfer_table.md"
    if mp.exists():
        raise SystemExit(f"REFUSING to overwrite {mp}")
    mp.write_text("\n".join(md))

    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{lccccc}", r"\toprule",
           r"Scale & $n$ & Serve-all & Ratio vs Hung.\ dist. & "
           r"Ratio vs Hung.\ $\kappa$ & Gap closure vs MILP \\"]
    for group, title in order:
        tex.append(r"\midrule")
        tex.append(r"\multicolumn{6}{l}{\emph{" + title + r"}} \\")
        for s in inv["scales"]:
            if s["group"] != group:
                continue
            r = by_key[s["key"]]
            g = r["vs_baselines"]["distance-only Hungarian"]
            k = r["vs_baselines"]["kappa-weighted Hungarian"]
            gc = r["gap_closure"]
            gcs = (r"\textemdash" if gc is None else
                   f"{gc['vs_milp']['ratio_of_means']:.3f} "
                   f"{{\\scriptsize [{gc['vs_milp']['ratio_of_means_ci95'][0]:.3f}, "
                   f"{gc['vs_milp']['ratio_of_means_ci95'][1]:.3f}]}}")
            tex.append(
                f"$R={s['R']},T={s['T']}$ & "
                f"{r['n_instances_in_statistics']} & "
                f"{s['n_serve_all']}/{s['n_instances']} & "
                f"{g['ratio_of_means']:.3f} {{\\scriptsize "
                f"[{g['ratio_of_means_ci95'][0]:.3f}, {g['ratio_of_means_ci95'][1]:.3f}]}} & "
                f"{k['ratio_of_means']:.3f} {{\\scriptsize "
                f"[{k['ratio_of_means_ci95'][0]:.3f}, {k['ratio_of_means_ci95'][1]:.3f}]}} & "
                f"{gcs} \\\\"
            )
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Zero-shot transfer of the policy trained at $R=6, T=18$. "
            r"Cost ratio is policy over baseline, so below one favours the "
            r"policy; brackets are 95\% bootstrap intervals over 10{,}000 "
            r"paired resamples. Only the $R=6,T=18$ row has a stored offline "
            r"anticipative MILP benchmark and therefore a defined gap closure; "
            r"the remaining rows report cost ratios only. The $R=10,T=60$ row "
            r"changes size and density together and is not a further point on "
            r"the ratio-matched series above it.}",
            r"\label{tab:zero-shot-transfer}", r"\end{table}"]
    xp = d / "transfer_table.tex"
    if xp.exists():
        raise SystemExit(f"REFUSING to overwrite {xp}")
    xp.write_text("\n".join(tex) + "\n")

    print(f"wrote {tp}\nwrote {mp}\nwrote {xp}")

    print("\n=== environment ===")
    for k, v in env.items():
        print(f"  {k:32s} = {v}")
    base = timings.get("r6t18")
    for s in inv["scales"]:
        t = timings.get(s["key"])
        if not t:
            continue
        print(f"\n=== R={s['R']}, T={s['T']} "
              f"({t['n_decisions']} decisions over {t['n_instances']} instances, "
              f"{t['mean_decisions_per_instance']:.2f} decisions/instance) ===")
        print(f"{'stage':24s}{'median':>9s}{'mean':>9s}{'p5':>9s}{'p25':>9s}"
              f"{'p75':>9s}{'p95':>9s}{'x r6t18':>9s}")
        for _, label in STAGES:
            v = t["stages"][label]
            grow = (v["median_us"] / base["stages"][label]["median_us"]
                    if base else float("nan"))
            print(f"{label:24s}{v['median_us']:9.2f}{v['mean_us']:9.2f}"
                  f"{v['p5_us']:9.2f}{v['p25_us']:9.2f}{v['p75_us']:9.2f}"
                  f"{v['p95_us']:9.2f}{grow:9.2f}")
        h = t["stages"]["Hungarian solve"]["median_us"]
        f_ = t["stages"]["GNN forward"]["median_us"]
        print(f"  Hungarian / GNN forward (median) = {h / f_:.3f}")
        print(f"  per-instance total = {t['per_instance_total_s'] * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
