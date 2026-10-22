# Figures — three parallel implementations

The same two schematics exist in three forms, on purpose: each one is the right
tool in a different situation.  Nothing here is generated at build time, and no
figure depends on a network service.

| form | source | renderer | when to use it |
|---|---|---|---|
| **raster / repository** | `plot_architecture.py`, `plot_loss.py`, `plot_trajectory_cases.py`, `plot_acoat_block.py`, `plot_pareto.py` | matplotlib (Agg) and Plotly+Kaleido for the 3-D panels | what the README and the paper on GitHub show; reproducible from a clean checkout |
| **Graphviz** | `plot_model_dot.py` → `model_schematic.dot` | `dot` (Graphviz) | the STRUCTURE lives in text and the layout is computed: easy to diff, no coordinates in the source |
| **LaTeX / TikZ** | `model_schematic_tikz.tex`, `hybrid_loss_tikz.tex` | `pdflatex` (MiKTeX/TeX Live) | the version for a manuscript: vector, inherits the document's fonts, editable like any LaTeX |

## Building each one

```powershell
# raster (no external dependency)
python plot_architecture.py            # model schematic
python plot_loss.py                    # hybrid objective
python plot_acoat_block.py             # ACoAtt-V2 block
python plot_bench_10s.py               # the comparison sheets of the 10 s benchmark
python plot_trajectory_cases.py --config config_365d_10s_bench.json `
    --model "AeroTraj-Ultra (ours)=config_365d_10s_bench.json=checkpoints_bench_10s/ours.pt" `
    --model "FlightBERT++ V2=config_365d_10s_bench.json=checkpoints_bench_10s/fbpp_v2.pt" `
    --model "FlightBERT=config_365d_10s_bench.json=checkpoints_bench_10s/flightbert.pt"

# Graphviz (winget install Graphviz.Graphviz)
python plot_model_dot.py               # writes .dot + .png + .svg

# LaTeX (winget install MiKTeX.MiKTeX, then allow package auto-install)
cd docs/figures
pdflatex model_schematic_tikz.tex
pdflatex hybrid_loss_tikz.tex
```

## The comparison sheets of the 365-day / 10 s benchmark

`plot_bench_10s.py` reads `results_bench_10s/benchmark_results.json` (the test
metrics of every model trained on the same data with the same budget) and writes
four sheets into `docs/figures/`:

| sheet | what it answers |
|---|---|
| `bench_10s_horizon.png` | the error per horizon, and the error relative to the best model of that horizon: **who wins where** |
| `bench_10s_ranking.png` | the overall ranking, with the parameters and the training time of each model: **accuracy against cost** |
| `bench_10s_gain.png` | the error reduction with respect to the best classical filter, per horizon: **what the learning buys, and where** |
| `bench_10s_table.png` | the comparison table, rendered: the sheet to drop into a slide |

The full-budget reference run (180M samples) is drawn as a dashed line and is
*not* part of the same-budget comparison: it is there to show what the budget
costs.

## Why the 3-D panels are in Plotly

`plot_trajectory_cases.py` uses Plotly + Kaleido (which renders through the system
Chrome).  Two reasons, both measured:

* matplotlib's default 3-D look (grey panes, crowded ticks) was the weakest figure
  in the report;
* more importantly, the published figures of this field put **degrees** on two axes
  and **metres** on the third, which compresses a real trajectory into a blob.  The
  panels here use a **local metric frame** anchored at the last observed position,
  so all three axes are in metres and each panel is a geometrically honest view.

## Notes on the LaTeX route

* `matplotlib`'s PGF backend needs `pdflatex` present *even to write the .pgf
  source*, so `--format pgf` only works once a TeX distribution is installed.
  `plot_architecture.py --format pgf` and `plot_loss.py --format pgf` emit `.pgf`
  (low-level PGF paths, not hand-written TikZ).
* The two `.tex` files here are hand-written TikZ: nodes, arrows, `fit`-based stage
  containers and the bit-vector macro.  They need only `tikz` with
  `positioning, arrows.meta, fit, backgrounds, calc`, `lmodern` and `amsmath`, and
  they produce a tight PDF through the `standalone` class.
* A bare `>` inside a Graphviz **HTML-like label** terminates the label, so every
  `->` must be written `-&gt;`; this is the one non-obvious rule in
  `plot_model_dot.py`.
