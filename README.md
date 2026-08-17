# Dynamic multi-robot task allocation

A graph neural network scores robot and task pairs and an exact Hungarian solve commits the assignment. The scorer is trained first by imitation of an anticipative MILP expert, then against realised cost by policy gradient.

## Notebooks

`notebooks/01_regeneration.ipynb` regenerates every table and figure in the report from the cached results in `provenance/`. No solver, no training, about three seconds.

`notebooks/02_demonstration.ipynb` runs the full pipeline at reduced scale. About eleven minutes and needs a Gurobi licence.

## Setup

```
conda env create -f environment.yml
conda activate coaml
PYTHONPATH="$PWD/src" python -m pytest tests/ -q
```

Python 3.11.13 and PyTorch 2.2.2. On Apple silicon this runs under x86 translation, so absolute timings are higher than native hardware would give. All policies were measured the same way, so the comparisons hold.

## Layout

- `src/` the method, being the generator, simulator, MILP, both trainers, scorers and evaluators
- `provenance/` cached results behind every reported number
- `scripts/regenerate/` the code that produced those files, most of which needs the instance cache or a checkpoint and so cannot run from a clone
- `scripts/experiments/` the individual experiment runners, included so each number has a visible producer
- `scripts/analysis/` shared reporting and statistics helpers
- `figures/` plotting code and the two TikZ schematics
- `sweep/` configuration and entry point for a single training run

Instances are synthetic. Training seeds 10000 to 10999, validation 11000 to 11199, test 11200 to 11399. The test split was evaluated once, after all design decisions were fixed.
