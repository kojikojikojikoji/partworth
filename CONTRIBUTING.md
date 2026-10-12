# Contributing

Issues and pull requests are welcome, especially:

- comparisons with other software (Sawtooth CBC/HB, bayesm, apollo, Biogeme) on data you
  can share;
- public CBC data sets that could become examples;
- reports of convergence problems with a reproducible (simulated or public) example.

## Ground rules

- **Public or simulated data only.** Do not add examples, fixtures or attribute sets
  derived from non-public studies (see `docs/adr/0001-clean-room-and-sources.md`).
- **No GPL code.** Do not port code from bayesm, apollo or other GPL packages. Comparing
  against them from `validation/` in a separate process is fine.
- Record design decisions that change behaviour in a new ADR under `docs/adr/`.

## Development

```
uv venv --python 3.12 && uv pip install -e ".[dev]"
pytest -q                       # fast unit tests
pytest -q --run-slow            # MCMC tests (parameter recovery), a few minutes
pytest -q --run-network         # tests that download the camera data
ruff check src tests validation
```

Validation scripts (need more time; `compare_camera.py` also needs R with bayesm):

```
python validation/compare_camera.py
python validation/sbc.py --reps 200
python validation/performance.py
```

On macOS 26+ with PyTensor 3.3.3 or earlier, set `PYTENSOR_FLAGS=cxx=` (see README).
