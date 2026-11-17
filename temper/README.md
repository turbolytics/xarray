# xarray compute and invariant map

[Temper](https://github.com/turbolytics/temper) map of what xarray's
reductions promise on each compute path, and of invariants in xarray's own
semantics. `MATRIX.md` at the repository root is the result.

- `lattices/rolling.contract.yml` and `lattices/groupby.contract.yml`: the
  guarantees, checked on each compute path (numpy, numbagg, bottleneck, flox,
  dask). The reference is NumPy applied to each window or group by hand.
- `invariants.yml`: properties checked with hypothesis.
- `subjects/`: one file per compute path, plus `core.dataarray`.
- `../xarray/tests/test_temper_compute_contract.py` and
  `../xarray/tests/test_temper_invariants.py`: the tests. Each one states what
  it proves with `@pytest.mark.covers`.

Run it:

```sh
uv venv
uv pip install -e ".[accel,parallel]" pytest pytest-asyncio pytest-mypy-plugins hypothesis temper-python
brew install turbolytics/tap/temper   # or: go install github.com/turbolytics/temper/cmd/temper@v0.1.2
temper clean
temper run --level ci -- .venv/bin/python -m pytest xarray/tests/test_temper_compute_contract.py xarray/tests/test_temper_invariants.py
temper matrix
```

`temper matrix` exits 1 while any cell is failing.
