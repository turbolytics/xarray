"""The rolling.contract and groupby.contract lattices (temper/lattices).

Each test states what it proves with @pytest.mark.covers (temper-python).
Without `temper run`, the marks do nothing.

Each compute path runs the same reduction. The reference is NumPy applied to
each window or group by hand, never another xarray path.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.tests import (
    requires_bottleneck,
    requires_dask,
    requires_flox,
    requires_numbagg,
)

OFF = {"use_bottleneck": False, "use_numbagg": False}


def _as_dask(arr: np.ndarray) -> Any:
    import dask.array

    return dask.array.from_array(arr, chunks=2)


# name -> (options, wrap data, extra reduction kwargs, marks)
ROLLING_PATHS: dict[str, tuple[dict, Callable, dict, list]] = {
    "numpy": (OFF, lambda a: a, {}, []),
    "numbagg": ({}, lambda a: a, {}, [requires_numbagg]),
    "bottleneck": (
        {"use_bottleneck": True, "use_numbagg": False},
        lambda a: a,
        {},
        [requires_bottleneck],
    ),
    "dask": ({}, _as_dask, {}, [requires_dask]),
}
GROUPBY_PATHS: dict[str, tuple[dict, Callable, dict, list]] = {
    "numpy": ({"use_flox": False}, lambda a: a, {}, []),
    "flox": ({}, lambda a: a, {}, [requires_flox, requires_numbagg]),
    "flox_numpy": ({}, lambda a: a, {"engine": "numpy"}, [requires_flox]),
    "dask_flox": ({}, _as_dask, {}, [requires_flox, requires_dask]),
}


def params(lattice: str, member: str, paths: dict, feature_member: str) -> list[Any]:
    kind = lattice.split(".", maxsplit=1)[0]
    return [
        pytest.param(
            name,
            id=name,
            marks=[
                *marks,
                pytest.mark.covers(f"{lattice}/{member}", subject=f"{kind}.{name}"),
                *(
                    [pytest.mark.covers(f"{kind}.{name}")]
                    if member == feature_member
                    else []
                ),
            ],
        )
        for name, (_, _, _, marks) in paths.items()
    ]


@contextlib.contextmanager
def path(paths: dict, name: str) -> Iterator[tuple[Callable, dict]]:
    opts, wrap, kwargs, _ = paths[name]
    with xr.set_options(**opts):
        yield wrap, kwargs


def quiet(func: Callable, vals: np.ndarray) -> float:
    """NumPy's reference value, without its all-NaN RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(vals)


def windows(n: int, window: int, center: bool) -> list[slice]:
    left = window // 2 if center else window - 1
    return [slice(max(0, i - left), min(n, i - left + window)) for i in range(n)]


DATA = np.array([1.0, np.nan, 3.0, 4.0, np.nan, np.nan, 7.0, 2.0, -1.0, 5.0])
NAN_FUNCS = {
    "sum": np.nansum,
    "mean": np.nanmean,
    "std": np.nanstd,
    "var": np.nanvar,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
    "count": lambda v: float(np.count_nonzero(~np.isnan(v))),
}


@pytest.mark.parametrize("op", list(NAN_FUNCS))
@pytest.mark.parametrize(
    ("window", "min_periods", "center"), [(3, None, False), (3, 1, False), (4, 2, True)]
)
@pytest.mark.parametrize(
    "name",
    params(
        "rolling.contract",
        "reductions_match_numpy",
        ROLLING_PATHS,
        "reductions_match_numpy",
    ),
)
def test_rolling_reductions_match_numpy(
    name: str, op: str, window: int, min_periods: int | None, center: bool
) -> None:
    mp = window if min_periods is None else min_periods
    want = []
    for s in windows(len(DATA), window, center):
        vals = DATA[s]
        valid = np.count_nonzero(~np.isnan(vals))
        want.append(quiet(NAN_FUNCS[op], vals) if valid >= mp else np.nan)
    with path(ROLLING_PATHS, name) as (wrap, _):
        da = xr.DataArray(wrap(DATA), dims="t")
        got = getattr(
            da.rolling(t=window, min_periods=min_periods, center=center), op
        )().values
    np.testing.assert_allclose(
        got, np.array(want, dtype=float), rtol=1e-12, equal_nan=True
    )


# Each array ties with the value xarray pads windows with: True or False for
# bool, +inf or -inf for floats, and the type's max or min for ints.
TIE_ARRAYS = {
    "bool": np.array([True, True, False, True, False, False]),
    "inf": np.array([np.inf, np.inf, 1.0, 2.0, -np.inf, 3.0]),
    "intmax": np.array([np.iinfo("i8").max, 5, 1, 7, np.iinfo("i8").min, 2]),
}


@pytest.mark.parametrize("arr_name", list(TIE_ARRAYS))
@pytest.mark.parametrize("op", ["argmin", "argmax"])
@pytest.mark.parametrize("cumulative", [False, True])
@pytest.mark.parametrize(
    "name",
    params(
        "rolling.contract",
        "argmin_argmax_match_numpy",
        ROLLING_PATHS,
        "reductions_match_numpy",
    ),
)
def test_rolling_argmin_argmax_match_numpy(
    name: str, arr_name: str, op: str, cumulative: bool
) -> None:
    arr = TIE_ARRAYS[arr_name]
    window = len(arr) if cumulative else 3
    want = [
        float(getattr(np, op)(arr[s])) for s in windows(len(arr), window, center=False)
    ]
    with path(ROLLING_PATHS, name) as (wrap, _):
        da = xr.DataArray(wrap(arr), dims="t")
        roller = (
            da.cumulative("t") if cumulative else da.rolling(t=window, min_periods=1)
        )
        got = getattr(roller, op)().values
    assert got.tolist() == want


LABELS = np.array(["a", "b", "a", "b", "c", "c", "a", "b", "a", "b"])
GROUPS = {g: DATA[LABELS == g] for g in ("a", "b", "c")}
GROUP_FUNCS = {**NAN_FUNCS, "prod": np.nanprod}


@pytest.mark.parametrize("op", list(GROUP_FUNCS))
@pytest.mark.parametrize(
    "name",
    params(
        "groupby.contract",
        "reductions_match_numpy",
        GROUPBY_PATHS,
        "reductions_match_numpy",
    ),
)
def test_groupby_reductions_match_numpy(name: str, op: str) -> None:
    want = [quiet(GROUP_FUNCS[op], GROUPS[g]) for g in ("a", "b", "c")]
    with path(GROUPBY_PATHS, name) as (wrap, kwargs):
        da = xr.DataArray(wrap(DATA), dims="t", coords={"g": ("t", LABELS)})
        got = getattr(da.groupby("g"), op)(**kwargs).compute()
    assert got["g"].values.tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(
        got.values, np.array(want, dtype=float), rtol=1e-12, equal_nan=True
    )


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize(
    "name",
    params(
        "groupby.contract",
        "any_all_match_numpy",
        GROUPBY_PATHS,
        "reductions_match_numpy",
    ),
)
def test_groupby_any_all_match_numpy(name: str, op: str) -> None:
    # Group a is all NaN, b mixes 0.0 and NaN, c mixes 1.0 and 0.0.
    data = np.array([np.nan, 0.0, np.nan, np.nan, 1.0, 0.0])
    labels = np.array(["a", "b", "a", "b", "c", "c"])
    want = [bool(getattr(np, op)(data[labels == g])) for g in ("a", "b", "c")]
    with path(GROUPBY_PATHS, name) as (wrap, kwargs):
        da = xr.DataArray(wrap(data), dims="t", coords={"g": ("t", labels)})
        got = getattr(da.groupby("g"), op)(**kwargs).compute()
    assert got.values.tolist() == want


@pytest.mark.parametrize("op", ["sum", "prod"])
@pytest.mark.parametrize(
    "name",
    params(
        "groupby.contract",
        "min_count_follows_skipna",
        GROUPBY_PATHS,
        "reductions_match_numpy",
    ),
)
def test_groupby_min_count_follows_skipna(name: str, op: str) -> None:
    ints = np.array([1, 2, 3, 4, 5])
    labels = np.array(["a", "a", "b", "b", "c"])
    func = {"sum": np.sum, "prod": np.prod}[op]
    with path(GROUPBY_PATHS, name) as (wrap, kwargs):
        # Integers: skipna defaults to False, so min_count is not used.
        da = xr.DataArray(wrap(ints), dims="t", coords={"g": ("t", labels)})
        got = getattr(da.groupby("g"), op)(min_count=2, **kwargs).compute()
        assert got.dtype == ints.dtype
        assert got.values.tolist() == [
            int(func(ints[labels == g])) for g in ("a", "b", "c")
        ]
        # Floats: skipna defaults to True, so group c (one value) becomes NaN.
        da = xr.DataArray(
            wrap(ints.astype(float)), dims="t", coords={"g": ("t", labels)}
        )
        got = getattr(da.groupby("g"), op)(min_count=2, **kwargs).compute()
        np.testing.assert_array_equal(
            got.values, [func([1.0, 2.0]), func([3.0, 4.0]), np.nan]
        )
