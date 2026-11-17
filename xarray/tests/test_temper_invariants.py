"""The invariants in temper/invariants.yml, checked with hypothesis.

Each test states what it proves with @pytest.mark.covers (temper-python).
Without `temper run`, the marks do nothing. derandomize=True keeps every run
on the same examples, and @example pins each known failure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import xarray as xr
from xarray.tests.test_temper_compute_contract import ROLLING_PATHS, path

pytest.importorskip("hypothesis")

import itertools

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=list(HealthCheck),
)
CORE = "core.dataarray"


@st.composite
def arrays_1d(draw, min_size=1, max_size=8, dtypes=("f8", "f4", "i8", "i4", "bool")):
    dt = draw(st.sampled_from(dtypes))
    n = draw(st.integers(min_size, max_size))
    if dt == "bool":
        return np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))
    if dt.startswith("i"):
        info = np.iinfo(dt)
        values = st.integers(int(info.min), int(info.max))
        return np.array(draw(st.lists(values, min_size=n, max_size=n)), dtype=dt)
    values = st.floats(allow_nan=True, allow_infinity=True, width=int(dt[1:]) * 8)
    return np.array(draw(st.lists(values, min_size=n, max_size=n)), dtype=dt)


@st.composite
def labeled(draw):
    arr = draw(arrays_1d())
    coord = draw(
        st.lists(st.integers(0, 20), min_size=len(arr), max_size=len(arr), unique=True)
    )
    return xr.DataArray(arr, dims="x", coords={"x": coord})


@pytest.mark.covers("core.align_then_add", subject=CORE)
@pytest.mark.covers(CORE)
@SETTINGS
@given(labeled(), labeled(), st.sampled_from(["inner", "outer", "left", "right"]))
def test_align_then_add(a, b, join):
    a, b = a.astype("f8"), b.astype("f8")
    with xr.set_options(arithmetic_join=join):
        direct = a + b
    a2, b2 = xr.align(a, b, join=join)
    xr.testing.assert_identical(direct, a2 + b2)


@pytest.mark.covers("core.align_is_idempotent", subject=CORE)
@SETTINGS
@given(labeled(), labeled())
def test_align_is_idempotent(a, b):
    a2, b2 = xr.align(a, b, join="outer")
    a3, b3 = xr.align(a2, b2, join="outer")
    xr.testing.assert_identical(a2, a3)
    xr.testing.assert_identical(b2, b3)


@pytest.mark.covers("core.concat_of_split", subject=CORE)
@SETTINGS
@given(labeled(), st.data())
def test_concat_of_split(a, data):
    n = a.sizes["x"]
    cuts = st.lists(st.integers(1, max(1, n - 1)), max_size=3, unique=True)
    bounds = [0, *sorted(data.draw(cuts) if n > 1 else []), n]
    parts = [
        a.isel(x=slice(lo, hi)) for lo, hi in itertools.pairwise(bounds) if hi > lo
    ]
    xr.testing.assert_identical(xr.concat(parts, "x"), a)


@pytest.mark.covers("core.reduce_ignores_transpose", subject=CORE)
@SETTINGS
@given(
    arrays_1d(max_size=6),
    st.integers(1, 3),
    st.sampled_from(["sum", "mean", "max", "min", "std", "argmin", "argmax"]),
)
def test_reduce_ignores_transpose(row, ncols, op):
    da = xr.DataArray(np.stack([row] * ncols, axis=1), dims=("x", "y"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_array_equal(
            getattr(da, op)("x").values, getattr(da.T, op)("x").values
        )


@pytest.mark.covers("core.groupby_map_identity", subject=CORE)
@SETTINGS
@given(labeled(), st.lists(st.sampled_from("abc"), min_size=1, max_size=8))
def test_groupby_map_identity(a, labs):
    a = a.assign_coords(g=("x", (labs * len(a))[: len(a)]))
    xr.testing.assert_identical(a.groupby("g").map(lambda d: d), a)


@pytest.mark.covers("core.keep_attrs", subject=CORE)
@SETTINGS
@given(
    labeled(),
    st.sampled_from(["sum", "mean", "max", "min", "std", "median", "count", "cumsum"]),
)
def test_keep_attrs(a, op):
    a = a.assign_attrs(units="m", note="x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert getattr(a, op)("x", keep_attrs=True).attrs == a.attrs


@pytest.mark.covers("core.sortby_reindex", subject=CORE)
@SETTINGS
@given(labeled())
def test_sortby_reindex(a):
    xr.testing.assert_identical(a.sortby("x").reindex(x=a.x), a)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            id=name,
            marks=[
                *marks,
                pytest.mark.covers(
                    "rolling.equals_construct", subject=f"rolling.{name}"
                ),
            ],
        )
        for name, (_, _, _, marks) in ROLLING_PATHS.items()
    ],
)
@SETTINGS
@given(
    arr=arrays_1d(max_size=10, dtypes=("f8",)),
    window=st.integers(1, 4),
    center=st.booleans(),
    op=st.sampled_from(["sum", "mean", "max", "min", "std", "var", "median"]),
)
# A large value makes later windows lose small values.
@example(arr=np.array([1e16, 1.0, 1.0, 1.0]), window=1, center=False, op="sum")
# An infinite value makes later windows NaN (GH #8811).
@example(arr=np.array([np.inf, 0.0, 1.0]), window=1, center=False, op="sum")
def test_rolling_equals_construct(name, arr, window, center, op):
    window = min(window, len(arr))
    with path(ROLLING_PATHS, name) as (wrap, _):
        da = xr.DataArray(wrap(arr), dims="x")
        rolling = da.rolling(x=window, center=center, min_periods=window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            direct = getattr(rolling, op)().values
            built = getattr(rolling.construct("w"), op)("w", skipna=False).values
    np.testing.assert_allclose(direct, built, rtol=1e-9, equal_nan=True)
