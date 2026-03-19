"""Unit tests for sapientml_core.meta_features helpers.

Regression tests for issue #84: boolean dtype columns passed to
_get_ttest_pvalue / _get_kstest_pvalue / _get_pearsonr_values must not raise
``ValueError: data type <class 'numpy.bool_'> not inexact`` from
numpy.finfo / scipy.stats internals.
"""

import math

import pandas as pd
import pytest

from sapientml_core.meta_features import (
    _get_kstest_pvalue,
    _get_pearsonr_values,
    _get_ttest_pvalue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_col():
    return pd.Series([True, False, True, False, True, False], dtype=bool)


def _str_target():
    return pd.Series(["a", "b", "a", "b", "a", "b"])


def _float_target():
    return pd.Series([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# _get_ttest_pvalue
# ---------------------------------------------------------------------------


def test_ttest_bool_column_does_not_raise():
    """bool column must not raise ValueError (issue #84)."""
    p = _get_ttest_pvalue(_bool_col(), _str_target())
    assert isinstance(p, float)


def test_ttest_bool_column_returns_valid_pvalue():
    p = _get_ttest_pvalue(_bool_col(), _str_target())
    assert 0.0 <= p <= 1.0


def test_ttest_float_column_unchanged():
    c = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    p = _get_ttest_pvalue(c, _str_target())
    assert 0.0 <= p <= 1.0


def test_ttest_single_class_returns_one():
    """When y has only one unique value, p-value should be 1.0."""
    c = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series(["a", "a", "a"])
    assert _get_ttest_pvalue(c, y) == 1.0


# ---------------------------------------------------------------------------
# _get_kstest_pvalue
# ---------------------------------------------------------------------------


def test_kstest_bool_column_does_not_raise():
    """bool column must not raise ValueError (issue #84)."""
    p = _get_kstest_pvalue(_bool_col(), _str_target())
    assert isinstance(p, float)


def test_kstest_bool_column_returns_valid_pvalue():
    p = _get_kstest_pvalue(_bool_col(), _str_target())
    assert 0.0 <= p <= 1.0


def test_kstest_single_class_returns_one():
    c = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series(["a", "a", "a"])
    assert _get_kstest_pvalue(c, y) == 1.0


# ---------------------------------------------------------------------------
# _get_pearsonr_values
# ---------------------------------------------------------------------------


def test_pearsonr_bool_column_does_not_raise():
    """bool column must not raise ValueError (issue #84)."""
    corr, p = _get_pearsonr_values(_bool_col(), _float_target())
    assert isinstance(corr, float)
    assert isinstance(p, float)


def test_pearsonr_bool_column_returns_valid_values():
    corr, p = _get_pearsonr_values(_bool_col(), _float_target())
    assert -1.0 <= corr <= 1.0
    assert 0.0 <= p <= 1.0


def test_pearsonr_perfect_correlation():
    c = pd.Series([0.0, 1.0, 2.0, 3.0])
    y = pd.Series([0.0, 1.0, 2.0, 3.0])
    corr, _ = _get_pearsonr_values(c, y)
    assert math.isclose(corr, 1.0, abs_tol=1e-9)
