"""Unit tests for the seed-study stats helper (ADR 0005)."""

import pytest

from src.study import mean_std


def test_mean_std_basic():
    # population std of [2, 4] = sqrt(((1)^2 + (1)^2)/2) = 1.0
    m, sd = mean_std([2.0, 4.0])
    assert m == pytest.approx(3.0)
    assert sd == pytest.approx(1.0)


def test_mean_std_single_has_zero_spread():
    assert mean_std([5.0]) == (5.0, 0.0)


def test_mean_std_empty_is_zero_not_error():
    assert mean_std([]) == (0.0, 0.0)
