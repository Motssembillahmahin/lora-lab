"""Unit tests for the eval aggregation (ADR 0004).

The harness reports a corpus-level response-NLL: each example contributes its
mean response-NLL weighted by how many response tokens it scored, so the result
is sum(NLL) / sum(tokens) over the whole eval set — not a mean-of-means. That
token weighting is the point (see docs/math/03 §2 on the loss denominator).
"""

import math

import pytest

from src.eval import weighted_mean


def test_token_weighting_is_not_mean_of_means():
    # Two examples: NLL 1.0 over 1 token, NLL 3.0 over 3 tokens.
    # Corpus mean = (1*1 + 3*3) / (1+3) = 10/4 = 2.5, NOT (1+3)/2 = 2.0.
    assert weighted_mean([1.0, 3.0], [1, 3]) == 2.5


def test_equal_weights_equals_plain_mean():
    assert weighted_mean([2.0, 4.0, 6.0], [5, 5, 5]) == pytest.approx(4.0)


def test_single_value():
    assert weighted_mean([1.7], [42]) == pytest.approx(1.7)


def test_zero_total_weight_returns_zero():
    # No scored tokens anywhere -> define the corpus loss as 0.0, never 0/0 NaN.
    result = weighted_mean([], [])
    assert result == 0.0
    assert not math.isnan(result)
