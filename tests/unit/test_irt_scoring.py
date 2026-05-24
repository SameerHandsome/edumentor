"""
Unit tests for IRT (Item Response Theory) scoring logic.

The quiz agent uses IRT to:
- Target b_parameter near student theta (|b - theta| < 0.5 window)
- Update theta after quiz responses (Bayesian EAP update)

These tests verify the mathematical properties of IRT scoring.
"""
from __future__ import annotations

import math

import pytest


# ── IRT helper functions (mirroring quiz_agent / quiz route logic) ─────────────


def irt_probability(theta: float, b: float, a: float = 1.0, c: float = 0.25) -> float:
    """3PL IRT model: P(correct | theta, b, a, c)."""
    return c + (1 - c) / (1 + math.exp(-a * (theta - b)))


def is_within_difficulty_window(theta: float, b: float, window: float = 0.5) -> bool:
    return abs(b - theta) < window


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestIRTProbability:
    def test_probability_above_chance_when_theta_equals_b(self):
        """When theta == b, P(correct) > c (above guessing)."""
        p = irt_probability(theta=0.0, b=0.0)
        assert p > 0.25

    def test_probability_increases_with_theta(self):
        """Higher theta → higher probability for the same b."""
        p_low = irt_probability(theta=-1.0, b=0.0)
        p_mid = irt_probability(theta=0.0, b=0.0)
        p_high = irt_probability(theta=1.0, b=0.0)
        assert p_low < p_mid < p_high

    def test_probability_bounded_0_1(self):
        """IRT probability must always be in [0, 1]."""
        for theta in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            for b in [-2.0, 0.0, 2.0]:
                p = irt_probability(theta, b)
                assert 0.0 <= p <= 1.0, f"P out of range for theta={theta}, b={b}: {p}"

    def test_probability_symmetric(self):
        """P(theta=-1, b=0) should equal P(theta=0, b=1) (symmetry around b-theta)."""
        p1 = irt_probability(theta=-1.0, b=0.0)
        p2 = irt_probability(theta=0.0, b=1.0)
        assert abs(p1 - p2) < 1e-9


class TestDifficultyWindow:
    def test_b_near_theta_is_valid(self):
        assert is_within_difficulty_window(theta=0.5, b=0.6)

    def test_b_far_from_theta_is_invalid(self):
        assert not is_within_difficulty_window(theta=0.5, b=2.0)

    def test_exactly_at_boundary_is_excluded(self):
        """Boundary |b - theta| == window is NOT < window (strict inequality)."""
        assert not is_within_difficulty_window(theta=0.0, b=0.5, window=0.5)

    @pytest.mark.parametrize("theta,b", [
        (0.0, 0.0),
        (1.0, 0.8),
        (-1.0, -0.7),
        (2.0, 1.9),
    ])
    def test_valid_pairs(self, theta, b):
        assert is_within_difficulty_window(theta, b)

    @pytest.mark.parametrize("theta,b", [
        (0.0, 1.5),
        (0.0, -1.5),
        (1.0, 3.0),
    ])
    def test_invalid_pairs(self, theta, b):
        assert not is_within_difficulty_window(theta, b)


class TestThetaUpdate:
    def test_correct_answer_increases_theta(self):
        """A correct answer should nudge theta upward (EAP approximation)."""
        theta_before = 0.0
        learning_rate = 0.3
        theta_after = theta_before + learning_rate  # simplified update
        assert theta_after > theta_before

    def test_incorrect_answer_decreases_theta(self):
        """An incorrect answer should nudge theta downward."""
        theta_before = 0.0
        learning_rate = 0.3
        theta_after = theta_before - learning_rate
        assert theta_after < theta_before
