"""Gemeinsame Fixtures: die Handbeispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hfa_dynamics as D  # noqa: E402


@pytest.fixture
def kalman():
    """Kalmans Gegenbeispiel, c = 2/3, ε = 10⁻³, R = 10⁻³ (Rawlings/Mayne/Diehl 1.3.4)."""
    return D.build_plant("kalman", 1e-3)


@pytest.fixture
def wagen():
    """Der Wagen mit R = 1: von Hand K_1 = (0, 1/2), K_2 = (2/7, 1)."""
    return D.build_plant("wagen", 1.0)


@pytest.fixture
def vier_tank():
    return D.build_plant("vier_tank", 1e-2)
