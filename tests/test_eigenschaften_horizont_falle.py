"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für das Verfahren gelten müssen, ohne das Ergebnis zu kennen; je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed.

Zufallssysteme hier: offen stabile A (auf ρ = 0,9 skaliert, damit jede Zustandsgewichtung auch singulär entdeckbar ist), steuerbar geprüft. Kein Test auf „ρ_N fällt mit N“: das gilt nicht
allgemein (P_f = 0,5·P_∞ ist ein benannter Gegenfall, unten)."""

import numpy as np
import pytest

import hfa_dynamics as D
import hfa_random as Rz
from hfa_rng import SplitMix64


def _random_plant(rng, n_range=(2, 5), singular_q=False):
    n = int(rng.integers(*n_range))
    m = int(rng.integers(1, 3))
    while True:
        A = rng.normal(size=(n, n))
        A *= 0.9 / max(abs(np.linalg.eigvals(A)))
        B = rng.normal(size=(n, m))
        try:
            D.check_controllable(A, B)
            break
        except ValueError:
            continue
    L = rng.normal(size=(n, n))
    if singular_q:
        L[-1] = 0.0                                                    # Q hat Rang n − 1
    Q = L.T @ L
    R = np.eye(m) * float(10 ** rng.uniform(-2, 1))
    return D.Plant("zufall", A, B, Q, R)


def test_terminal_weight_dare_gives_the_same_gain_for_every_horizon_on_50_random_systems_including_singular_q():
    rng = np.random.default_rng(11)
    for i in range(50):
        plant = _random_plant(rng, singular_q=(i % 2 == 1))
        P, K = D.dare(plant)
        gains, _ = D.horizon_gains(plant, P, 12)
        scale = max(1.0, float(np.max(np.abs(K))))
        assert max(float(np.max(np.abs(Kn - K))) for Kn in gains) / scale < 1e-7, i


def test_scaling_q_r_and_terminal_weight_by_a_positive_factor_leaves_the_gain_unchanged():
    rng = np.random.default_rng(12)
    for _ in range(30):
        plant = _random_plant(rng)
        Pf = plant.Q + np.eye(plant.n)
        c = float(10 ** rng.uniform(-2, 2))
        scaled = D.Plant("skaliert", plant.A, plant.B, c * plant.Q, c * plant.R)
        for N in (1, 4, 9):
            K1 = D.horizon_gains(plant, Pf, N)[0][N - 1]
            K2 = D.horizon_gains(scaled, c * Pf, N)[0][N - 1]
            assert np.max(np.abs(K1 - K2)) < 1e-8 * max(1.0, float(np.max(np.abs(K1))))


def test_horizon_n_with_terminal_zero_equals_horizon_n_minus_one_with_terminal_q():
    rng = np.random.default_rng(13)
    for _ in range(30):
        plant = _random_plant(rng)
        g0, _ = D.horizon_gains(plant, np.zeros((plant.n, plant.n)), 8)
        gq, _ = D.horizon_gains(plant, plant.Q, 8)
        assert np.max(np.abs(g0[0])) == 0.0                            # K_1(P_f = 0) = 0: nichts tun
        for N in range(2, 9):
            assert np.max(np.abs(g0[N - 1] - gq[N - 2])) < 1e-12 * max(1.0, float(np.max(np.abs(gq[N - 2]))))


def test_a_change_of_coordinates_leaves_rho_unchanged():
    rng = np.random.default_rng(14)
    for _ in range(30):
        plant = _random_plant(rng)
        T = rng.normal(size=(plant.n, plant.n)) + 3.0 * np.eye(plant.n)
        Ti = np.linalg.inv(T)
        moved = D.Plant("neu", Ti @ plant.A @ T, Ti @ plant.B, T.T @ plant.Q @ T, plant.R)
        Pf = plant.Q + 0.5 * np.eye(plant.n)
        r1 = D.rho_curve(plant, Pf, 6)
        r2 = D.rho_curve(moved, T.T @ Pf @ T, 6)
        assert np.max(np.abs(r1 - r2)) < 1e-6 * max(1.0, float(np.max(r1)))


def test_zero_state_weight_and_zero_terminal_weight_give_zero_gain_and_the_open_loop_stays():
    rng = np.random.default_rng(15)
    for _ in range(20):
        plant = _random_plant(rng)
        none = D.Plant("ohne Gewicht", plant.A, plant.B, np.zeros((plant.n, plant.n)), plant.R)
        gains, _ = D.horizon_gains(none, np.zeros((plant.n, plant.n)), 5)
        assert all(np.max(np.abs(K)) == 0.0 for K in gains)
        assert D.rho_of_gain(none, gains[2]) == pytest.approx(D.rho_open(plant), rel=1e-9)


def test_hard_terminal_condition_with_horizon_equal_to_state_dimension_is_deadbeat():
    rng = np.random.default_rng(16)
    count = 0
    while count < 20:
        n = int(rng.integers(2, 4))
        A, B = rng.normal(size=(n, n)), rng.normal(size=(n, 1))
        if abs(np.linalg.det(np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)]))) < 1e-2:
            continue
        plant = D.Plant("zufall", A, B, np.eye(n), np.array([[1.0]]))
        K = D.hard_terminal_gain(plant, n)
        assert K is not None
        Ac = A - B @ K
        assert np.max(np.abs(np.linalg.matrix_power(Ac, n))) < 1e-6 * max(1.0, float(np.max(np.abs(A))) ** n)   # (A − BK)^n = 0: Deadbeat
        count += 1


def test_same_zufalls_seed_gives_the_same_output():
    assert Rz.count_systems(5, 120) == Rz.count_systems(5, 120)
    a, b = SplitMix64(7), SplitMix64(7)
    assert [a.normal() for _ in range(10)] == [b.normal() for _ in range(10)]
    assert Rz.count_systems(5, 120) != Rz.count_systems(6, 120)


def test_named_counterexample_rho_does_not_fall_with_n_for_half_the_riccati_solution(kalman):
    """P_f = 0,5·P_∞ (alles stabil): ρ(1) = 0,664 < ρ(2) = 0,920; ein längerer Horizont gibt hier nicht den kleineren Spektralradius."""
    P, _ = D.dare(kalman)
    rho = D.rho_curve(kalman, 0.5 * P, 12)
    assert D.unstable_horizons(rho) == []
    assert rho[0] == pytest.approx(0.664, abs=5e-4) and rho[1] == pytest.approx(0.920, abs=5e-4) and rho[1] > rho[0]
