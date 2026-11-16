"""Orakel-Tests (ORACLE-PLAYBOOK §5, §6; DEMO-PLAYBOOK §12): die Kernroutinen gegen einen anderen Rechenweg (`tests/oracle_helpers.py`), zusammen unter 10 Sekunden.

1. Riccati-Rekursion gegen die Batch-Lösung (alle Stellwerte gemeinsam, Normalgleichungen), in Gleitkomma (N bis 6) und in Brüchen (exakt gleich).
2. Riccati-Rekursion gegen numerische Minimierung der Horizontkosten über alle Stellfolgen (N = 1 bis 4, BFGS mit analytischem Gradienten über die Adjungierte).
3. ρ aus Eigenwerten gegen die Formel aus Spur und Determinante (n = 2) und gegen das Jury-Kriterium.
4. DARE (SciPy) gegen den Fixpunkt der Riccati-Rekursion.
5. Kosten der unendlichen Zukunft über die Lyapunov-Gleichung gegen eine lange Simulation und gegen eine Winkelsuche des schlechtesten Anfangszustands.
6. Harte Terminalbedingung (KKT) gegen den Grenzwert P_f → ∞ (Band) und gegen den exakten Deadbeat-Fall (test_dynamics.py).
7. Die Vier-Tank-Matrizen als Daten: offen stabil, Eigenwerte sind die Diagonale (die Matrix ist oberes Dreieck).

Toleranzen nach Messung: Riccati gegen Batch relativ 1e-8 (schlechtester gemessener Wert 5e-15 über die fünf Strecken und drei Terminalgewichte), gegen BFGS 1e-6 (gemessen 4e-9); die Referenz ist nie ungenauer als die Toleranz
(ORACLE-PLAYBOOK §2: BFGS mit analytischem Gradienten, gtol 1e-12). Eigenwerte und Lösungen aus SciPy nur als Band, nie bitgleich."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import hfa_dynamics as D
import hfa_exact as X
import oracle_helpers as O


def _plants():
    return [D.build_plant("kalman", 1e-3), D.build_plant("kalman", 1e-1, c=0.5, eps=1e-2), D.build_plant("wagen", 1.0), D.build_plant("wagen", 1e-2), D.build_plant("vier_tank", 1e-2)]


# ------------------------------------------------------------------------------------------------------------------ 1. Batch

@pytest.mark.parametrize("pf_kind", ["null", "q", "gamma_i"])
def test_riccati_agrees_with_the_batch_solution_for_n_up_to_6(pf_kind):
    for plant in _plants():
        Pf = D.terminal_matrix(pf_kind, plant, 3.0)
        gains, _ = D.horizon_gains(plant, Pf, 6)
        for N in range(1, 7):
            Kb = O.batch_gain_float(plant.A, plant.B, plant.Q, plant.R, Pf, N)
            scale = max(1.0, float(np.max(np.abs(Kb))))
            assert np.max(np.abs(gains[N - 1] - Kb)) / scale < 1e-8, (plant.name, pf_kind, N)


def test_riccati_agrees_with_the_batch_solution_on_random_systems():
    rng = np.random.default_rng(3)
    for _ in range(40):
        n, m = int(rng.integers(2, 5)), int(rng.integers(1, 3))
        A, B = rng.normal(size=(n, n)), rng.normal(size=(n, m))
        L = rng.normal(size=(n, n))
        Q, Pf = L.T @ L + 0.1 * np.eye(n), 2.0 * np.eye(n)
        R = np.eye(m) * float(10 ** rng.uniform(-2, 1))
        plant = D.Plant("zufall", A, B, Q, R)
        gains, _ = D.horizon_gains(plant, Pf, 5)
        for N in (1, 3, 5):
            Kb = O.batch_gain_float(A, B, Q, R, Pf, N)
            assert np.max(np.abs(gains[N - 1] - Kb)) / max(1.0, float(np.max(np.abs(Kb)))) < 1e-7


def test_riccati_agrees_with_the_batch_solution_in_fractions_exactly():
    A, B = X.frac_matrix([[1, 1], [0, 1]]), X.frac_matrix([[0], [1]])
    Q, R = X.frac_matrix([[Fr(4, 5), Fr(2, 5)], [Fr(2, 5), Fr(6, 5)]]), X.frac_matrix([[1]])
    for Pf in (Q, X.frac_matrix([[2, 2], [2, 4]]), X.frac_matrix([[0, 0], [0, 0]])):
        g = X.horizon_gains_exact(A, B, Q, R, Pf, 6)
        for N in range(1, 7):
            assert g[N - 1][0] == O.batch_gain_fractions(A, B, Q, R, Pf, N), N
    Ak, Bk = X.frac_matrix([[Fr(4, 3), Fr(-2, 3)], [1, 0]]), X.frac_matrix([[1], [0]])
    Qk = X.frac_matrix([[Fr(4, 9) + Fr(1, 1000), Fr(-2, 3)], [Fr(-2, 3), 1 + Fr(1, 1000)]])
    Rk = X.frac_matrix([[Fr(1, 1000)]])
    gk = X.horizon_gains_exact(Ak, Bk, Qk, Rk, Qk, 4)
    for N in range(1, 5):
        assert gk[N - 1][0] == O.batch_gain_fractions(Ak, Bk, Qk, Rk, Qk, N), N


# ------------------------------------------------------------------------------------------------------------------ 2. Minimierung über Stellfolgen

def test_riccati_first_control_agrees_with_minimising_the_cost_over_all_control_sequences():
    for plant in (D.build_plant("kalman", 1e-3), D.build_plant("kalman", 1e-1), D.build_plant("wagen", 1.0)):
        x0 = np.array([1.0, -0.5])
        for N in range(1, 5):
            for Pf in (plant.Q, np.zeros((2, 2))):
                K = D.horizon_gains(plant, Pf, N)[0][N - 1]
                u0 = O.brute_force_u0(plant.A, plant.B, plant.Q, plant.R, Pf, x0, N)
                assert np.max(np.abs(u0 - (-K @ x0))) < 1e-6 * max(1.0, float(np.max(np.abs(K @ x0)))), (plant.name, N)


def test_the_plan_gradient_matches_a_finite_difference():
    plant = D.build_plant("wagen", 0.3)
    rng = np.random.default_rng(1)
    u = rng.normal(size=(4, 1))
    x0 = np.array([0.7, -0.2])
    g = O.plan_gradient(plant.A, plant.B, plant.Q, plant.R, plant.Q, x0, u)
    h = 1e-6
    for k in range(4):
        up, um = u.copy(), u.copy()
        up[k] += h
        um[k] -= h
        fd = (O.plan_cost(plant.A, plant.B, plant.Q, plant.R, plant.Q, x0, up) - O.plan_cost(plant.A, plant.B, plant.Q, plant.R, plant.Q, x0, um)) / (2 * h)
        assert g[k, 0] == pytest.approx(fd, rel=1e-6)


# ------------------------------------------------------------------------------------------------------------------ 3. ρ aus Spur und Determinante

def test_spectral_radius_agrees_with_the_trace_determinant_formula_and_jury():
    rng = np.random.default_rng(5)
    for _ in range(300):
        M = rng.normal(size=(2, 2))
        spur, det = float(np.trace(M)), float(np.linalg.det(M))
        rho = D.spectral_radius(M)
        assert rho == pytest.approx(O.jury_rho(spur, det), rel=1e-9, abs=1e-12)
        if abs(rho - 1.0) > 1e-6:
            assert O.jury_stable(spur, det) == (rho < 1.0)


def test_spectral_radius_of_the_horizon_laws_agrees_with_the_formula():
    for plant in _plants()[:4]:
        gains, _ = D.horizon_gains(plant, plant.Q, 12)
        for K in gains:
            Ac = plant.A - plant.B @ K
            assert D.spectral_radius(Ac) == pytest.approx(O.jury_rho(float(np.trace(Ac)), float(np.linalg.det(Ac))), rel=1e-9, abs=1e-12)


# ------------------------------------------------------------------------------------------------------------------ 4. DARE

def test_dare_agrees_with_the_fixpoint_of_the_riccati_recursion():
    for plant in (D.build_plant("kalman", 1e-3), D.build_plant("wagen", 1.0), D.build_plant("wagen", 1e-1), D.build_plant("kalman", 1e-1, c=2.0, eps=1e-1)):
        P, _ = D.dare(plant)
        Pfix = O.dare_fixpoint(plant.A, plant.B, plant.Q, plant.R)
        assert np.max(np.abs(P - Pfix)) < 1e-8 * max(1.0, float(np.max(np.abs(P)))), plant.name


def test_terminal_weight_equal_to_the_dare_solution_gives_k_inf_for_every_horizon():
    for plant in _plants():
        P, K = D.dare(plant)
        gains, costs = D.horizon_gains(plant, P, 40)
        scale = max(1.0, float(np.max(np.abs(K))))
        assert max(float(np.max(np.abs(Kn - K))) for Kn in gains) / scale < 1e-8, plant.name
        assert max(float(np.max(np.abs(Pn - P))) for Pn in costs) / max(1.0, float(np.max(np.abs(P)))) < 1e-8


# ------------------------------------------------------------------------------------------------------------------ 5. Kosten über die Lyapunov-Gleichung

def test_cost_of_an_infinite_future_by_lyapunov_agrees_with_a_long_simulation(kalman):
    gains, _ = D.horizon_gains(kalman, kalman.Q, 10)
    K = gains[7]                                                       # N = 8, ρ = 0,846: stabil
    S = D.lyapunov_cost_matrix(kalman, K)
    Ac = kalman.A - kalman.B @ K
    for x0 in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.6, -0.8])):
        x, total = x0.copy(), 0.0
        for _ in range(3000):
            u = -K @ x
            total += x @ kalman.Q @ x + float(u @ kalman.R @ u)
            x = Ac @ x
        assert total == pytest.approx(float(x0 @ S @ x0), rel=1e-9)


def test_cost_ratio_agrees_with_a_search_over_all_initial_directions(kalman):
    P, _ = D.dare(kalman)
    gains, _ = D.horizon_gains(kalman, kalman.Q, 12)
    th = np.linspace(0.0, np.pi, 20001)
    X0 = np.vstack([np.cos(th), np.sin(th)])
    for N in (7, 8, 10):
        K = gains[N - 1]
        S = D.lyapunov_cost_matrix(kalman, K)
        ratio_scan = float(np.max(np.einsum("ik,ij,jk->k", X0, S, X0) / np.einsum("ik,ij,jk->k", X0, P, X0)))
        assert D.cost_ratio(kalman, K, P) == pytest.approx(ratio_scan, rel=1e-6), N
    assert D.cost_ratio(kalman, gains[0], P) == float("inf")           # N = 1 instabil: Kosten nicht endlich
    assert D.cost_ratio(kalman, D.dare(kalman)[1], P) == pytest.approx(1.0, abs=1e-9)   # das optimale Gesetz hat das Verhältnis 1


# ------------------------------------------------------------------------------------------------------------------ 6. harte Bedingung

def test_hard_terminal_condition_agrees_with_the_limit_of_a_large_terminal_weight(kalman):
    for N in range(3, 9):
        K = D.hard_terminal_gain(kalman, N)
        Kbig = D.horizon_gains(kalman, 1e8 * np.eye(2), N)[0][N - 1]
        assert D.rho_of_gain(kalman, K) == pytest.approx(D.rho_of_gain(kalman, Kbig), abs=1e-4), N
        assert np.max(np.abs(K - Kbig)) < 1e-3 * max(1.0, float(np.max(np.abs(K)))), N


# ------------------------------------------------------------------------------------------------------------------ 7. Vier-Tank-Matrizen

def test_four_tank_open_loop_is_stable_and_eigenvalues_are_the_diagonal(vier_tank):
    ev = np.sort(np.linalg.eigvals(vier_tank.A).real)
    assert np.allclose(ev, [0.918, 0.921, 0.924, 0.937], atol=1e-12)
    assert np.allclose(vier_tank.A, np.triu(vier_tank.A))
    assert D.rho_open(vier_tank) == pytest.approx(0.937, abs=1e-12) and D.rho_open(vier_tank) < 1
    assert np.linalg.matrix_rank(np.hstack([vier_tank.B, vier_tank.A @ vier_tank.B, vier_tank.A @ vier_tank.A @ vier_tank.B, vier_tank.A @ vier_tank.A @ vier_tank.A @ vier_tank.B])) == 4
