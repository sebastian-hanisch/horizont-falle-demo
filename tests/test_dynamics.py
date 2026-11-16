"""Kern gegen die Handbeispiele (Mini-Instanzen von Hand gerechnet, DEMO-PLAYBOOK §1b): Skalar, Wagen, Wagen mit rationaler Riccati-Lösung, Kalmans Gegenbeispiel,
harte Terminalbedingung. Was „exakt“ heißt, ist in Brüchen (`fractions`) gerechnet und gilt auf jeder Plattform; Eigenwerte und SciPy-Lösungen nur als Band.

Konvention: u = −K x, Kosten Σ (x′Qx + u′Ru) + x_N′P_f x_N, K_N gehört zum Horizont N."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import hfa_dynamics as D
import hfa_exact as X


# ------------------------------------------------------------------------------------------------------------------ Handbeispiel 1: Skalar x⁺ = 3x + u

def _scalar_poles(pf, n_max, q=Fr(9, 10), a=Fr(3), r=Fr(1)):
    A, B, Q, R = X.frac_matrix([[a]]), X.frac_matrix([[1]]), X.frac_matrix([[q]]), X.frac_matrix([[r]])
    gains = X.horizon_gains_exact(A, B, Q, R, X.frac_matrix([[pf]]), n_max)
    return [a - K[0][0] for K, _ in gains], [P[0][0] for _, P in gains]


def test_scalar_poles_by_hand_with_terminal_weight_q():
    """Von Hand: Pol des Horizonts N = a·r/(r + p_(N−1)) mit p₀ = P_f = 9/10: N = 1 gibt 3/(1 + 9/10) = 30/19; N = 2 gibt 570/1171."""
    poles, costs = _scalar_poles(Fr(9, 10), 5)
    assert poles[0] == Fr(30, 19) and poles[0] > 1                   # instabil
    assert costs[0] == Fr(9, 10) + Fr(81, 19)                         # p₁ = q + a²·p₀·r/(r + p₀) = 9/10 + 81/19
    assert poles[1] == Fr(570, 1171) and poles[1] < 1                 # N = 2 schon stabil
    assert [round(float(p), 3) for p in poles[2:]] == [0.318, 0.302, 0.3]


def test_scalar_riccati_solution_gives_pole_three_tenths_for_every_horizon():
    """Von Hand: p = 9 löst p = q + a²p/(1 + p) (9/10 + 81/10 = 9), K = 3·9/10 = 27/10, Pol 3/10; mit P_f = 9 für N = 1 bis 7 exakt."""
    assert Fr(9, 10) + 9 * Fr(9) / (1 + Fr(9)) == Fr(9)
    poles, costs = _scalar_poles(Fr(9), 7)
    assert poles == [Fr(3, 10)] * 7 and costs == [Fr(9)] * 7


# ------------------------------------------------------------------------------------------------------------------ Handbeispiel 2: Wagen, Q = I, R = 1, P_f = Q

A_W, B_W = X.frac_matrix([[1, 1], [0, 1]]), X.frac_matrix([[0], [1]])
I2 = X.frac_matrix([[1, 0], [0, 1]])
R1 = X.frac_matrix([[1]])


def test_wagen_n1_gives_gain_half_and_eigenvalues_one_and_half():
    g = X.horizon_gains_exact(A_W, B_W, I2, R1, I2, 4)
    K1, P1 = g[0]
    assert K1 == [[0, Fr(1, 2)]]                                      # B′QB = 1, R + 1 = 2, B′QA = (0, 1)
    Ac = X.closed_loop_exact(A_W, B_W, K1)
    assert Ac == [[1, 1], [0, Fr(1, 2)]]
    assert X.trace_det(Ac) == (Fr(3, 2), Fr(1, 2))                    # Eigenwerte 1 und 1/2
    assert X.mat_mul([[1, 2]], Ac) == [[1, 2]]                        # (1, 2) ist Linkseigenvektor zum Eigenwert 1: Endlage x₁(0) + 2·x₂(0)
    assert P1 == [[2, 1], [1, Fr(5, 2)]]                              # Π(1) = I + A′A − A′B·(1/2)·B′A


def test_wagen_n2_and_n3_by_hand():
    g = X.horizon_gains_exact(A_W, B_W, I2, R1, I2, 4)
    K2, K3 = g[1][0], g[2][0]
    assert K2 == [[Fr(2, 7), 1]]                                      # R + B′ΠB = 7/2, B′ΠA = (1, 7/2)
    assert X.trace_det(X.closed_loop_exact(A_W, B_W, K2)) == (1, Fr(2, 7))
    assert X.rho_from_trace_det(Fr(1), Fr(2, 7)) == pytest.approx(np.sqrt(2 / 7), abs=1e-15)          # komplexes Paar: ρ = √(2/7) = 0,535
    assert K3 == [[Fr(2, 5), Fr(6, 5)]]
    assert X.trace_det(X.closed_loop_exact(A_W, B_W, K3))[1] == Fr(1, 5)


def test_wagen_terminal_zero_shifts_the_horizon_by_one():
    """P_f = 0: K_1 = 0 (nichts tun), und K_N(P_f = 0) = K_(N−1)(P_f = Q) für N = 2 bis 4, exakt."""
    zero = X.frac_matrix([[0, 0], [0, 0]])
    g0 = X.horizon_gains_exact(A_W, B_W, I2, R1, zero, 4)
    gq = X.horizon_gains_exact(A_W, B_W, I2, R1, I2, 4)
    assert g0[0][0] == [[0, 0]]
    assert all(g0[j][0] == gq[j - 1][0] for j in range(1, 4))


def test_wagen_one_over_r_plus_one_in_exact_fractions():
    """N = 1: K_1 = (0, 1/(R+1)), der zweite Eigenwert ist R/(R+1) (R = 1, 1/10, 1/100); ρ = 1 für jedes R."""
    for R in (Fr(1), Fr(1, 10), Fr(1, 100)):
        K1 = X.horizon_gains_exact(A_W, B_W, I2, [[R]], I2, 1)[0][0]
        assert K1 == [[0, 1 / (R + 1)]]
        spur, det = X.trace_det(X.closed_loop_exact(A_W, B_W, K1))
        assert det == R / (R + 1) and spur == 1 + R / (R + 1)


# ------------------------------------------------------------------------------------------------------------------ Handbeispiel 3: Wagen mit rationaler Riccati-Lösung

P_RAT = X.frac_matrix([[2, 2], [2, 4]])


def _q_for_p():
    S = X.mat_add(R1, X.mat_mul(X.transpose(B_W), X.mat_mul(P_RAT, B_W)))
    K = X.mat_mul(X.mat_inv(S), X.mat_mul(X.transpose(B_W), X.mat_mul(P_RAT, A_W)))
    Q = X.mat_add(P_RAT, X.mat_add(X.mat_mul(X.transpose(A_W), X.mat_mul(P_RAT, A_W)), X.mat_mul(X.transpose(A_W), X.mat_mul(P_RAT, X.mat_mul(B_W, K))), -1), -1)
    return K, Q


def test_rational_riccati_solution_k_and_q_by_hand():
    """Von Hand: B′PB = 4, R + 4 = 5, B′PA = (2, 6), also K = (2/5, 6/5); Q = P − A′PA + A′PBK = [[4/5, 2/5], [2/5, 6/5]]; A − BK hat Spur 4/5 und Determinante 1/5."""
    K, Q = _q_for_p()
    assert K == [[Fr(2, 5), Fr(6, 5)]]
    assert Q == [[Fr(4, 5), Fr(2, 5)], [Fr(2, 5), Fr(6, 5)]]
    assert Q[0][0] > 0 and Q[0][0] * Q[1][1] - Q[0][1] ** 2 > 0       # positiv definit
    assert X.trace_det(X.closed_loop_exact(A_W, B_W, K)) == (Fr(4, 5), Fr(1, 5))


def test_rational_riccati_solution_is_a_fixpoint_for_30_horizons_but_p_f_equal_q_is_not():
    K, Q = _q_for_p()
    gp = X.horizon_gains_exact(A_W, B_W, Q, R1, P_RAT, 30)
    assert all(Kn == K and Pn == P_RAT for Kn, Pn in gp)              # K_N = K und Π_N = P für N = 1 bis 30, exakt
    gq = X.horizon_gains_exact(A_W, B_W, Q, R1, Q, 6)
    assert sum(Kn != K for Kn, _ in gq) == 6                          # mit P_f = Q statt P: K_N ≠ K in 6 von 6 Horizonten


def test_scipy_dare_agrees_with_the_rational_solution():
    _, Q = _q_for_p()
    plant = D.Plant("rational", np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([[0.0], [1.0]]), np.array([[float(x) for x in r] for r in Q]), np.array([[1.0]]))
    P, K = D.dare(plant)
    assert np.max(np.abs(P - np.array([[2.0, 2.0], [2.0, 4.0]]))) < 1e-12
    assert np.max(np.abs(K - np.array([[0.4, 1.2]]))) < 1e-12


# ------------------------------------------------------------------------------------------------------------------ Handbeispiel 4: Kalmans Gegenbeispiel

EPS = Fr(1, 1000)
A_K = X.frac_matrix([[Fr(4, 3), Fr(-2, 3)], [1, 0]])
B_K = X.frac_matrix([[1], [0]])
C_K = X.frac_matrix([[Fr(-2, 3), 1]])
Q_K = X.mat_add(X.mat_mul(X.transpose(C_K), C_K), X.frac_matrix([[EPS, 0], [0, EPS]]))
R_K = X.frac_matrix([[EPS]])


def test_kalman_n1_by_hand():
    """Von Hand: Q₁₁ = 4/9 + 1/1000, R + Q₁₁ = 4/9 + 2/1000; K_1 = (−982/6027, −4009/6027); Spur 3006/2009, Determinante 3/2009: 2009 z² − 3006 z + 3 = 0, Eigenwerte 1,4953 und 0,000999."""
    K1 = X.horizon_gains_exact(A_K, B_K, Q_K, R_K, Q_K, 1)[0][0]
    assert K1 == [[Fr(-982, 6027), Fr(-4009, 6027)]]
    spur, det = X.trace_det(X.closed_loop_exact(A_K, B_K, K1))
    assert (spur, det) == (Fr(3006, 2009), Fr(3, 2009))
    rho = X.rho_from_trace_det(spur, det)
    assert rho == pytest.approx(1.49527, abs=1e-5)
    assert float(det) / rho == pytest.approx(0.001, abs=2e-6)


def test_kalman_horizons_one_to_twelve_in_fractions_give_the_source_values_and_n7_as_smallest_stable():
    g = X.horizon_gains_exact(A_K, B_K, Q_K, R_K, Q_K, 12)
    rho = [X.rho_from_trace_det(*X.trace_det(X.closed_loop_exact(A_K, B_K, K))) for K, _ in g]
    assert [round(r, 3) for r in rho] == [1.495, 1.484, 1.458, 1.405, 1.307, 1.158, 0.989, 0.846, 0.756, 0.707, 0.684, 0.673]
    assert next(i + 1 for i, r in enumerate(rho) if r < 1) == 7
    assert rho[4] > 1 > rho[6]                                        # Quelle: N = 5 instabil (1,307), N = 7 stabil (0,989)


def test_kalman_floating_point_route_agrees_with_the_fractions(kalman):
    rho_f = D.rho_curve(kalman, kalman.Q, 12)
    g = X.horizon_gains_exact(A_K, B_K, Q_K, R_K, Q_K, 12)
    rho_x = [X.rho_from_trace_det(*X.trace_det(X.closed_loop_exact(A_K, B_K, K))) for K, _ in g]
    assert np.max(np.abs(rho_f - np.array(rho_x))) < 1e-9
    assert D.first_stable(rho_f) == 7 and D.unstable_horizons(rho_f) == [1, 2, 3, 4, 5, 6]


def test_kalman_infinite_horizon_by_dare(kalman):
    P, K = D.dare(kalman)
    assert D.rho_of_gain(kalman, K) == pytest.approx(0.66404, abs=1e-5)
    assert D.rho_open(kalman) == pytest.approx(np.sqrt(2 / 3), abs=1e-12)      # offene Strecke: Eigenwerte vom Betrag √(2/3) = 0,8165


# ------------------------------------------------------------------------------------------------------------------ Handbeispiel 5: harte Terminalbedingung N = 2

def test_hard_terminal_condition_n2_is_deadbeat(kalman):
    """Von Hand: die zweite Komponente von x(2) ist x₁(1); x(2) = 0 verlangt also x₁(1) = 0, das heißt u₀ = −(A x₀)₁ und K = (4/3, −2/3) = erste Zeile von A; A − BK = [[0, 0], [1, 0]]."""
    K = D.hard_terminal_gain(kalman, 2)
    assert np.max(np.abs(K - np.array([[4 / 3, -2 / 3]]))) < 1e-9
    Ac = kalman.A - kalman.B @ K
    assert np.max(np.abs(Ac - np.array([[0.0, 0.0], [1.0, 0.0]]))) < 1e-9
    exact = X.closed_loop_exact(A_K, B_K, [[Fr(4, 3), Fr(-2, 3)]])
    assert exact == [[0, 0], [1, 0]] and X.trace_det(exact) == (0, 0)


def test_hard_terminal_condition_is_not_solvable_in_one_step_for_two_states(kalman):
    assert D.hard_terminal_gain(kalman, 1) is None


# ------------------------------------------------------------------------------------------------------------------ Hilfsfunktionen

def test_first_stable_and_unstable_horizons_on_a_given_curve():
    rho = np.array([1.2, 1.0, 0.9, 0.95, 1.1, 0.5])
    assert D.unstable_horizons(rho) == [1, 2, 5]
    assert D.first_stable(rho) == 3
    assert D.later_unstable(rho) == 1                                  # nach dem kleinsten stabilen N (3) ist N = 5 wieder instabil
    assert D.first_stable(np.array([1.5, 1.2])) is None and D.later_unstable(np.array([1.5, 1.2])) == 0


def test_terminal_matrix_kinds(kalman):
    P, _ = D.dare(kalman)
    assert np.all(D.terminal_matrix("null", kalman) == 0)
    assert np.allclose(D.terminal_matrix("q", kalman), kalman.Q)
    assert np.allclose(D.terminal_matrix("gamma_i", kalman, 3.0), 3.0 * np.eye(2))
    assert np.allclose(D.terminal_matrix("gamma_pinf", kalman, 2.0, P), 2.0 * P)
    with pytest.raises(ValueError):
        D.terminal_matrix("unbekannt", kalman)
