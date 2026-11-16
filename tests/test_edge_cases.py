"""Randfälle, an denen die Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): N = 1, sehr kleines R, ρ genau 1 (nur als Band), Nullstelle auf dem Einheitskreis,
singuläres Q, nicht steuerbares System, N bis 400, nicht symmetrisches Terminalgewicht, Randwerte der Regler."""

import numpy as np
import pytest

import hfa_constants as C
import hfa_dynamics as D
import hfa_evaluation as ev


def test_horizon_one_is_defined_for_every_system(kalman, wagen, vier_tank):
    for plant in (kalman, wagen, vier_tank):
        gains, costs = D.horizon_gains(plant, plant.Q, 1)
        assert len(gains) == 1 and gains[0].shape == (plant.m, plant.n) and np.all(np.isfinite(gains[0]))
        assert np.all(np.isfinite(costs[0]))


def test_very_small_r_stays_finite_and_well_defined():
    for R in (1e-8, 1e-6):
        for kind in ("kalman", "wagen", "vier_tank"):
            plant = D.build_plant(kind, R)
            rho = D.rho_curve(plant, plant.Q, 60)
            assert np.all(np.isfinite(rho)), (kind, R)
            P, K = D.dare(plant)
            assert np.all(np.isfinite(K)) and D.rho_of_gain(plant, K) < 1.0


def test_rho_exactly_one_at_the_wagen_is_only_asserted_as_a_band(wagen):
    """Wagen N = 1: Eigenwerte 1 und 1/2. In Gleitkomma nie `== 1`: Band ρ ≥ 1 − 1e-9 und ρ ≤ 1 + 1e-9."""
    rho = D.rho_of_gain(wagen, D.horizon_gains(wagen, wagen.Q, 1)[0][0])
    assert 1.0 - 1e-9 <= rho <= 1.0 + 1e-9
    assert D.unstable_horizons(np.array([rho, 0.5])) == [1]            # grenzstabil zählt als nicht asymptotisch stabil
    kind, text = ev.verdict_horizon("Wagen", "P_f = Q", 1, rho, 0.42, 1.0, 2, 0.535, [1])
    assert kind == "warning" and "grenzstabil" in text


def test_zero_on_the_unit_circle_is_barely_stable_at_n1():
    """c = 1: Nullstelle bei 1. ρ(N = 1) = 0,9987 (Band), kein instabiles N bis 400."""
    cell = ev.trap_cell(1.0, 1e-3, 1e-3)
    assert cell["n_unstable"] == 0 and cell["first_stable"] == 1
    assert 0.995 < cell["rho"][0] < 1.0


def test_singular_state_weight_is_allowed(vier_tank):
    assert np.linalg.matrix_rank(vier_tank.Q) == 2                     # Q = 3·C′C hat Rang 2 bei n = 4
    gains, _ = D.horizon_gains(vier_tank, vier_tank.Q, 5)
    assert all(np.all(np.isfinite(K)) for K in gains)


def test_uncontrollable_system_is_rejected_with_a_message_and_gives_no_result():
    A = np.array([[0.5, 0.0], [0.0, 0.8]])
    B = np.array([[1.0], [0.0]])                                       # zweiter Zustand nicht erreichbar
    plant = D.Plant("nicht steuerbar", A, B, np.eye(2), np.array([[1.0]]))
    with pytest.raises(ValueError, match="nicht steuerbar"):
        D.dare(plant)
    with pytest.raises(ValueError, match="nicht steuerbar"):
        D.check_controllable(A, B)


def test_horizon_up_to_400_stays_finite_and_converges(kalman):
    rho = D.rho_curve(kalman, kalman.Q, 400)
    assert np.all(np.isfinite(rho)) and abs(rho[-1] - D.rho_of_gain(kalman, D.dare(kalman)[1])) < 1e-9


def test_asymmetric_terminal_weight_is_rejected_and_rounding_asymmetry_is_symmetrised(kalman):
    with pytest.raises(ValueError, match="symmetrisch"):
        D.horizon_gains(kalman, np.array([[1.0, 0.5], [0.0, 1.0]]), 3)
    P = np.array([[1.0, 0.5], [0.5 + 1e-14, 1.0]])
    gains, _ = D.horizon_gains(kalman, P, 3)
    assert np.all(np.isfinite(gains[0]))


def test_hard_terminal_condition_edge_cases(kalman, wagen):
    assert D.hard_terminal_gain(kalman, 1) is None and D.hard_terminal_gain(wagen, 1) is None   # zwei Zustände, ein Eingang: N = 1 nicht erfüllbar
    assert D.hard_terminal_gain(wagen, 2) is not None


def test_slider_extremes_of_the_analysis_are_finite():
    for system in C.SYSTEMS:
        for N in (C.N_MIN, C.N_MAX):
            for R in (min(C.R_OPTIONS), max(C.R_OPTIONS)):
                for pf in C.PF_KINDS:
                    a = ev.analyse(ev.Settings(system, 2 / 3, 1e-3, R, N, pf, 1000.0))
                    assert np.all(np.isfinite(a.rho)) and np.isfinite(a.rho_sel)
                    kind, text = ev.verdict_horizon_analysis(a)
                    assert kind in ("success", "info", "warning") and "nan" not in text


def test_every_c_of_the_family_builds_a_controllable_plant():
    for c in C.C_OPTIONS:
        plant = D.build_plant("kalman", 1e-3, c=c)
        D.check_controllable(plant.A, plant.B)
        assert np.allclose(plant.Q, plant.Q.T)
