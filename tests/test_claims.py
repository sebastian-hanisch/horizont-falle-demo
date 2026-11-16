"""Claims-Tests: jede Zahl der Befunde B1 bis B10 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a), dazu je Preset die erwartete Verdikt-Zeile und die Zahlen der Hilfetexte.

Voraussetzung überall: lineares Modell, ohne Beschränkungen, ohne Störung, Gesetz u = −K_N x wiederholt. Plattformabhängiges (Eigenwerte, DARE, Lyapunov, Schwellen nahe 1) steht als Band; Zählungen, die an
keiner Schwelle knapp liegen, exakt. Die vorgerechneten Zufallssysteme (`data/hfa_frozen.json`) werden hier auf Konsistenz mit der README geprüft; ihre Neu-Rechnung (`tools/make_frozen.py`, rund 3 Minuten)
läuft nur für die ersten 500 Ziehungen von Zufalls-Seed 1 (Band ± 2) und für die 200 Zufallswerte c (exakte Zählung, die einzige Ausnahme liegt 1,2·10⁻⁴ unter der Schwelle)."""

import re
from pathlib import Path

import numpy as np
import pytest

import hfa_constants as C
import hfa_dynamics as D
import hfa_evaluation as ev
import hfa_random as Rz
from hfa_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
NB = " "


def _readme_has(*texts):
    flat = README.replace(NB, " ")
    for t in texts:
        assert t in flat, t


def _S(**kw):
    return ev.Settings(**kw)


KALMAN = _S()
P_INF_COST = None


# ------------------------------------------------------------------------------------------------------------------ B1

def test_b1_kalman_rho_for_n_1_to_12_and_the_threshold():
    a = ev.analyse(KALMAN)
    expected = [1.495, 1.484, 1.458, 1.405, 1.307, 1.158, 0.989, 0.846, 0.756, 0.707, 0.684, 0.673]
    assert np.max(np.abs(a.rho[:12] - np.array(expected))) < 5e-4
    assert a.rho_inf == pytest.approx(0.664, abs=5e-4) and a.rho_open == pytest.approx(0.816, abs=5e-4)
    assert a.unstable == [1, 2, 3, 4, 5, 6] and a.first_stable == 7 and a.later_unstable == 0
    assert 0.98 < a.rho[6] < 0.995                                      # Band um die Schwelle (1,1 % unter 1), nicht bitgleich
    assert int(np.sum(a.rho[12:40] >= 1.0 - D.TOL)) == 0 and len(a.rho[12:40]) == 28    # 0 von 28
    _readme_has("1,495; 1,484; 1,458; 1,405; 1,307; 1,158; 0,989; 0,846; 0,756; 0,707; 0,684; 0,673", "(0 von 28)", "kleinstes stabiles N = 7")


def test_b1_distance_to_the_limit_shrinks_geometrically():
    a = ev.analyse(KALMAN)
    for N, expected in ((20, 1.3e-5), (30, 3.5e-9), (40, 9.7e-13)):
        d = abs(a.rho[N - 1] - a.rho_inf)
        assert expected / 1.05 < d < expected * 1.05, (N, d)
    _readme_has("1,3·10⁻⁵ (N = 20), 3,5·10⁻⁹ (N = 30), 9,7·10⁻¹³ (N = 40)")


# ------------------------------------------------------------------------------------------------------------------ B2

def test_b2_terminal_weight_p_inf_gives_k_inf_for_every_horizon():
    plant = ev.plant_of(KALMAN)
    P, K = D.dare(plant)
    gains, costs = D.horizon_gains(plant, P, 40)
    assert max(float(np.max(np.abs(Kn - K))) for Kn in gains) < 1e-12                  # README: 3,3·10⁻¹⁶ (Schranke, weil plattformabhängig im letzten Bit)
    assert float(np.max(np.abs(costs[-1] - P))) < 1e-12                                 # README: 4,4·10⁻¹⁶
    for N in (1, 2, 5, 12, 40):
        assert D.rho_of_gain(plant, gains[N - 1]) == pytest.approx(0.66404, abs=5e-6)
    _readme_has("3,3·10⁻¹⁶ für N = 1 bis 40", "4,4·10⁻¹⁶", "ρ = 0,66404")


def test_b2_exact_wagen_statements_are_in_the_readme_and_tested_in_fractions():
    _readme_has("K_N = (2/5, 6/5) und Π_N = P für N = 1 bis 30", "dasselbe K_N für N = 1 bis 6", "K_N ≠ K in 6 von 6 Horizonten")


# ------------------------------------------------------------------------------------------------------------------ B3

def test_b3_frozen_random_counts_match_the_readme():
    rows = ev.load_frozen()["random"]
    assert [r["seed"] for r in rows] == list(C.SEEDS) and all(r["drawn"] == C.N_RANDOM_SYSTEMS for r in rows)
    valid, open_stable = sum(r["valid"] for r in rows), sum(r["open_stable"] for r in rows)
    n1, any_n = sum(r["n1_unstable"] for r in rows), sum(r["any_unstable"] for r in rows)
    assert (valid, open_stable, n1, any_n) == (119624, 40727, 1238, 1247)
    assert any_n - n1 == 9 and sum(1 for r in rows if r["any_unstable"] > r["n1_unstable"]) == 8
    shares = [r["n1_unstable"] / r["open_stable"] for r in rows]
    assert round(100 * min(shares), 1) == 2.5 and round(100 * max(shares), 1) == 3.5 and round(100 * n1 / open_stable, 1) == 3.0
    r12 = ev.random_row(12)
    assert (r12["n1_unstable"], r12["open_stable"]) == (67, 2047) and round(100 * 67 / 2047, 1) == 3.3
    assert max(r["dev_max"] for r in rows) < 3e-6 and max(r["dev_max"] for r in rows) > 2.5e-6
    assert sorted(r["dev_median"] for r in rows)[10] < 1e-15
    _readme_has("119 624 gültige Systeme", "40 727", "1 238 (3,0 %; je Zufalls-Seed 2,5 % bis 3,5 %)", "67 von 2 047 (3,3 %)", "1 247 Strecken, also in 9 mehr", "(in 8 der 20 Zufalls-Seeds)", "2,9·10⁻⁶", "8,2·10⁻¹⁶")


def test_b3_recomputing_the_first_500_draws_of_seed_1_reproduces_the_frozen_prefix_within_the_band():
    frozen = ev.random_row(1)["prefix500"]
    live = Rz.count_systems(1, 500)
    for key in ("valid", "open_stable", "n1_unstable", "any_unstable"):
        assert abs(live[key] - frozen[key]) <= 2, key
    assert live["dev_max"] < 1e-6


def test_b3_the_frozen_file_was_produced_with_the_documented_settings():
    rows = ev.load_frozen()["random"]
    assert all(r["prefix500"]["drawn"] == 500 for r in rows) and Rz.N_RANDOM_HORIZON == 10 and Rz.CTRL_MIN == 1e-3


# ------------------------------------------------------------------------------------------------------------------ B4

def test_b4_unstable_horizons_for_the_other_terminal_weights():
    table = ev.terminal_table(KALMAN)
    by = {r["label"]: r for r in table["rows"] if not r["chosen"]}
    assert by["P_f = 0"]["unstable"] == [2, 3, 4, 5, 6, 7] and by["P_f = Q"]["unstable"] == [1, 2, 3, 4, 5, 6]
    assert by["P_f = 10·Q"]["unstable"] == [1, 2, 3, 4] and by["P_f = 1 000·Q"]["unstable"] == [1]
    for lab in ("P_f = I", "P_f = 100·I", "P_f = 0,5·P_∞", "P_f = 0,9·P_∞", "P_f = 1,1·P_∞", "P_f = 2·P_∞", "P_f = 1 000·P_∞", "P_f = P_∞"):
        assert by[lab]["unstable"] == [], lab
    assert by["P_f = 0,5·P_∞"]["rho12"][0] == pytest.approx(0.664, abs=5e-4) and by["P_f = 0,5·P_∞"]["rho12"][1] == pytest.approx(0.920, abs=5e-4)
    assert by["P_f = P_∞"]["rho_all_equal"] and table["dev_pinf"] < 1e-12
    _readme_has("P_f = 0: N = 2 bis 7", "P_f = Q: N = 1 bis 6", "10·Q: N = 1 bis 4", "1 000·Q: nur N = 1", "ρ(1) = 0,664, aber ρ(2) = 0,920")


def test_b4_the_unstable_lists_up_to_n_60_equal_those_up_to_n_150():
    plant = ev.plant_of(KALMAN)
    for Pf in (np.zeros((2, 2)), plant.Q, 10 * plant.Q, 1000 * plant.Q):
        rho = D.rho_curve(plant, Pf, 150)
        assert D.unstable_horizons(rho[:60]) == D.unstable_horizons(rho)


# ------------------------------------------------------------------------------------------------------------------ B5

def test_b5_hard_terminal_condition():
    table = ev.terminal_table(KALMAN)
    hard = table["hard"]                                                # N = 2 bis 12
    assert hard[0] < 1e-6                                               # Deadbeat: Jordan-Block, ρ ~ 1e-8 (Wurzel der Rundung)
    assert np.max(np.abs(np.array(hard[1:]) - np.array([0.460, 0.585, 0.631, 0.650, 0.658, 0.661, 0.663, 0.664, 0.664, 0.664]))) < 5e-4
    plant = ev.plant_of(KALMAN)
    big = D.rho_curve(plant, 1e6 * np.eye(2), 12)
    assert np.max(np.abs(np.array(hard[1:]) - big[2:12])) < 5e-5        # fünf Stellen für N = 3 bis 12
    _readme_has("0,460; 0,585; 0,631; 0,650; 0,658; 0,661; 0,663; 0,664; 0,664; 0,664 (N = 3 bis 12)", "auf fünf Stellen")


# ------------------------------------------------------------------------------------------------------------------ B6

def test_b6_price_of_a_short_stable_horizon():
    r = ev.price_curve(KALMAN)
    assert np.all(np.isinf(r[:6]))
    for N, expected in ((7, 9.31), (8, 1.210), (9, 1.035), (10, 1.0066), (11, 1.0013), (12, 1.00025)):
        assert r[N - 1] == pytest.approx(expected, rel=1e-3), N
    ri = ev.price_curve(_S(pf="gamma_i", gamma=1.0))
    for N, expected in ((1, 1.791), (2, 1.094), (3, 1.017)):
        assert ri[N - 1] == pytest.approx(expected, rel=1e-3), N
    assert np.all(np.isfinite(ri))                                      # P_f = I: stabil für jedes N bis 40
    rw = ev.price_curve(_S(system="wagen", R=1.0))
    assert np.isinf(rw[0]) and rw[1] == pytest.approx(1.102, rel=1e-3) and rw[2] == pytest.approx(1.0026, rel=1e-4) and rw[3] == pytest.approx(1.00004, abs=2e-5)
    _readme_has("N = 7: 9,31 (stabil, ρ = 0,989)", "1,210; 9: 1,035; 10: 1,0066; 11: 1,0013; 12: 1,00025", "N = 1: 1,791; 2: 1,094; 3: 1,017", "N = 2: 1,102; 3: 1,0026; 4: 1,00004")


# ------------------------------------------------------------------------------------------------------------------ B7

def test_b7_time_varying_schedule():
    rows = ev.timevarying_rows(KALMAN, 40)
    geo = np.array([r["geo"] for r in rows])
    assert np.all(geo[:12] >= 1.0) and np.all(geo[12:] < 1.0)           # N = 1 bis 12: mindestens 1; N = 13 bis 40: darunter
    for N, expected in ((1, 1.495), (6, 1.379), (11, 1.070), (12, 1.029)):
        assert geo[N - 1] == pytest.approx(expected, abs=1e-3), N
    sigma = np.array([r["sigma"] for r in rows])
    assert int(np.argmax(sigma)) + 1 == 6 and sigma[5] == pytest.approx(8.28, abs=5e-3)
    _readme_has("für N = 1 bis 12 bei mindestens 1", "von N = 13 bis 40 darunter", "bis 8,28 (N = 6)")


# ------------------------------------------------------------------------------------------------------------------ B8

def test_b8_wagen():
    for R, rho2, rho_inf in ((1.0, 0.535, 0.422), (0.1, 0.497, 0.362), (0.01, 0.500, 0.380), (0.001, 0.500, 0.382)):
        plant = D.build_plant("wagen", R)
        rho = D.rho_curve(plant, plant.Q, 12)
        assert 1.0 - 1e-9 <= rho[0] <= 1.0 + 1e-9                       # Band, nie ==
        assert rho[1] == pytest.approx(rho2, abs=5e-4) and D.rho_of_gain(plant, D.dare(plant)[1]) == pytest.approx(rho_inf, abs=5e-4)
        assert D.unstable_horizons(rho) == [1]
        rho0 = D.rho_curve(plant, np.zeros((2, 2)), 12)
        assert D.unstable_horizons(rho0) == [1, 2]                      # P_f = 0: N = 1 und N = 2
    _readme_has("ρ = 0,535; 0,497; 0,500; 0,500 gegen ρ_∞ = 0,422; 0,362; 0,380; 0,382", "Endlage x₁(0) + (R + 1)·x₂(0)", "(4 von 4)")


def test_b8_wagen_endpoint_by_simulation():
    plant = D.build_plant("wagen", 1.0)
    K1 = D.horizon_gains(plant, plant.Q, 1)[0][0]
    x = np.array([2.0, 1.0])
    for _ in range(200):
        x = (plant.A - plant.B @ K1) @ x
    assert x[0] == pytest.approx(4.0, abs=1e-9) and abs(x[1]) < 1e-9    # x₁(0) + 2·x₂(0) = 4


# ------------------------------------------------------------------------------------------------------------------ B9

def test_b9_zero_class_counts_and_first_stable_n():
    rows = ev.nullstelle_rows()
    counts = ev.count_nullstelle(rows)
    assert counts == {"outside": 8, "outside_trap": 8, "rest": 5, "rest_trap": 0, "c_min": 0.1, "c_max": 0.9}
    out = [r for r in rows if r["class"] == "außerhalb"]
    assert [r["first_stable"] for r in sorted(out, key=lambda r: -r["c"])] == [14, 10, 7, 6, 5, 4, 4, 2]
    assert all(r["later_unstable"] == 0 for r in rows)
    by_c = {round(r["c"], 4): r for r in rows}
    for c, zero, rho1 in ((0.9, 1.111, 1.109), (0.8, 1.250, 1.247), (0.6667, 1.500, 1.495), (0.5, 2.000, 1.988)):
        assert by_c[c]["zero"] == pytest.approx(zero, abs=1e-3) and by_c[c]["rho1"] == pytest.approx(rho1, abs=1e-3)
    assert 0.995 < by_c[1.0]["rho1"] < 1.0                              # Nullstelle auf dem Kreis: knapp stabil (Band)
    _readme_has("(8 von 8)", "(5 von 5)", "kleinstes stabiles N = 14, 10, 7, 6, 5, 4, 4, 2", "1,109 bei 1,111; 1,247 bei 1,250; 1,495 bei 1,500; 1,988 bei 2,000")


def test_b9_maps_over_r_eps_and_c():
    m = ev.map_R_eps()
    assert m["first"] == [[8, 7, 7, 4, 1, 1, 1], [5, 5, 4, 4, 1, 1, 1], [2, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]] and m["later"] == 0
    mc = ev.map_c_R()
    assert mc["later"] == 0
    row = {round(c, 4): r for c, r in zip(mc["c"], mc["first"])}
    assert row[0.6667] == [8, 7, 7, 4, 1, 1, 1] and row[0.9] == [14, 14, 12, 5, 1, 1, 1] and row[0.1] == [2, 2, 2, 2, 1, 1, 1] and row[2.0] == [1] * 7
    _readme_has("8; 7; 7; 4; 1; 1; 1", "2; 2; 2; 2; 1; 1; 1", "bei ε = 1 überall 1", "in keiner der 28 Zellen")


def test_b9_cross_check_with_200_random_c_recomputed_and_frozen():
    frozen = ev.load_frozen()["random_c"]
    rng = SplitMix64(frozen["seed"])
    cs = [rng.uniform(0.05, 4.0) for _ in range(frozen["n"])]
    outside = trap_out = trap_rest = 0
    exceptions = []
    for c in cs:
        cell = ev.trap_cell(c, ev.NULL_PLANT_EPS, ev.NULL_PLANT_R)
        is_out = 0.0 < c < 1.0
        trap = cell["n_unstable"] > 0
        outside += is_out
        trap_out += is_out and trap
        trap_rest += (not is_out) and trap
        if is_out != trap:
            exceptions.append((c, cell["rho"][0]))
    assert (outside, trap_out, trap_rest) == (43, 42, 0) and len(exceptions) == 1
    assert exceptions[0][0] == pytest.approx(0.9988, abs=1e-4) and exceptions[0][1] == pytest.approx(0.99988, abs=2e-5) and 1.0 / exceptions[0][0] == pytest.approx(1.0012, abs=1e-4)
    assert (frozen["outside"], frozen["outside_trap"], frozen["rest"], frozen["rest_trap"], len(frozen["exceptions"])) == (43, 42, 157, 0, 1)
    _readme_has("bei Nullstelle außerhalb Falle in 42 von 43, sonst in 0 von 157", "c = 0,9988 (Nullstelle 1,0012)", "ρ(1) = 0,99988", "in 199 von 200 Fällen")


def test_b9_zero_just_outside_the_circle_c_099():
    cell = ev.trap_cell(0.99, 1e-3, 1e-3)
    assert cell["n_unstable"] == 5 and cell["rho"][0] == pytest.approx(1.009, abs=5e-4)
    _readme_has("c = 0,99, Nullstelle 1,0101: 5 instabile N, ρ(1) = 1,009")


# ------------------------------------------------------------------------------------------------------------------ B10

def test_b10_four_tank_thresholds_as_bands():
    expected = {1e-4: ((54, 55)), 1e-3: ((35, 36)), 1e-2: ((19, 20))}
    for R, (last_q, last_0) in expected.items():
        cq, c0 = ev.vier_tank_curve(R, "q"), ev.vier_tank_curve(R, "null")
        assert cq["unstable"][0] == 1 and abs(cq["unstable"][-1] - last_q) <= 1 and cq["unstable"] == list(range(1, cq["unstable"][-1] + 1)), R
        assert c0["unstable"][0] == 2 and abs(c0["unstable"][-1] - last_0) <= 1, R
    for R in (0.1, 1.0, 10.0, 100.0):
        assert ev.vier_tank_curve(R, "q")["unstable"] == [] and ev.vier_tank_curve(R, "null")["unstable"] == []
    cur = ev.vier_tank_curve(1e-4, "q")
    assert cur["rho"][53] == pytest.approx(1.0008, abs=1e-3) and cur["rho"][54] == pytest.approx(0.9973, abs=1e-3)
    assert cur["rho"][53] > 1.0 - 1e-4 and cur["rho"][54] < 1.0 + 1e-4  # Schwelle zwischen N = 54 und 55, Band ± 1
    assert cur["rho_inf"] == pytest.approx(0.943, abs=5e-4) and ev.vier_tank_curve(1e-3, "q")["rho_inf"] == pytest.approx(0.943, abs=5e-4)
    assert ev.vier_tank_curve(1e-2, "q")["rho_inf"] == pytest.approx(0.9426, abs=5e-5)
    assert D.rho_open(D.build_plant("vier_tank", 1e-2)) == pytest.approx(0.937, abs=1e-9)
    _readme_has("N = 1 bis 54 (P_f = Q) bzw. N = 2 bis 55 (P_f = 0)", "1 bis 35 bzw. 2 bis 36", "1 bis 19 bzw. 2 bis 20", "ρ(54) = 1,0008 und ρ(55) = 0,9973", "ρ_∞ = 0,943 bei R = 10⁻⁴ und 10⁻³ (0,9426 bei 10⁻²)")


def test_b10_terminal_weight_dare_gives_k_inf_for_every_r():
    for R in C.R_DECADES:
        assert ev.vier_tank_curve(R, "gamma_pinf")["dev20"] < 1e-8, R


# ------------------------------------------------------------------------------------------------------------------ Presets

def _verdict(name):
    p = C.PRESETS[name]
    return ev.verdict_horizon_analysis(ev.analyse(_S(system=p["system"], c=p["c"], eps=p["eps"], R=p["R"], N=p["N"], pf=p["pf"], gamma=p["gamma"])))


def test_preset_verdicts_have_the_expected_line_and_no_effect_preset_reports_no_effect():
    kind, text = _verdict("Von Hand (Kalman, N = 1)")
    assert kind == "warning" and "N = 1 gibt ρ = 1,495" in text and "um 50 % je Schritt" in text and "erst N = 7 ist stabil (ρ = 0,989)" in text and "N → ∞: ρ = 0,664" in text
    kind, text = _verdict("Kleinstes stabiles N (N = 7)")
    assert kind == "info" and "N = 7 ist stabil (ρ = 0,989)" in text and "instabil sind N = 1 bis 6" in text
    kind, text = _verdict("Terminalkosten = Riccati-Lösung")
    assert kind == "success" and "P_f = P_∞" in text and "N = 1 ist stabil (ρ = 0,664)" in text and "kein Horizont von N = 1 bis 150 ist instabil" in text
    kind, text = _verdict("Wagen (N = 1 grenzstabil)")
    assert kind == "warning" and "ρ = 1 (Band)" in text and "grenzstabil" in text and "erst N = 2 ist stabil (ρ = 0,535)" in text and "wächst" not in text
    kind, text = _verdict("Keine Falle (Nullstelle innen)")
    assert kind == "success" and "wächst" not in text and "destabil" not in text and "obwohl" not in text and "kein Horizont von N = 1 bis 150 ist instabil" in text
    kind, text = _verdict("Vier-Tank (R = 10⁻²)")
    assert kind == "warning" and "N = 10 gibt ρ = 1,028" in text and "um 2,8 % je Schritt" in text and "erst N = 20 ist stabil (ρ = 0,998)" in text and "offene Strecke stabil ist (ρ = 0,937)" in text


def test_preset_help_numbers_match_the_measurements():
    plant = ev.plant_of(_S())
    rho = D.rho_curve(plant, plant.Q, 12)
    helps = C.PRESET_HELP
    assert "1,495" in helps["Von Hand (Kalman, N = 1)"] and rho[0] == pytest.approx(1.495, abs=5e-4)
    assert "0,989" in helps["Kleinstes stabiles N (N = 7)"] and "9,31-fachen" in helps["Kleinstes stabiles N (N = 7)"] and rho[6] == pytest.approx(0.989, abs=5e-4)
    assert ev.price_curve(_S())[6] == pytest.approx(9.31, rel=1e-3)
    assert "0,664" in helps["Terminalkosten = Riccati-Lösung"] and ev.analyse(_S()).rho_inf == pytest.approx(0.664, abs=5e-4)
    assert "1 und 1/2" in helps["Wagen (N = 1 grenzstabil)"]
    assert "c = 2" in helps["Keine Falle (Nullstelle innen)"] and "0,5" in helps["Keine Falle (Nullstelle innen)"]
    assert "N = 10" in helps["Vier-Tank (R = 10⁻²)"] and "N = 20" in helps["Vier-Tank (R = 10⁻²)"]
    assert D.first_stable(D.rho_curve(D.build_plant("vier_tank", 1e-2), D.build_plant("vier_tank", 1e-2).Q, 150)) == 20


def test_numbers_in_the_static_app_texts():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "bei N = 1 bis 6 und ist erst ab N = 7 stabil" in app
    assert "ρ bei N = 55 nur etwa 0,3 % unter 1" in app
    a = ev.vier_tank_curve(1e-4, "q")
    assert 0.002 < 1.0 - a["rho"][54] < 0.004                           # „etwa 0,3 %“: 0,27 %
    assert re.search(r"91 Zellen bis N = 400", app) and re.search(r"28 Zellen bis N = 400", app)
    assert len(C.C_OPTIONS) * len(C.R_DECADES) == 91 and len(C.EPS_OPTIONS) * len(C.R_DECADES) == 28 and C.N_SEARCH == 400
    assert C.N_RANDOM_SYSTEMS == 6000 and len(C.SEEDS) == 20
