"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt
Werte, nicht Seeds. Die Zeilen je Preset (echte Strecken) stehen in test_claims.py."""

from fractions import Fraction as Fr

import numpy as np

import hfa_evaluation as ev


# ------------------------------------------------------------------------------------------------------------------ Anzeigehilfen

def test_display_helpers():
    assert ev.de(1.4953, 3) == "1,495" and ev.de(-0.5, 1) == "−0,5"
    assert ev.fmt_int(2047) == "2 047" and ev.pct(0.0327) == "3,3 %"
    assert ev.sci(3.3306e-16) == "3,3·10⁻¹⁶" and ev.sci(2.94e-6) == "2,9·10⁻⁶"
    assert ev.r_label(1e-3) == "10⁻³" and ev.r_label(3e-3) == "3·10⁻³" and ev.r_label(1.0) == "1" and ev.r_label(100.0) == "100"
    assert ev.ranges_text([1, 2, 3, 4, 5, 6]) == "N = 1 bis 6" and ev.ranges_text([2, 5, 6, 7]) == "N = 2, 5 bis 7" and ev.ranges_text([3]) == "N = 3"
    assert ev.ranges_text([1, 2]) == "N = 1, 2" and ev.ranges_text([]) == "keines"
    assert ev.growth_text(1.30673) == "31 %" and ev.growth_text(1.0008) == "0,1 %"
    assert ev.c_label(2 / 3) == "c = 2/3 (Nullstelle 1,50)" and ev.c_label(0.0) == "c = 0 (keine Nullstelle)" and ev.c_label(2.0) == "c = 2,0 (Nullstelle 0,500)"
    assert ev.pf_label("gamma_pinf", 1.0) == "P_f = P_∞" and ev.pf_label("gamma_i", 10.0) == "P_f = 10·I" and ev.pf_label("gamma_i", 0.5) == "P_f = 0,5·I"
    assert ev.rho_cell(1.0) == "1,000 ▲" and ev.rho_cell(1.0 - 1e-10) == "1,000 ▲" and ev.rho_cell(1.0 - 1e-6) == "1,000" and ev.rho_cell(0.9) == "0,900" and ev.rho_cell(None) == "–"


# ------------------------------------------------------------------------------------------------------------------ Experiment 1

def test_verdict_horizon_unstable_names_growth_open_loop_first_stable_and_limit():
    kind, text = ev.verdict_horizon("Kalmans Gegenbeispiel", "P_f = Q", 5, 1.30673, 0.66404, 0.81649, 7, 0.98857, [1, 2, 3, 4, 5, 6])
    assert kind == "warning"
    assert "N = 5 gibt ρ = 1,307" in text and "um 31 % je Schritt" in text and "obwohl die offene Strecke stabil ist (ρ = 0,816)" in text
    assert "erst N = 7 ist stabil (ρ = 0,989)" in text and "N → ∞: ρ = 0,664" in text


def test_verdict_horizon_open_loop_variants_and_no_stable_horizon():
    _, t1 = ev.verdict_horizon("A", "P_f = Q", 1, 1.2, 0.5, 1.0, 2, 0.5, [1])
    assert "die offene Strecke ist grenzstabil" in t1
    _, t2 = ev.verdict_horizon("A", "P_f = Q", 1, 1.2, 0.5, 1.4, 2, 0.5, [1])
    assert "die offene Strecke ist selbst instabil (ρ = 1,400)" in t2 and "obwohl" not in t2
    kind, t3 = ev.verdict_horizon("A", "P_f = Q", 3, 1.2, 0.5, 0.9, None, None, [1, 2, 3], n_max=3)
    assert kind == "warning" and "bis N = 3 wird kein Horizont stabil" in t3


def test_verdict_horizon_threshold_at_rho_one():
    assert ev.verdict_horizon("A", "P", 1, 1.0 + 2e-9, 0.5, 0.9, 2, 0.5, [1])[1].count("wächst") == 1       # knapp über 1: instabil
    kind, text = ev.verdict_horizon("A", "P", 1, 1.0 + 1e-10, 0.5, 0.9, 2, 0.5, [1])                       # im Band: grenzstabil
    assert kind == "warning" and "ρ = 1 (Band)" in text and "grenzstabil" in text and "wächst" not in text
    kind, text = ev.verdict_horizon("A", "P", 1, 1.0 - 1e-10, 0.5, 0.9, 2, 0.5, [1])
    assert "grenzstabil" in text
    kind, text = ev.verdict_horizon("A", "P", 3, 1.0 - 1e-6, 0.5, 0.9, 1, 0.9, [])
    assert kind == "success"


def test_verdict_horizon_stable_with_and_without_unstable_horizons():
    kind, text = ev.verdict_horizon("A", "P_f = Q", 10, 0.707, 0.664, 0.816, 7, 0.989, [1, 2, 3, 4, 5, 6])
    assert kind == "info" and "N = 10 ist stabil (ρ = 0,707)" in text and "instabil sind N = 1 bis 6" in text and "erst N = 7 ist stabil (ρ = 0,989)" in text
    kind, text = ev.verdict_horizon("A", "P_f = Q", 1, 0.5, 0.4, 0.9, 1, 0.5, [], n_max=150)
    assert kind == "success" and "kein Horizont von N = 1 bis 150 ist instabil" in text and "N → ∞: ρ = 0,400" in text


# ------------------------------------------------------------------------------------------------------------------ Experiment 2

def _row(label, rho12, unstable, chosen=False, equal=False):
    return {"label": label, "chosen": chosen, "rho12": rho12, "unstable": unstable, "rho_max": max(rho12), "rho_all_equal": equal}


def test_nonmonotone_pair():
    assert ev.nonmonotone_pair([0.9, 0.8, 0.8, 0.7]) is None
    assert ev.nonmonotone_pair([0.664, 0.920, 0.8]) == (1, 0.664, 0.920)
    assert ev.nonmonotone_pair([0.9, 0.8, 0.85]) == (2, 0.8, 0.85)


def test_verdict_terminal_lists_unstable_stable_and_the_nonmonotone_case():
    table = {"rows": [_row("P_f = Q (gewählt)", [1.5, 0.5], [1], chosen=True), _row("P_f = Q", [1.5, 0.5], [1, 2]), _row("P_f = I", [0.1, 0.2], []),
                      _row("P_f = 0,5·P_∞", [0.664, 0.92], []), _row("P_f = P_∞", [0.664, 0.664], [], equal=True)],
             "dev_pinf": 3.3e-16, "hard": [], "rho_inf": 0.66404}
    kind, text = ev.verdict_terminal("Beispiel", table, n_max=150)
    assert kind == "info"
    assert "P_f = P_∞ gibt K_N = K_∞ für N = 1 bis 150 (größte Abweichung 3,3·10⁻¹⁶), ρ = 0,66404 in 150 von 150 Horizonten" in text
    assert "instabile Horizonte: P_f = Q: N = 1, 2" in text and "kein instabiles N bei P_f = I, P_f = 0,5·P_∞" in text
    assert "nicht monoton: bei P_f = 0,5·P_∞ ist ρ(1) = 0,664, aber ρ(2) = 0,920" in text
    table["rows"][-1]["rho_all_equal"] = False
    assert "gibt hier nicht für jedes N dasselbe Gesetz" in ev.verdict_terminal("Beispiel", table, 150)[1]


def test_terminal_frame_marks_unstable_cells_and_adds_the_hard_condition_row():
    table = {"rows": [_row("P_f = Q (gewählt)", [1.2] + [0.5] * 11, [1], chosen=True)], "hard": [0.0] + [0.4] * 10, "dev_pinf": 0.0, "rho_inf": 0.4}
    frame = ev.terminal_frame(table, 150)
    assert frame[0]["Terminalgewicht"] == "► P_f = Q (gewählt)" and frame[0]["N = 1"] == "1,200 ▲" and frame[0]["N = 2"] == "0,500" and frame[0]["instabile N (bis 150)"] == "1"
    assert frame[1]["Terminalgewicht"] == "harte Bedingung x(N) = 0" and frame[1]["N = 1"] == "–" and frame[1]["N = 2"] == "0,000" and frame[1]["N = 12"] == "0,400"


# ------------------------------------------------------------------------------------------------------------------ Experiment 3

def test_verdict_price_branches_and_thresholds():
    ratios = np.array([np.inf, np.inf, 9.31, 1.21, 1.0066, 1.001])
    kind, text = ev.verdict_price("A", "P_f = Q", 3, 9.31, ratios, 0.989)
    assert kind == "warning" and "N = 3 ist stabil (ρ = 0,989)" in text and "das 9,31-fache der optimalen Kosten" in text and "Ab N = 5 liegen die Kosten höchstens 1 % über dem Optimum" in text
    kind, text = ev.verdict_price("A", "P_f = Q", 1, np.inf, ratios, 1.495)
    assert kind == "warning" and "N = 1 ist nicht stabil (ρ = 1,495)" in text and "nicht endlich" in text
    assert "grenzstabil (ρ = 1)" in ev.verdict_price("A", "P_f = Q", 1, np.inf, ratios, 1.0)[1]
    assert ev.verdict_price("A", "P_f = Q", 5, 1.01, ratios, 0.5)[0] == "success"                              # genau 1,01: noch gut
    kind, text = ev.verdict_price("A", "P_f = Q", 5, 1.0101, ratios, 0.5)
    assert kind == "warning" and "das 1,01-fache der optimalen Kosten" in text
    kind, text = ev.verdict_price("A", "P_f = Q", 5, 1.0066, ratios, 0.5)
    assert kind == "success" and "höchstens 1,0066-mal so teuer" in text
    assert "über 1,01" in ev.verdict_price("A", "P", 1, 1.5, np.array([1.5, 1.4]), 0.5)[1]


def test_first_within():
    assert ev.first_within(np.array([np.inf, 1.5, 1.0066, 1.001, 1.0])) == 3
    assert ev.first_within(np.array([1.0, 1.02, 1.0])) == 3                                                    # danach bleibt es innerhalb
    assert ev.first_within(np.array([1.5, 1.4])) is None


def test_verdict_timevarying():
    rows = [{"N": 1, "rho_repeat": 1.495, "geo": 1.495, "sigma": 1.8}, {"N": 2, "rho_repeat": 1.48, "geo": 1.01, "sigma": 8.28}, {"N": 3, "rho_repeat": 0.7, "geo": 0.9, "sigma": 2.0}]
    kind, text = ev.verdict_timevarying("A", "P_f = Q", rows)
    assert kind == "warning" and "für N = 1, 2 ein geometrisches Mittel pro Schritt von mindestens 1 (N = 1: 1,495)" in text and "bis 8,28 (bei N = 2)" in text
    rows[0]["geo"] = rows[1]["geo"] = 0.9
    kind, text = ev.verdict_timevarying("A", "P_f = Q", rows)
    assert kind == "success" and "für jedes N bis 3 ein geometrisches Mittel unter 1" in text


# ------------------------------------------------------------------------------------------------------------------ Experiment 4

def test_verdict_trap_counts_seed_and_selected_cell():
    counts = {"outside": 8, "outside_trap": 8, "rest": 5, "rest_trap": 0, "c_min": 0.1, "c_max": 0.9}
    rand = {"open_stable": 2047, "n1_unstable": 67}
    cell = {"first_stable": 7, "n_unstable": 6}
    kind, text = ev.verdict_trap(counts, rand, cell, 2 / 3, 1e-3, 1e-3, 12)
    assert kind == "info"
    assert "(c = 0,1 bis 0,9): die Falle tritt in 8 von 8 gemessenen Fällen auf, sonst in 0 von 5" in text and "Abhängigkeit, nicht Ursache" in text
    assert "Zufalls-Seed 12): bei 67 von 2 047 offen stabilen Strecken (3,3 %) ist N = 1 instabil" in text
    assert "kleinstes stabiles N = 7, 6 instabile N bis 400" in text
    assert "Der gewählte Fall" not in ev.verdict_trap(counts, rand, None, 2 / 3, 1e-3, 1e-3, 12)[1]
    assert "kein instabiles N bis 400" in ev.verdict_trap(counts, rand, {"first_stable": 1, "n_unstable": 0}, 2.0, 1e-3, 1e-3, 12)[1]


def test_zero_class_and_counts():
    assert [ev.zero_class(c) for c in (3.0, 1.0, 0.9, 0.0)] == ["innerhalb", "auf dem Kreis", "außerhalb", "keine"]
    rows = [{"c": 0.5, "class": "außerhalb", "n_unstable": 4}, {"c": 0.2, "class": "außerhalb", "n_unstable": 0}, {"c": 2.0, "class": "innerhalb", "n_unstable": 0}, {"c": 1.0, "class": "auf dem Kreis", "n_unstable": 1}]
    assert ev.count_nullstelle(rows) == {"outside": 2, "outside_trap": 1, "rest": 2, "rest_trap": 1, "c_min": 0.2, "c_max": 0.5}


# ------------------------------------------------------------------------------------------------------------------ Experiment 5 und 6

def test_verdict_wagen_names_eigenvalues_endpoint_and_n2():
    rows = [{"N": 1, "rho": 1.0}, {"N": 2, "rho": 0.53452}]
    kind, text = ev.verdict_wagen(Fr(1), rows, 0.42208, (2.0, 1.0))
    assert kind == "info"
    assert "R = 1, N = 1: Eigenwerte 1 und 1/2" in text and "x₁(0) + 2·x₂(0) = 4,00" in text and "N = 2: ρ = 0,535, stabil; N → ∞: ρ = 0,422" in text
    assert "Eigenwerte 1 und 1/11" in ev.verdict_wagen(Fr(1, 10), rows, 0.36, (2.0, 1.0))[1] and "x₁(0) + 11/10·x₂(0)" in ev.verdict_wagen(Fr(1, 10), rows, 0.36, (2.0, 1.0))[1]


def test_wagen_endpoint_by_hand():
    assert ev.wagen_endpoint(Fr(1), (2.0, 1.0)) == 4.0 and ev.wagen_endpoint(Fr(1), (0.0, 0.0)) == 0.0 and ev.wagen_endpoint(Fr(1), (-1.0, 0.5)) == 0.0
    assert ev.wagen_endpoint(Fr(1, 10), (1.0, 1.0)) == 2.1


def test_verdict_vier_tank_branches():
    cur = {"unstable": list(range(1, 20)), "first_stable": 20, "rho_inf": 0.9426, "rho_open": 0.937, "dev20": 4.6e-15}
    kind, text = ev.verdict_vier_tank(1e-2, "q", cur)
    assert kind == "warning" and "R = 10⁻², P_f = C′QC (= Q): N = 1 bis 19 instabil, ab N = 20 stabil (ρ_∞ = 0,9426, offene Strecke 0,937)" in text
    assert "größte relative Abweichung 4,6·10⁻¹⁵" in text and "nicht das datenbasierte Schema der Quelle" in text
    cur2 = {"unstable": [], "first_stable": 1, "rho_inf": 0.9404, "rho_open": 0.937, "dev20": 7.0e-15}
    kind, text = ev.verdict_vier_tank(1e-1, "null", cur2)
    assert kind == "success" and "kein Horizont von N = 1 bis 150 ist instabil" in text
    cur3 = {"unstable": list(range(1, 151)), "first_stable": None, "rho_inf": 0.9, "rho_open": 0.9, "dev20": 0.0}
    assert "bis N = 150 wird kein Horizont stabil" in ev.verdict_vier_tank(1e-4, "q", cur3)[1]
