"""Kennzahlen, Verdikte und Messreihen der Demo. Alle Verdikte sind reine Funktionen (Art, Text); die App ruft st.<Art>(Text), `tests/test_verdicts.py` prüft Schwellen und
jede Zahl im Text an handgesetzten Werten, `tests/test_claims.py` die Zahlen der README und je Preset die erwartete Verdikt-Zeile.

Zähleinheit überall: Horizont N in Schritten; „instabil“ heißt ρ ≥ 1 − 10⁻⁹ (also auch grenzstabil), „stabil“ ρ < 1 − 10⁻⁹. Alles gilt im linearen Modell ohne Beschränkungen und
ohne Störung."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

import hfa_constants as C
import hfa_dynamics as D
import hfa_exact as X

ROOT = Path(__file__).resolve().parent
FROZEN_PATH = ROOT / "data" / "hfa_frozen.json"


# ------------------------------------------------------------------------------------------------------------------ Anzeige

def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def fmt_int(n: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (2047 -> 2 047)."""
    return f"{n:,}".replace(",", " ")


def pct(x: float, nd: int = 1) -> str:
    return de(100.0 * x, nd) + " %"


def sci(x: float, nd: int = 1) -> str:
    """Zahl in Zehnerpotenzschreibweise mit Dezimalkomma: 3,3·10⁻¹⁶."""
    mant, exp = f"{x:.{nd}e}".split("e")
    return f"{mant.replace('.', ',')}·10{str(int(exp)).translate(C.SUPERSCRIPT)}"


def r_label(R: float) -> str:
    """R als Text: Zehnerpotenzen unter 1 mit Hochzahl (10⁻³, 3·10⁻³), ab 1 als ganze Zahl."""
    if R >= 1.0:
        return fmt_int(int(round(R)))
    mant, exp = f"{R:.0e}".split("e")
    base = f"10{str(int(exp)).translate(C.SUPERSCRIPT)}"
    return base if mant == "1" else f"{mant}·{base}"


def c_label(c: float) -> str:
    """c und die zugehörige Nullstelle z₀ = 1/c der Strecke."""
    names = {2 / 3: "2/3"}
    ctext = names.get(c, de(c, 1) if abs(c * 10 - round(c * 10)) < 1e-9 else de(c, 2))
    if c == 0.0:
        return "c = 0 (keine Nullstelle)"
    return f"c = {ctext} (Nullstelle {de(1.0 / c, 2 if c < 1 else 3)})"


def ranges_text(ns: list[int]) -> str:
    """Horizonte als Text: [1, 2, 3, 4, 5, 6] -> „N = 1 bis 6“, [2, 5, 6, 7] -> „N = 2, 5 bis 7“, [] -> „keines“."""
    if not ns:
        return "keines"
    parts, start, prev = [], ns[0], ns[0]
    for n in ns[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else (f"{start}, {prev}" if prev == start + 1 else f"{start} bis {prev}"))
        if n is not None:
            start = prev = n
    return "N = " + ", ".join(parts)


def pf_label(kind: str, gamma: float = 1.0) -> str:
    if kind == "gamma_i":
        return "P_f = I" if gamma == 1.0 else f"P_f = {de(gamma, 1) if gamma < 10 else fmt_int(int(gamma))}·I"
    if kind == "gamma_pinf":
        return "P_f = P_∞" if gamma == 1.0 else f"P_f = {de(gamma, 1) if gamma < 10 else fmt_int(int(gamma))}·P_∞"
    return C.PF_LABELS[kind]


@lru_cache(maxsize=1)
def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------------ Analyse eines Falls

@dataclass(frozen=True)
class Settings:
    system: str = "kalman"
    c: float = C.DEFAULT_C
    eps: float = C.DEFAULT_EPS
    R: float = C.DEFAULT_R
    N: int = C.DEFAULT_N
    pf: str = C.DEFAULT_PF
    gamma: float = C.DEFAULT_GAMMA


def plant_of(s: Settings) -> D.Plant:
    return D.build_plant(s.system, s.R, s.c, s.eps)


@dataclass
class Analysis:
    settings: Settings
    label: str                      # Name der Strecke
    pf_label: str
    rho: np.ndarray                 # ρ_N für N = 1 bis N_MAX
    rho_inf: float
    rho_open: float
    rho_sel: float
    first_stable: int | None
    unstable: list[int]
    later_unstable: int
    ratio_sel: float                # Kostenverhältnis des gewählten N (inf, wenn nicht stabil)
    gain_sel: np.ndarray
    gain_inf: np.ndarray
    poles_sel: np.ndarray
    poles_inf: np.ndarray
    poles_open: np.ndarray
    traj_sel: np.ndarray
    traj_inf: np.ndarray
    open_stable: bool = field(init=False)

    def __post_init__(self):
        self.open_stable = self.rho_open < 1.0 - D.TOL


def analyse(s: Settings) -> Analysis:
    plant = plant_of(s)
    Pinf, Kinf = D.dare(plant)
    Pf = D.terminal_matrix(s.pf, plant, s.gamma, Pinf)
    gains, _ = D.horizon_gains(plant, Pf, C.N_MAX)
    rho = np.array([D.rho_of_gain(plant, K) for K in gains])
    K_sel = gains[s.N - 1]
    x0 = np.zeros(plant.n)
    x0[0] = 1.0
    steps = C.TRAJ_STEPS[s.system]
    return Analysis(
        settings=s, label=C.SYSTEM_SHORT[s.system], pf_label=pf_label(s.pf, s.gamma), rho=rho, rho_inf=D.rho_of_gain(plant, Kinf), rho_open=D.rho_open(plant),
        rho_sel=float(rho[s.N - 1]), first_stable=D.first_stable(rho), unstable=D.unstable_horizons(rho), later_unstable=D.later_unstable(rho),
        ratio_sel=D.cost_ratio(plant, K_sel, Pinf), gain_sel=K_sel, gain_inf=Kinf,
        poles_sel=np.linalg.eigvals(plant.A - plant.B @ K_sel), poles_inf=np.linalg.eigvals(plant.A - plant.B @ Kinf), poles_open=np.linalg.eigvals(plant.A),
        traj_sel=D.trajectory_norms(plant, K_sel, x0, steps), traj_inf=D.trajectory_norms(plant, Kinf, x0, steps),
    )


def growth_text(rho: float) -> str:
    """Wachstum je Schritt in Prozent: ≥ 10 % ganze Zahl, sonst eine Stelle."""
    g = (rho - 1.0) * 100.0
    return de(g, 0) + " %" if g >= 10.0 else de(g, 1) + " %"


# ------------------------------------------------------------------------------------------------------------------ Experiment 1: das Gesetz des Horizonts N

def verdict_horizon(name: str, pf: str, N: int, rho_sel: float, rho_inf: float, rho_open: float, first_stable: int | None, rho_first: float | None, unstable: list[int],
                    n_max: int = C.N_MAX) -> tuple[str, str]:
    """Verdikt der Kernfrage 1. `unstable` ist die Liste aller instabilen Horizonte N ≤ n_max (ρ ≥ 1 − 10⁻⁹), `rho_first` der Wert ρ beim kleinsten stabilen N."""
    head = f"{name}, {pf}: "
    first = f"erst N = {first_stable} ist stabil (ρ = {de(rho_first, 3)})" if first_stable is not None else f"bis N = {n_max} wird kein Horizont stabil"
    inf = f"N → ∞: ρ = {de(rho_inf, 3)}."
    if rho_sel >= 1.0 + D.TOL:
        if rho_open < 1.0 - D.TOL:
            extra = f"obwohl die offene Strecke stabil ist (ρ = {de(rho_open, 3)})"
        elif abs(rho_open - 1.0) <= D.TOL:
            extra = "die offene Strecke ist grenzstabil (ρ = 1)"
        else:
            extra = f"die offene Strecke ist selbst instabil (ρ = {de(rho_open, 3)})"
        return "warning", head + (f"N = {N} gibt ρ = {de(rho_sel, 3)}: der größte Pol liegt außerhalb des Einheitskreises, der Zustand wächst bei Anteil im instabilen Mode asymptotisch um {growth_text(rho_sel)} je Schritt, "
                                  f"{extra}; {first}. {inf}")
    if rho_sel >= 1.0 - D.TOL:
        return "warning", head + (f"N = {N} gibt ρ = 1 (Band): ein Pol liegt auf dem Einheitskreis, das Gesetz ist grenzstabil, der Zustand kommt nicht in jeder Richtung zur Ruhe; "
                                  f"{first}. {inf}")
    if unstable:
        return "info", head + f"N = {N} ist stabil (ρ = {de(rho_sel, 3)}); instabil sind {ranges_text(unstable)}, {first}. {inf}"
    return "success", head + f"N = {N} ist stabil (ρ = {de(rho_sel, 3)}), und kein Horizont von N = 1 bis {n_max} ist instabil (kein Fall der Falle in diesem Beispiel). {inf}"


def verdict_horizon_analysis(a: Analysis) -> tuple[str, str]:
    s = a.settings
    rho_first = None if a.first_stable is None else float(a.rho[a.first_stable - 1])
    return verdict_horizon(a.label, a.pf_label, s.N, a.rho_sel, a.rho_inf, a.rho_open, a.first_stable, rho_first, a.unstable)


# ------------------------------------------------------------------------------------------------------------------ Experiment 2: Terminalkosten

CANDIDATES = (
    ("P_f = 0", "zero", 0.0), ("P_f = Q", "Q", 1.0), ("P_f = 10·Q", "Q", 10.0), ("P_f = 1 000·Q", "Q", 1000.0),
    ("P_f = I", "I", 1.0), ("P_f = 100·I", "I", 100.0),
    ("P_f = 0,5·P_∞", "P", 0.5), ("P_f = 0,9·P_∞", "P", 0.9), ("P_f = P_∞", "P", 1.0), ("P_f = 1,1·P_∞", "P", 1.1), ("P_f = 2·P_∞", "P", 2.0), ("P_f = 1 000·P_∞", "P", 1000.0),
)


def candidate_matrix(plant: D.Plant, Pinf: np.ndarray, base: str, factor: float) -> np.ndarray:
    if base == "zero":
        return np.zeros((plant.n, plant.n))
    return factor * {"Q": plant.Q, "I": np.eye(plant.n), "P": Pinf}[base]


def nonmonotone_pair(rho12: list[float]) -> tuple[int, float, float] | None:
    """Erstes N, bei dem ρ von N auf N + 1 steigt: (N, ρ_N, ρ_{N+1}); None, wenn ρ_N über die Spalten nicht steigt."""
    for i in range(len(rho12) - 1):
        if rho12[i + 1] > rho12[i] + 1e-9:
            return i + 1, rho12[i], rho12[i + 1]
    return None


def terminal_table(s: Settings) -> dict:
    """Je Terminalgewicht (CANDIDATES) ρ_N für N = 1 bis 12 und die Liste der instabilen N bis N_MAX; dazu das gewählte Gewicht der Seitenleiste als erste Zeile,
    die größte Abweichung |K_N − K_∞| bei P_f = P_∞ und die harte Bedingung x(N) = 0 (N = 2 bis 12)."""
    plant = plant_of(s)
    Pinf, Kinf = D.dare(plant)
    rows = []
    items = [(pf_label(s.pf, s.gamma) + " (gewählt)", D.terminal_matrix(s.pf, plant, s.gamma, Pinf))] + [(lab, candidate_matrix(plant, Pinf, b, f)) for lab, b, f in CANDIDATES]
    dev = 0.0
    for i, (lab, Pf) in enumerate(items):
        gains, _ = D.horizon_gains(plant, Pf, C.N_MAX)
        rho = np.array([D.rho_of_gain(plant, K) for K in gains])
        unst = D.unstable_horizons(rho)
        if lab == "P_f = P_∞":
            dev = max(float(np.max(np.abs(K - Kinf))) for K in gains)
        rows.append({"label": lab, "chosen": i == 0, "rho12": [float(v) for v in rho[: C.N_TABLE]], "unstable": unst, "rho_max": float(np.max(rho)), "rho_all_equal": bool(np.ptp(rho) < 1e-9)})
    hard = []
    for N in range(2, C.N_TABLE + 1):
        K = D.hard_terminal_gain(plant, N)
        hard.append(None if K is None else D.rho_of_gain(plant, K))
    return {"rows": rows, "dev_pinf": dev, "hard": hard, "rho_inf": D.rho_of_gain(plant, Kinf)}


def verdict_terminal(name: str, table: dict, n_max: int = C.N_MAX) -> tuple[str, str]:
    rows = [r for r in table["rows"] if not r["chosen"]]
    pinf = next(r for r in rows if r["label"] == "P_f = P_∞")
    bad = [r for r in rows if r["unstable"]]
    good = [r for r in rows if not r["unstable"]]
    parts = [f"{name}: P_f = P_∞ gibt K_N = K_∞ für N = 1 bis {n_max} (größte Abweichung {sci(table['dev_pinf'])}), ρ = {de(table['rho_inf'], 5)} in {n_max} von {n_max} Horizonten"
             if pinf["rho_all_equal"] else f"{name}: P_f = P_∞ gibt hier nicht für jedes N dasselbe Gesetz (Abweichung {sci(table['dev_pinf'])})"]
    if bad:
        parts.append("instabile Horizonte: " + "; ".join(f"{r['label']}: {ranges_text(r['unstable'])}" for r in bad))
    if good:
        parts.append("kein instabiles N bei " + ", ".join(r["label"] for r in good if r["label"] != "P_f = P_∞"))
    cand = next((r for r in good if r["label"] == "P_f = 0,5·P_∞"), None)
    pair = nonmonotone_pair(cand["rho12"]) if cand else None
    if pair is None:
        for r in good:
            pair = nonmonotone_pair(r["rho12"])
            if pair is not None and r["label"] != "P_f = P_∞":
                cand = r
                break
            pair = None
    if pair is not None and cand is not None:
        parts.append(f"nicht monoton: bei {cand['label']} ist ρ({pair[0]}) = {de(pair[1], 3)}, aber ρ({pair[0] + 1}) = {de(pair[2], 3)}; ein längerer Horizont gibt hier nicht den kleineren Spektralradius")
    return "info", ". ".join(parts) + "."


# ------------------------------------------------------------------------------------------------------------------ Experiment 3: der Preis des kurzen Horizonts

def price_curve(s: Settings) -> np.ndarray:
    """Kostenverhältnis (schlechtester Anfangszustand, Kosten der unendlichen Zukunft unter dem Gesetz K_N durch die optimalen) für N = 1 bis N_PRICE; inf, wenn nicht stabil."""
    plant = plant_of(s)
    Pinf, _ = D.dare(plant)
    gains, _ = D.horizon_gains(plant, D.terminal_matrix(s.pf, plant, s.gamma, Pinf), C.N_PRICE)
    return np.array([D.cost_ratio(plant, K, Pinf) for K in gains])


def first_within(ratios: np.ndarray, limit: float = 1.01) -> int | None:
    """Kleinstes N, ab dem das Verhältnis höchstens `limit` beträgt (und danach im gerechneten Bereich bleibt)."""
    ok = np.isfinite(ratios) & (ratios <= limit)
    for i in range(len(ratios)):
        if ok[i:].all():
            return i + 1
    return None


def verdict_price(name: str, pf: str, N: int, ratio_sel: float, ratios: np.ndarray, rho_sel: float) -> tuple[str, str]:
    head = f"{name}, {pf}: "
    within = first_within(ratios)
    w = f" Ab N = {within} liegen die Kosten höchstens 1 % über dem Optimum." if within is not None else f" Im gerechneten Bereich (N ≤ {len(ratios)}) bleibt das Verhältnis über 1,01."
    if not np.isfinite(ratio_sel):
        state = "grenzstabil (ρ = 1)" if rho_sel < 1.0 + D.TOL else f"nicht stabil (ρ = {de(rho_sel, 3)})"
        return "warning", head + f"N = {N} ist {state}: die Kosten der unendlichen Zukunft sind im schlechtesten Anfangszustand nicht endlich." + w
    if ratio_sel > 1.01:
        return "warning", head + f"N = {N} ist stabil (ρ = {de(rho_sel, 3)}), kostet im schlechtesten Anfangszustand aber das {de(ratio_sel, 2 if ratio_sel < 10 else 1)}-fache der optimalen Kosten." + w
    return "success", head + f"N = {N} ist stabil und im schlechtesten Anfangszustand höchstens {de(ratio_sel, 4)}-mal so teuer wie das optimale Gesetz." + w


def timevarying_rows(s: Settings, n_max: int = C.N_PRICE) -> list[dict]:
    """Der Plan über N Schritte als zeitvarianter Ablauf K_N, K_{N−1}, …, K_1: geometrisches Mittel des Spektralradius pro Schritt und größter Singulärwert des Produkts."""
    plant = plant_of(s)
    Pinf, _ = D.dare(plant)
    gains, _ = D.horizon_gains(plant, D.terminal_matrix(s.pf, plant, s.gamma, Pinf), n_max)
    rows = []
    for N in range(1, n_max + 1):
        g, sv = D.time_varying_summary(plant, gains, N)
        rows.append({"N": N, "rho_repeat": D.rho_of_gain(plant, gains[N - 1]), "geo": g, "sigma": sv})
    return rows


def verdict_timevarying(name: str, pf: str, rows: list[dict]) -> tuple[str, str]:
    above = [r["N"] for r in rows if r["geo"] >= 1.0 - D.TOL]
    peak = max(rows, key=lambda r: r["sigma"])
    if not above:
        return "success", f"{name}, {pf}: der zeitvariante Ablauf K_N, …, K_1 hat für jedes N bis {rows[-1]['N']} ein geometrisches Mittel unter 1; der größte Singulärwert des Produkts ist höchstens {de(peak['sigma'], 2)} (bei N = {peak['N']})."
    first = rows[above[0] - 1]
    return "warning", (f"{name}, {pf}: der zeitvariante Ablauf K_N, …, K_1 hat für {ranges_text(above)} ein geometrisches Mittel pro Schritt von mindestens 1 (N = {first['N']}: {de(first['geo'], 3)}); "
                      f"der größte Singulärwert des Produkts steigt bis {de(peak['sigma'], 2)} (bei N = {peak['N']}).")


# ------------------------------------------------------------------------------------------------------------------ Experiment 4: wann gibt es die Falle?

NULL_PLANT_R = 1e-3
NULL_PLANT_EPS = 1e-3


def zero_class(c: float) -> str:
    """Lage der Nullstelle 1/c zum Einheitskreis: „außerhalb“ (0 < c < 1), „auf dem Kreis“ (c = 1), „innerhalb“ (c > 1) oder „keine“ (c = 0)."""
    if c == 0.0:
        return "keine"
    if c == 1.0:
        return "auf dem Kreis"
    return "außerhalb" if c < 1.0 else "innerhalb"


def trap_cell(c: float, eps: float, R: float, n_search: int = C.N_SEARCH) -> dict:
    """Kalman-Familie mit P_f = Q: ρ_N bis n_search; kleinstes stabiles N, Zahl der instabilen N, Zahl der instabilen N hinter dem kleinsten stabilen."""
    plant = D.build_plant("kalman", R, c, eps)
    rho = D.rho_curve(plant, plant.Q, n_search)
    return {"first_stable": D.first_stable(rho), "n_unstable": len(D.unstable_horizons(rho)), "later_unstable": D.later_unstable(rho), "rho": [float(v) for v in rho[:12]],
            "rho_inf": D.rho_of_gain(plant, D.dare(plant)[1])}


def nullstelle_rows() -> list[dict]:
    """Die 13 Werte von c (Nullstelle 1/c) bei ε = 10⁻³, R = 10⁻³, P_f = Q."""
    rows = []
    for c in C.C_OPTIONS:
        cell = trap_cell(c, NULL_PLANT_EPS, NULL_PLANT_R)
        rows.append({"c": c, "zero": None if c == 0.0 else 1.0 / c, "class": zero_class(c), "first_stable": cell["first_stable"], "n_unstable": cell["n_unstable"],
                     "later_unstable": cell["later_unstable"], "rho1": cell["rho"][0], "rho_inf": cell["rho_inf"]})
    return rows


def count_nullstelle(rows: list[dict]) -> dict:
    out = [r for r in rows if r["class"] == "außerhalb"]
    rest = [r for r in rows if r["class"] != "außerhalb"]
    return {"outside": len(out), "outside_trap": sum(1 for r in out if r["n_unstable"] > 0), "rest": len(rest), "rest_trap": sum(1 for r in rest if r["n_unstable"] > 0),
            "c_min": min(r["c"] for r in out), "c_max": max(r["c"] for r in out)}


def map_c_R(eps: float = NULL_PLANT_EPS) -> dict:
    """Kleinstes stabiles N über c (Zeilen) und R (Spalten, 7 Zehnerpotenzen) bei festem ε, P_f = Q; dazu die Zahl der Zellen mit einem instabilen N hinter dem kleinsten stabilen."""
    cells = [[trap_cell(c, eps, R) for R in C.R_DECADES] for c in C.C_OPTIONS]
    return {"c": list(C.C_OPTIONS), "R": list(C.R_DECADES), "first": [[x["first_stable"] for x in row] for row in cells], "later": sum(x["later_unstable"] > 0 for row in cells for x in row)}


def map_R_eps(c: float = C.DEFAULT_C) -> dict:
    """Kleinstes stabiles N über ε (Zeilen) und R (Spalten) bei festem c, P_f = Q."""
    cells = [[trap_cell(c, e, R) for R in C.R_DECADES] for e in C.EPS_OPTIONS]
    return {"eps": list(C.EPS_OPTIONS), "R": list(C.R_DECADES), "first": [[x["first_stable"] for x in row] for row in cells], "later": sum(x["later_unstable"] > 0 for row in cells for x in row)}


def random_row(seed: int) -> dict:
    return next(r for r in load_frozen()["random"] if r["seed"] == seed)


def random_summary() -> dict:
    """Kennzahlen über alle vorgerechneten Zufalls-Seeds: Spanne des Anteils „N = 1 instabil“ unter den offen stabilen Strecken, größte Abweichung K_N − K_∞ bei P_f = P_∞."""
    rows = load_frozen()["random"]
    shares = [r["n1_unstable"] / r["open_stable"] for r in rows]
    return {"seeds": len(rows), "systems": sum(r["valid"] for r in rows), "share_min": min(shares), "share_max": max(shares), "share_mean": sum(shares) / len(shares),
            "dev_max": max(r["dev_max"] for r in rows), "extra_any": sum(r["any_unstable"] - r["n1_unstable"] for r in rows)}


def verdict_trap(counts: dict, rand: dict, cell: dict | None, c: float, R: float, eps: float, seed: int) -> tuple[str, str]:
    """Verdikt des Experiments 4. `counts` aus count_nullstelle, `rand` die vorgerechnete Zeile des Zufalls-Seeds, `cell` die Zelle (c, ε, R) der Seitenleiste (None, wenn die Strecke der Seitenleiste nicht Kalmans Familie ist)."""
    share = rand["n1_unstable"] / rand["open_stable"]
    text = (f"Nullstelle außerhalb des Einheitskreises (c = {de(counts['c_min'], 1)} bis {de(counts['c_max'], 1)}): die Falle tritt in {counts['outside_trap']} von {counts['outside']} gemessenen Fällen auf, "
            f"sonst in {counts['rest_trap']} von {counts['rest']}; das gilt bei ε = {r_label(NULL_PLANT_EPS)}, R = {r_label(NULL_PLANT_R)}, P_f = Q; Abhängigkeit, nicht Ursache. "
            f"Zufallssysteme (Zufalls-Seed {seed}): bei {fmt_int(rand['n1_unstable'])} von {fmt_int(rand['open_stable'])} offen stabilen Strecken ({pct(share)}) ist N = 1 instabil.")
    if cell is not None:
        text += f" Der gewählte Fall ({c_label(c)}, ε = {r_label(eps)}, R = {r_label(R)}, P_f = Q): " + (
            f"kleinstes stabiles N = {cell['first_stable']}, {cell['n_unstable']} instabile N bis {C.N_SEARCH}." if cell["n_unstable"] else f"kein instabiles N bis {C.N_SEARCH}.")
    return "info", text


# ------------------------------------------------------------------------------------------------------------------ Experiment 5: der Wagen

def parse_fraction(text: str) -> Fraction:
    return Fraction(text)


def wagen_rows(R: Fraction, n_max: int = 3) -> list[dict]:
    """Der Wagen mit Q = I, P_f = Q, N = 1 bis n_max in Brüchen: Gewinn, Spur und Determinante der Matrix A − B K_N und ρ (aus Spur und Determinante)."""
    A = X.frac_matrix([[1, 1], [0, 1]])
    B = X.frac_matrix([[0], [1]])
    Q = X.frac_matrix([[1, 0], [0, 1]])
    rows = []
    for N, (K, _) in enumerate(X.horizon_gains_exact(A, B, Q, [[R]], Q, n_max), start=1):
        spur, det = X.trace_det(X.closed_loop_exact(A, B, K))
        rows.append({"N": N, "K": K[0], "trace": spur, "det": det, "rho": X.rho_from_trace_det(spur, det)})
    return rows


def wagen_endpoint(R: Fraction, x0: tuple[float, float]) -> float:
    """Endlage des Wagens bei N = 1 (K_1 = (0, 1/(R + 1))): x₁(∞) = x₁(0) + (R + 1)·x₂(0); der Vektor (1, R + 1) ist Linkseigenvektor zum Eigenwert 1."""
    return x0[0] + float(R + 1) * x0[1]


def wagen_paths(R: Fraction, x0: tuple[float, float], steps: int = 25) -> dict:
    """Zustände (Position, Geschwindigkeit) des Wagens unter den Gesetzen N = 1, 2, 3 und ∞ (P_f = Q bzw. Riccati-Lösung)."""
    plant = D.build_plant("wagen", float(R))
    gains, _ = D.horizon_gains(plant, plant.Q, 3)
    _, Kinf = D.dare(plant)
    x = np.array(x0, dtype=float)
    out = {f"N = {i + 1}": D.trajectory_states(plant, gains[i], x, steps) for i in range(3)}
    out["N → ∞"] = D.trajectory_states(plant, Kinf, x, steps)
    return out


def verdict_wagen(R: Fraction, rows: list[dict], rho_inf: float, x0: tuple[float, float]) -> tuple[str, str]:
    r1, r2 = rows[0], rows[1]
    Rt = X.fraction_text(R)
    second = X.fraction_text(R / (R + 1))
    end = wagen_endpoint(R, x0)
    coef = X.fraction_text(R + 1)
    return "info", (f"R = {Rt}, N = 1: Eigenwerte 1 und {second}; der Wagen bremst seine Geschwindigkeit, bringt die Position aber nicht zurück: er kommt bei x₁(0) + {coef}·x₂(0) = {de(end, 2)} zur Ruhe statt bei 0 "
                    f"(Anfangszustand ({de(x0[0], 1)}; {de(x0[1], 1)})). N = 2: ρ = {de(r2['rho'], 3)}, stabil; N → ∞: ρ = {de(rho_inf, 3)}.")


# ------------------------------------------------------------------------------------------------------------------ Experiment 6: Vier-Tank

def vier_tank_curve(R: float, pf: str, n_max: int = C.N_MAX) -> dict:
    plant = D.build_plant("vier_tank", R)
    Pinf, Kinf = D.dare(plant)
    Pf = D.terminal_matrix(pf, plant, 1.0, Pinf)
    rho = D.rho_curve(plant, Pf, n_max)
    gains_d, _ = D.horizon_gains(plant, Pinf, 20)
    scale = max(float(np.max(np.abs(Kinf))), 1.0)
    dev = max(float(np.max(np.abs(K - Kinf))) for K in gains_d) / scale
    return {"rho": rho, "rho_inf": D.rho_of_gain(plant, Kinf), "rho_open": D.rho_open(plant), "unstable": D.unstable_horizons(rho), "first_stable": D.first_stable(rho), "dev20": dev}


def vier_tank_table() -> list[dict]:
    """Je R (7 Zehnerpotenzen) und P_f ∈ {C′QC (= Q), 0}: die instabilen Horizonte N ≤ 150."""
    rows = []
    for R in C.R_DECADES:
        row = {"R": R}
        for key, label in (("q", "Q"), ("null", "null")):
            cur = vier_tank_curve(R, key)
            row[label] = cur["unstable"]
        row["rho_inf"] = cur["rho_inf"]
        row["dev20"] = cur["dev20"]
        rows.append(row)
    return rows


def pf_name_vier_tank(pf: str) -> str:
    return {"q": "P_f = C′QC (= Q)", "null": "P_f = 0", "gamma_pinf": "P_f = P_∞ (Riccati-Lösung)"}[pf]


def verdict_vier_tank(R: float, pf: str, cur: dict) -> tuple[str, str]:
    unst, first = cur["unstable"], cur["first_stable"]
    base = f"Vier-Tank-System (modellbasiert), R = {r_label(R)}, {pf_name_vier_tank(pf)}: "
    if not unst:
        return "success", base + f"kein Horizont von N = 1 bis {C.N_MAX} ist instabil (ρ_∞ = {de(cur['rho_inf'], 4)}, offene Strecke {de(cur['rho_open'], 3)}); das Gesetz K_N ist für jedes N stabil. " \
            f"Mit P_f = P_∞ ist K_N = K_∞ für N = 1 bis 20 (größte relative Abweichung {sci(cur['dev20'])})."
    kind = "warning"
    return kind, base + (f"{ranges_text(unst)} instabil" + (f", ab N = {first} stabil" if first is not None else f"; bis N = {C.N_MAX} wird kein Horizont stabil") +
                         f" (ρ_∞ = {de(cur['rho_inf'], 4)}, offene Strecke {de(cur['rho_open'], 3)}). Mit P_f = P_∞ ist K_N = K_∞ für N = 1 bis 20 (größte relative Abweichung {sci(cur['dev20'])}). "
                         "Das ist das modellbasierte Gegenstück, nicht das datenbasierte Schema der Quelle.")


# ------------------------------------------------------------------------------------------------------------------ Tabellen (Anzeige)

def rho_cell(rho: float | None) -> str:
    """Zelle der Tabellen: ρ mit drei Stellen, ▲ bei ρ ≥ 1 (nicht asymptotisch stabil), „–“ ohne Wert."""
    if rho is None:
        return "–"
    return de(rho, 3) + (" ▲" if rho >= 1.0 - D.TOL else "")


def terminal_frame(table: dict, n_max: int = C.N_MAX) -> list[dict]:
    """Zeilen der Terminalkosten-Tabelle (Texte): gewähltes Gewicht zuerst, dann die zwölf festen, zuletzt die harte Bedingung x(N) = 0."""
    out = []
    for r in table["rows"]:
        row = {"Terminalgewicht": ("► " if r["chosen"] else "") + r["label"]}
        for N in range(1, C.N_TABLE + 1):
            row[f"N = {N}"] = rho_cell(r["rho12"][N - 1])
        row[f"instabile N (bis {n_max})"] = ranges_text(r["unstable"]).replace("N = ", "") if r["unstable"] else "keine"
        out.append(row)
    row = {"Terminalgewicht": "harte Bedingung x(N) = 0"}
    for N in range(1, C.N_TABLE + 1):
        row[f"N = {N}"] = "–" if N == 1 else rho_cell(table["hard"][N - 2])
    row[f"instabile N (bis {n_max})"] = "–"
    out.append(row)
    return out


def timevarying_frame(rows: list[dict]) -> list[dict]:
    return [{"N": r["N"], "ρ des wiederholten Gesetzes K_N": de(r["rho_repeat"], 3), "zeitvariant: geometrisches Mittel pro Schritt": de(r["geo"], 3), "zeitvariant: größter Singulärwert": de(r["sigma"], 2)}
            for r in rows[: C.N_TABLE]]


def nullstelle_frame(rows: list[dict]) -> list[dict]:
    return [{"c": "2/3" if r["c"] == 2 / 3 else de(r["c"], 1), "Nullstelle 1/c": "–" if r["zero"] is None else de(r["zero"], 3), "Lage zum Einheitskreis": r["class"],
             "ρ bei N = 1": de(r["rho1"], 3), "ρ bei N → ∞": de(r["rho_inf"], 3), "kleinstes stabiles N": str(r["first_stable"]), "instabile N (bis 400)": str(r["n_unstable"])} for r in rows]


def wagen_frame(rows: list[dict], rho_inf: float) -> list[dict]:
    out = [{"N": str(r["N"]), "Gewinn K_N (Brüche)": "(" + ", ".join(X.fraction_text(v) for v in r["K"]) + ")", "Spur von A − B K_N": X.fraction_text(r["trace"]),
            "Determinante": X.fraction_text(r["det"]), "ρ": de(r["rho"], 4) + (" (grenzstabil)" if abs(r["rho"] - 1.0) < D.TOL else "")} for r in rows]
    out.append({"N": "→ ∞", "Gewinn K_N (Brüche)": "(DARE, Zahlen)", "Spur von A − B K_N": "–", "Determinante": "–", "ρ": de(rho_inf, 4)})
    return out


def vier_tank_frame(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({"R": r_label(r["R"]), "instabile N bei P_f = Q": ranges_text(r["Q"]).replace("N = ", "") if r["Q"] else "keine", "instabile N bei P_f = 0": ranges_text(r["null"]).replace("N = ", "") if r["null"] else "keine",
                    "ρ bei N → ∞": de(r["rho_inf"], 4)})
    return out
