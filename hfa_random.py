"""Zufallssysteme für die Gegenprobe der Absolutaussagen (Experiment 4): n = 2 Zustände, ein Eingang, Gewichte Q = q·I und R = r.

Ziehung je System (SplitMix64 mit Zufalls-Seed, `hfa_rng.py`): A (2×2) und B (2×1) standardnormal, q = 10^U(−3, 0), r = 10^U(−3, 1). Gültig ist ein System, wenn es steuerbar ist
(|det [B, AB]| ≥ 10⁻³); dann hat die Riccati-Gleichung für Q > 0 eine stabilisierende Lösung. Gezählt wird, wie oft das Gesetz des Horizonts N mit Terminalgewicht P_f = Q eine
**offen stabile** Strecke (ρ(A) < 1) destabilisiert, und wie weit K_N bei P_f = P_∞ vom unendlichen Gewinn abweicht. Eigene Messung in diesem Modell; die Familie ist eine Wahl, keine
Aussage über „typische“ Systeme."""

from __future__ import annotations

import numpy as np

import hfa_dynamics as D
from hfa_rng import SplitMix64

N_RANDOM_HORIZON = 10      # Horizonte N = 1 bis 10 werden je System geprüft
CTRL_MIN = 1e-3            # Steuerbarkeitsgrenze |det [B, AB]|


def draw_system(rng: SplitMix64) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Ein Zufallssystem (A, B, q, r) in fester Ziehungsreihenfolge: vier Normalzahlen für A (zeilenweise), zwei für B, dann zwei Gleichverteilte für q und r."""
    A = np.array([[rng.normal(), rng.normal()], [rng.normal(), rng.normal()]])
    B = np.array([[rng.normal()], [rng.normal()]])
    q = 10.0 ** rng.uniform(-3.0, 0.0)
    r = 10.0 ** rng.uniform(-3.0, 1.0)
    return A, B, q, r


def is_controllable(A: np.ndarray, B: np.ndarray) -> bool:
    return abs(float(np.linalg.det(np.hstack([B, A @ B])))) >= CTRL_MIN


def evaluate_system(A, B, q, r, with_dare: bool = True) -> dict | None:
    """Ergebnis für ein Zufallssystem; None, wenn es nicht steuerbar oder die Riccati-Gleichung nicht lösbar ist.

    Schlüssel: open_stable, rho_N1 (ρ des Gesetzes N = 1 bei P_f = Q, nur offen stabil), any_unstable (irgendein N ≤ 10 instabil bei P_f = Q, nur offen stabil),
    dev (größte relative Abweichung |K_N − K_∞| bei P_f = P_∞ über N ≤ 10; nur mit `with_dare`)."""
    if not is_controllable(A, B):
        return None
    plant = D.Plant("Zufallssystem", A, B, q * np.eye(2), np.array([[r]]))
    out: dict = {"open_stable": D.spectral_radius(A) < 1.0}
    Pinf = Kinf = None
    if with_dare:
        try:
            Pinf, Kinf = D.dare(plant)
        except (np.linalg.LinAlgError, ValueError):
            return None
        if D.rho_of_gain(plant, Kinf) >= 1.0:
            return None
        gains, _ = D.horizon_gains(plant, Pinf, N_RANDOM_HORIZON)
        scale = max(float(np.max(np.abs(Kinf))), 1e-9)
        out["dev"] = max(float(np.max(np.abs(K - Kinf))) / scale for K in gains)
    if out["open_stable"]:
        rho = D.rho_curve(plant, plant.Q, N_RANDOM_HORIZON)
        out["rho_N1"] = float(rho[0])
        out["n1_unstable"] = bool(rho[0] >= 1.0)
        out["any_unstable"] = bool(np.any(rho >= 1.0))
    return out


def count_systems(seed: int, n_systems: int, with_dare: bool = True) -> dict:
    """Zählung über `n_systems` Ziehungen mit dem Zufalls-Seed `seed`: gültige Systeme, offen stabile, davon mit instabilem N = 1 bzw. irgendeinem instabilen N ≤ 10,
    größte und mittlere relative Abweichung von K_∞ bei P_f = P_∞ (mit `with_dare`)."""
    rng = SplitMix64(seed)
    valid = open_stable = n1 = anyn = 0
    devs: list[float] = []
    for _ in range(n_systems):
        A, B, q, r = draw_system(rng)
        res = evaluate_system(A, B, q, r, with_dare)
        if res is None:
            continue
        valid += 1
        if with_dare:
            devs.append(res["dev"])
        if res["open_stable"]:
            open_stable += 1
            n1 += int(res["n1_unstable"])
            anyn += int(res["any_unstable"])
    out = {"seed": seed, "drawn": n_systems, "valid": valid, "open_stable": open_stable, "n1_unstable": n1, "any_unstable": anyn}
    if with_dare and devs:
        out["dev_max"] = max(devs)
        out["dev_median"] = float(np.median(devs))
    return out
