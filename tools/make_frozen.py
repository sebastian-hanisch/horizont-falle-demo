"""Erzeugt `data/hfa_frozen.json`: die Messreihen, die die App nicht live rechnet (die Normalzahlen über `decimal` kosten je Zufalls-Seed einige Sekunden).

Aufruf (Wurzelordner der Demo): _venvs/test/Scripts/python.exe -X utf8 tools/make_frozen.py        (rund 3 Minuten)
Inhalt (Modell: lineares System ohne Beschränkungen, ohne Störung; Zähleinheit Horizont N und System):
  random      je Zufalls-Seed 1 bis 20: 6 000 Zufallssysteme (n = 2, ein Eingang, Q = q·I, R = r), gültige (steuerbar, Riccati lösbar), offen stabile, davon mit instabilem N = 1 bzw. irgendeinem
              instabilen N ≤ 10 bei P_f = Q, größte und mittlere relative Abweichung |K_N − K_∞| bei P_f = P_∞ (N ≤ 10); dazu `prefix500`, dieselbe Zählung für die ersten 500 Ziehungen
              (der Test rechnet nur diese neu)
  random_c    200 Zufallswerte c in (0,05; 4) (Zufalls-Seed 31): Falle (instabiles N ≤ 400 bei ε = R = 10⁻³, P_f = Q) gegen „Nullstelle 1/c außerhalb des Einheitskreises“; Ausnahmen mit Werten
Die Zufallszahlen kommen aus dem SplitMix64 der Demo (hfa_rng.py), nie aus Python-`hash()` oder `random`."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hfa_constants as C  # noqa: E402
import hfa_evaluation as ev  # noqa: E402
import hfa_random as Rz  # noqa: E402
from hfa_rng import SplitMix64  # noqa: E402

SEED_C = 31
N_C = 200
PREFIX = 500


def random_rows() -> list[dict]:
    rows = []
    for seed in C.SEEDS:
        full = Rz.count_systems(seed, C.N_RANDOM_SYSTEMS)
        full["prefix500"] = Rz.count_systems(seed, PREFIX)
        rows.append(full)
    return rows


def random_c() -> dict:
    rng = SplitMix64(SEED_C)
    cs = [rng.uniform(0.05, 4.0) for _ in range(N_C)]
    outside = trap_outside = rest = trap_rest = 0
    exceptions = []
    for c in cs:
        cell = ev.trap_cell(c, ev.NULL_PLANT_EPS, ev.NULL_PLANT_R)
        is_out = 0.0 < c < 1.0
        trap = cell["n_unstable"] > 0
        outside += is_out
        rest += not is_out
        trap_outside += is_out and trap
        trap_rest += (not is_out) and trap
        if is_out != trap:
            exceptions.append({"c": c, "zero": 1.0 / c, "n_unstable": cell["n_unstable"], "rho1": cell["rho"][0]})
    return {"seed": SEED_C, "n": N_C, "outside": outside, "outside_trap": trap_outside, "rest": rest, "rest_trap": trap_rest, "exceptions": exceptions}


def main() -> None:
    out = {"random": random_rows(), "random_c": random_c()}
    path = ROOT / "data" / "hfa_frozen.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben:", path)
    for r in out["random"]:
        print(r["seed"], r["valid"], r["open_stable"], r["n1_unstable"], r["any_unstable"], f"{r['dev_max']:.2e}")
    print(out["random_c"])


if __name__ == "__main__":
    main()
