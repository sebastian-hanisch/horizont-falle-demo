"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Presets: kein Preset legt einen Wert genau auf die Grenze ρ = 1, außer dem benannten Fall „Wagen, N = 1“ (dort ist ρ = 1 der Befund; die Tests prüfen ihn nur als Band
ρ ≥ 1 − 10⁻⁹, nie mit == in Gleitkomma). „Stabil“ heißt in dieser Demo ρ < 1, „grenzstabil“ heißt ρ = 1 (Band)."""

SYSTEMS = ("kalman", "wagen", "vier_tank")
SYSTEM_LABELS = {
    "kalman": "Kalmans Gegenbeispiel zum endlichen Horizont",
    "wagen": "Der Wagen (Doppelintegrator)",
    "vier_tank": "Vier-Tank-System (modellbasierte Entsprechung)",
}
SYSTEM_SHORT = {"kalman": "Kalmans Gegenbeispiel", "wagen": "Wagen", "vier_tank": "Vier-Tank-System"}

N_MIN, N_MAX, DEFAULT_N = 1, 150, 1          # Horizont N des Gesetzes; bis 150 wird gerechnet (die Vier-Tank-Schwelle liegt bei N = 55)
N_SHOW = {"kalman": 40, "wagen": 40, "vier_tank": 150}   # bis zu diesem N zeigt das Diagramm ρ_N
N_TABLE = 12                                 # Spalten der Terminalkosten-Tabelle: N = 1 bis 12
N_PRICE = 40                                 # Kostenverhältnis für N = 1 bis 40
TRAJ_STEPS = {"kalman": 40, "wagen": 40, "vier_tank": 150}
N_SEARCH = 400                               # Suchbereich des kleinsten stabilen N in den Karten von Experiment 4

C_OPTIONS = (3.0, 2.0, 1.2, 1.0, 0.9, 0.8, 2 / 3, 0.6, 0.5, 0.4, 0.3, 0.1, 0.0)   # Kalman-Familie: Ausgang C = (−c, 1), Nullstelle bei 1/c (c = 0: keine)
DEFAULT_C = 2 / 3
EPS_OPTIONS = (1e-3, 1e-2, 1e-1, 1.0)        # Q = C′C + ε·I
DEFAULT_EPS = 1e-3
R_OPTIONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0)   # Stellgewicht R (logarithmische Stufen)
DEFAULT_R = 1e-3
R_DECADES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)   # die Zeilen der Karten und der Vier-Tank-Tabelle
PF_KINDS = ("null", "q", "gamma_i", "gamma_pinf")
PF_LABELS = {"null": "P_f = 0", "q": "P_f = Q", "gamma_i": "P_f = γ·I", "gamma_pinf": "P_f = γ·P_∞"}
DEFAULT_PF = "q"
GAMMA_OPTIONS = (0.5, 0.9, 1.0, 1.1, 2.0, 10.0, 100.0, 1000.0)
DEFAULT_GAMMA = 1.0
WAGEN_R_OPTIONS = ("1", "1/10", "1/100", "1/1000")   # Experiment 5: R als Bruch (Brüche in der Poltabelle)
WAGEN_X0_POS = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
DEFAULT_WAGEN_X0 = (2.0, 1.0)
SEEDS = tuple(range(1, 21))                  # Zufalls-Seeds der vorgerechneten Zufallssysteme (Experiment 4)
DEFAULT_SEED = 12
N_RANDOM_SYSTEMS = 6000

COLOR_RHO = "#2F6B65"
COLOR_BAD = "#b23a2a"
COLOR_LINE = "#4c78a8"
COLOR_ORANGE = "#c9742a"
COLOR_GREY = "rgba(150,150,150,0.8)"

PRESETS = {
    "Von Hand (Kalman, N = 1)": {"system": "kalman", "c": 2 / 3, "eps": 1e-3, "R": 1e-3, "N": 1, "pf": "q", "gamma": 1.0},
    "Kleinstes stabiles N (N = 7)": {"system": "kalman", "c": 2 / 3, "eps": 1e-3, "R": 1e-3, "N": 7, "pf": "q", "gamma": 1.0},
    "Terminalkosten = Riccati-Lösung": {"system": "kalman", "c": 2 / 3, "eps": 1e-3, "R": 1e-3, "N": 1, "pf": "gamma_pinf", "gamma": 1.0},
    "Wagen (N = 1 grenzstabil)": {"system": "wagen", "c": 2 / 3, "eps": 1e-3, "R": 1.0, "N": 1, "pf": "q", "gamma": 1.0},
    "Keine Falle (Nullstelle innen)": {"system": "kalman", "c": 2.0, "eps": 1e-3, "R": 1e-3, "N": 1, "pf": "q", "gamma": 1.0},
    "Vier-Tank (R = 10⁻²)": {"system": "vier_tank", "c": 2 / 3, "eps": 1e-3, "R": 1e-2, "N": 10, "pf": "q", "gamma": 1.0},
}
PRESET_HELP = {
    "Von Hand (Kalman, N = 1)": "Kalmans Gegenbeispiel, Terminalgewicht P_f = Q, Horizont N = 1: das Handbeispiel. Der größte Pol liegt bei 1,495, obwohl die offene Strecke stabil ist.",
    "Kleinstes stabiles N (N = 7)": "Dieselbe Strecke mit N = 7, dem kleinsten Horizont mit stabilem Gesetz (ρ = 0,989): knapp stabil, im schlechtesten Fall mit dem 9,31-fachen der optimalen Kosten.",
    "Terminalkosten = Riccati-Lösung": "Terminalgewicht P_f = P_∞ (Lösung der algebraischen Riccati-Gleichung): jedes N gibt dasselbe Gesetz wie der unendliche Horizont, ρ = 0,664.",
    "Wagen (N = 1 grenzstabil)": "Der Wagen mit R = 1, P_f = Q und N = 1: die Eigenwerte sind 1 und 1/2; der Wagen bremst seine Geschwindigkeit, aber nicht die Position (ρ = 1, grenzstabil).",
    "Keine Falle (Nullstelle innen)": "Kalman-Familie mit c = 2 (Nullstelle bei 0,5, innerhalb des Einheitskreises): in diesem Beispiel ist kein Horizont instabil.",
    "Vier-Tank (R = 10⁻²)": "Vier-Tank-System (modellbasierte Entsprechung), R = 10⁻², P_f = Q, N = 10: instabil, obwohl die offene Strecke stabil ist; erst ab N = 20 ist das Gesetz stabil.",
}

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
