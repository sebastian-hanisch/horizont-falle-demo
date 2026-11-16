"""Kern der Demo (nur numpy und scipy): das Gesetz des Horizonts N für lineare Strecken x⁺ = A x + B u mit quadratischen Kosten.

Konvention (überall in dieser Demo): Gesetz u = −K x; Kosten über N Schritte  Σ_{k<N} (xₖ′Q xₖ + uₖ′R uₖ) + x_N′ P_f x_N  (ohne Faktor ½;
Gewinn und Kostenmatrix sind davon unabhängig). K_N ist der Gewinn der ersten Stufe des Horizonts N; „MPC ohne Beschränkungen“ wendet in jedem
Schritt u = −K_N x an (Plan über N Schritte, nur den ersten Stellwert benutzen, neu rechnen).

Rückwärtsrechnung (Riccati): Π₀ = P_f, K_j = (R + B′Π_{j−1}B)⁻¹ B′Π_{j−1}A, Π_j = Q + A′Π_{j−1}(A − B K_j). K_N gehört zum Horizont N.

Einheiten, jede einzeln testbar (DEMO-PLAYBOOK §1b): `riccati_step`, `horizon_gains`, `spectral_radius`, `dare`, `terminal_matrix`, `hard_terminal_gain`,
`cost_ratio`, `time_varying_product`, `trajectory_norms`, `first_stable`, `unstable_horizons`. Die Strecken stehen in `build_plant`.

Eigene Herleitung und Messung in diesem Modell (ohne Beschränkungen, ohne Störung); die Vier-Tank-Matrizen sind eine eigene modellbasierte Entsprechung
zum Beispiel von Berberich u. a. (arXiv 1906.04679, Abschnitt V), keine Kopie von deren Verfahren."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

TOL = 1e-9   # ρ ≥ 1 − TOL heißt „nicht asymptotisch stabil“ (instabil oder grenzstabil); stabil heißt ρ < 1 − TOL


@dataclass(frozen=True)
class Plant:
    """Strecke und Gewichte; `Q` ist die Zustandsgewichtung (bei Kalmans Beispiel C′C + ε·I), `R` die Stellgewichtung."""

    name: str
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]


# ------------------------------------------------------------------------------------------------------------------ Strecken

def kalman_matrices(c: float = 2 / 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalmans Gegenbeispiel zum endlichen Horizont (Rawlings/Mayne/Diehl 1.3.4, S. 21 bis 22, dort c = 2/3): A, B und der Ausgang C = (−c, 1).

    Die Übertragungsfunktion der Strecke ist (1 − c z)/(z² − 4z/3 + 2/3); ihre Nullstelle liegt bei 1/c (c = 0: keine endliche Nullstelle)."""
    A = np.array([[4.0 / 3.0, -2.0 / 3.0], [1.0, 0.0]])
    B = np.array([[1.0], [0.0]])
    C = np.array([[-c, 1.0]])
    return A, B, C


def wagen_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Der Wagen: Doppelintegrator mit Abtastzeit 1, Zustand (Position, Geschwindigkeit), Stellgröße = Beschleunigung (nach Zeitpunkt 0 konstant)."""
    return np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([[0.0], [1.0]])


def vier_tank_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vier-Tank-System, linearisiert und abgetastet: Matrizen A, B, C wie aus der Textextraktion von Berberich u. a. (2021), Abschnitt V, gelesen.

    Die Zeilenfolge von B ist nicht gegen das Seitenbild der Quelle geprüft; das System dient hier nur als eigene modellbasierte Entsprechung."""
    A = np.array([[0.921, 0.0, 0.041, 0.0], [0.0, 0.918, 0.0, 0.033], [0.0, 0.0, 0.924, 0.0], [0.0, 0.0, 0.0, 0.937]])
    B = np.array([[0.017, 0.001], [0.001, 0.023], [0.0, 0.061], [0.072, 0.0]])
    C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    return A, B, C


def build_plant(kind: str, R: float, c: float = 2 / 3, eps: float = 1e-3) -> Plant:
    """Strecke nach Art: „kalman“ (Q = C′C + ε·I, R skalar), „wagen“ (Q = I, R skalar), „vier_tank“ (Q = 3 C′C, R·I mit zwei Eingängen)."""
    if kind == "kalman":
        A, B, C = kalman_matrices(c)
        return Plant("Kalmans Gegenbeispiel", A, B, C.T @ C + eps * np.eye(2), np.array([[float(R)]]))
    if kind == "wagen":
        A, B = wagen_matrices()
        return Plant("Wagen", A, B, np.eye(2), np.array([[float(R)]]))
    if kind == "vier_tank":
        A, B, C = vier_tank_matrices()
        return Plant("Vier-Tank-System", A, B, 3.0 * C.T @ C, float(R) * np.eye(2))
    raise ValueError(f"unbekannte Strecke: {kind!r}")


# ------------------------------------------------------------------------------------------------------------------ Riccati-Rückwärtsrechnung

def riccati_step(A, B, Q, R, P):
    """Ein Schritt rückwärts: aus der Kostenmatrix P der Folgestufe den Gewinn K und die Kostenmatrix der Stufe, K = (R + B′PB)⁻¹ B′PA, P⁺ = Q + A′P(A − BK)."""
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, Q + A.T @ P @ (A - B @ K)


def check_controllable(A: np.ndarray, B: np.ndarray) -> None:
    """Wirft ValueError („nicht steuerbar“), wenn das Paar (A, B) nicht steuerbar ist (Rang der Steuerbarkeitsmatrix [B, AB, …] kleiner als n); ohne Steuerbarkeit gibt es kein Gesetz zu zeigen."""
    n = A.shape[0]
    blocks, M = [], B
    for _ in range(n):
        blocks.append(M)
        M = A @ M
    if np.linalg.matrix_rank(np.hstack(blocks), tol=1e-9) < n:
        raise ValueError("Das System (A, B) ist nicht steuerbar: für diese Strecke gibt es kein Gesetz des Horizonts N.")


def check_symmetric(P: np.ndarray) -> np.ndarray:
    """Terminalgewichte müssen symmetrisch sein; kleine Asymmetrie (Rundung, ≤ 1e-9 relativ) wird symmetrisiert, größere abgelehnt (ValueError)."""
    P = np.array(P, dtype=float)
    if not np.allclose(P, P.T, rtol=1e-9, atol=1e-12):
        raise ValueError("Das Terminalgewicht P_f ist nicht symmetrisch.")
    return (P + P.T) / 2.0


def horizon_gains(plant: Plant, Pf: np.ndarray, n_max: int):
    """K_1 bis K_{n_max} (Liste) und die Kostenmatrizen Π_1 bis Π_{n_max} zum Terminalgewicht P_f: K_N gehört zum Horizont N."""
    P = check_symmetric(Pf)
    gains, costs = [], []
    for _ in range(n_max):
        K, P = riccati_step(plant.A, plant.B, plant.Q, plant.R, P)
        gains.append(K)
        costs.append(P)
    return gains, costs


def spectral_radius(M: np.ndarray) -> float:
    """Betrag des größten Eigenwerts (der „größte Pol“ des geschlossenen Kreises, wenn M = A − BK)."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def rho_of_gain(plant: Plant, K: np.ndarray) -> float:
    return spectral_radius(plant.A - plant.B @ K)


def rho_curve(plant: Plant, Pf: np.ndarray, n_max: int) -> np.ndarray:
    """ρ_N = Spektralradius von A − B K_N für N = 1 bis n_max (Index 0 gehört zu N = 1)."""
    gains, _ = horizon_gains(plant, Pf, n_max)
    return np.array([rho_of_gain(plant, K) for K in gains])


def dare(plant: Plant) -> tuple[np.ndarray, np.ndarray]:
    """Lösung P_∞ der algebraischen Riccati-Gleichung (SciPy) und Gewinn K_∞ des unendlichen Horizonts; ValueError, wenn (A, B) nicht steuerbar ist."""
    check_controllable(plant.A, plant.B)
    P = la.solve_discrete_are(plant.A, plant.B, plant.Q, plant.R)
    K = np.linalg.solve(plant.R + plant.B.T @ P @ plant.B, plant.B.T @ P @ plant.A)
    return P, K


def rho_open(plant: Plant) -> float:
    """Spektralradius der offenen Strecke."""
    return spectral_radius(plant.A)


def terminal_matrix(kind: str, plant: Plant, gamma: float = 1.0, Pinf: np.ndarray | None = None) -> np.ndarray:
    """Terminalgewicht: „null“ (P_f = 0), „q“ (P_f = Q), „gamma_i“ (γ·I), „gamma_pinf“ (γ·P_∞, braucht die Riccati-Lösung)."""
    if kind == "null":
        return np.zeros((plant.n, plant.n))
    if kind == "q":
        return plant.Q.copy()
    if kind == "gamma_i":
        return gamma * np.eye(plant.n)
    if kind == "gamma_pinf":
        if Pinf is None:
            Pinf = dare(plant)[0]
        return gamma * Pinf
    raise ValueError(f"unbekanntes Terminalgewicht: {kind!r}")


def unstable_horizons(rho: np.ndarray, tol: float = TOL) -> list[int]:
    """Horizonte N (ab 1), bei denen das Gesetz nicht asymptotisch stabil ist (ρ ≥ 1 − tol)."""
    return [i + 1 for i, v in enumerate(rho) if v >= 1.0 - tol]


def first_stable(rho: np.ndarray, tol: float = TOL) -> int | None:
    """Kleinstes N mit ρ_N < 1 − tol (None, wenn es im gerechneten Bereich keines gibt). Das heißt nicht, dass danach alle N stabil bleiben (siehe `later_unstable`)."""
    for i, v in enumerate(rho):
        if v < 1.0 - tol:
            return i + 1
    return None


def later_unstable(rho: np.ndarray, tol: float = TOL) -> int:
    """Zahl der instabilen Horizonte hinter dem kleinsten stabilen N (0 heißt: ab dem kleinsten stabilen N nie wieder instabil, im gerechneten Bereich)."""
    f = first_stable(rho, tol)
    if f is None:
        return 0
    return int(np.sum(rho[f - 1:] >= 1.0 - tol))


# ------------------------------------------------------------------------------------------------------------------ Kosten und Verläufe

def lyapunov_cost_matrix(plant: Plant, K: np.ndarray) -> np.ndarray | None:
    """Kosten der unendlichen Zukunft unter u = −Kx als x′Sx, S aus (A−BK)′S(A−BK) − S = −(Q + K′RK); None, wenn das Gesetz nicht stabil ist."""
    Ac = plant.A - plant.B @ K
    if spectral_radius(Ac) >= 1.0 - TOL:
        return None
    return la.solve_discrete_lyapunov(Ac.T, plant.Q + K.T @ plant.R @ K)


def cost_ratio(plant: Plant, K: np.ndarray, Pinf: np.ndarray) -> float:
    """Kosten der unendlichen Zukunft unter dem Gesetz K im schlechtesten Anfangszustand, geteilt durch die optimalen Kosten (≥ 1); inf, wenn das Gesetz nicht stabil ist.

    Schlechtester Anfangszustand: max x′Sx / x′P_∞x, der größte verallgemeinerte Eigenwert von (S, P_∞)."""
    S = lyapunov_cost_matrix(plant, K)
    if S is None:
        return float("inf")
    return float(np.max(np.real(np.linalg.eigvals(np.linalg.solve(Pinf, S)))))


def trajectory_norms(plant: Plant, K: np.ndarray, x0: np.ndarray, steps: int) -> np.ndarray:
    """|x_k| (Euklid-Norm) für k = 0 bis steps unter dem festen Gesetz u = −K x ohne Störung."""
    Ac = plant.A - plant.B @ K
    x = np.array(x0, dtype=float)
    out = [float(np.linalg.norm(x))]
    for _ in range(steps):
        x = Ac @ x
        out.append(float(np.linalg.norm(x)))
    return np.array(out)


def trajectory_states(plant: Plant, K: np.ndarray, x0: np.ndarray, steps: int) -> np.ndarray:
    """Zustände x_0 bis x_steps (Zeilen) unter u = −K x."""
    Ac = plant.A - plant.B @ K
    x = np.array(x0, dtype=float)
    out = [x.copy()]
    for _ in range(steps):
        x = Ac @ x
        out.append(x.copy())
    return np.array(out)


def time_varying_product(plant: Plant, gains: list[np.ndarray], N: int) -> np.ndarray:
    """Produkt (A − B K_1) ⋯ (A − B K_{N−1})(A − B K_N): der Plan über N Schritte, der K_N, K_{N−1}, …, K_1 nacheinander anwendet (so ist der Plan gedacht)."""
    M = np.eye(plant.n)
    for j in range(N, 0, -1):
        M = (plant.A - plant.B @ gains[j - 1]) @ M
    return M


def time_varying_summary(plant: Plant, gains: list[np.ndarray], N: int) -> tuple[float, float]:
    """(geometrisches Mittel des Spektralradius pro Schritt, größter Singulärwert) des N-Schritt-Produkts."""
    M = time_varying_product(plant, gains, N)
    return spectral_radius(M) ** (1.0 / N), float(np.linalg.norm(M, 2))


# ------------------------------------------------------------------------------------------------------------------ Harte Terminalbedingung

def hard_terminal_gain(plant: Plant, N: int) -> np.ndarray | None:
    """Gewinn des Horizonts N mit der harten Bedingung x_N = 0 (Grenzfall P_f → ∞, ohne Beschränkungen), aus dem KKT-System aller Stellwerte u_0 bis u_{N−1}.

    None, wenn die Bedingung in N Schritten nicht für jeden Anfangszustand erfüllbar ist (singuläres System, z. B. N = 1 bei zwei Zuständen und einem Eingang)."""
    A, B, Q, R = plant.A, plant.B, plant.Q, plant.R
    n, m = plant.n, plant.m
    G = np.zeros((n * N, m * N))
    F = np.zeros((n * N, n))
    for k in range(1, N + 1):
        F[(k - 1) * n: k * n] = np.linalg.matrix_power(A, k)
        for j in range(k):
            G[(k - 1) * n: k * n, j * m: (j + 1) * m] = np.linalg.matrix_power(A, k - 1 - j) @ B
    Qb = np.zeros((n * N, n * N))
    for k in range(1, N):
        Qb[(k - 1) * n: k * n, (k - 1) * n: k * n] = Q
    H = G.T @ Qb @ G + np.kron(np.eye(N), R)
    Gt, Ft = G[(N - 1) * n: N * n], F[(N - 1) * n: N * n]
    KKT = np.block([[H, Gt.T], [Gt, np.zeros((n, n))]])
    if np.linalg.matrix_rank(Gt, tol=1e-9) < n:
        return None
    sol = np.linalg.solve(KKT, np.vstack([-G.T @ Qb @ F, -Ft]))
    return -sol[:m]
