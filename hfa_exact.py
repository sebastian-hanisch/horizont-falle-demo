"""Exakte Bruchrechnung (`fractions`) für die Handbeispiele und die Poltabelle am Wagen: dieselbe Riccati-Rückwärtsrechnung wie in `hfa_dynamics.py`, aber ohne Rundung.

Matrizen sind Listen von Listen mit `Fraction`. Alles, was hier „exakt“ heißt, ist in Brüchen gerechnet und damit auf jeder Plattform gleich (anders als Eigenwerte
und Lösungen aus SciPy, die an der letzten Stelle abweichen können). Die Aufgabe ist klein (n ≤ 2): Gauß-Jordan reicht."""

from __future__ import annotations

import math
from fractions import Fraction as Fr


def frac_matrix(rows) -> list[list[Fr]]:
    return [[Fr(x) for x in r] for r in rows]


def identity(n: int) -> list[list[Fr]]:
    return [[Fr(1 if i == j else 0) for j in range(n)] for i in range(n)]


def transpose(X):
    return [list(r) for r in zip(*X)]


def mat_mul(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))] for i in range(len(X))]


def mat_add(X, Y, sign=1):
    return [[X[i][j] + sign * Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def mat_inv(X):
    """Gauß-Jordan mit Brüchen; wirft ZeroDivisionError, wenn X singulär ist."""
    n = len(X)
    aug = [list(X[i]) + identity(n)[i] for i in range(n)]
    for c in range(n):
        p = next((r for r in range(c, n) if aug[r][c] != 0), None)
        if p is None:
            raise ZeroDivisionError("singuläre Matrix")
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c]
        aug[c] = [v / piv for v in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [aug[r][j] - f * aug[c][j] for j in range(2 * n)]
    return [row[n:] for row in aug]


def riccati_step_exact(A, B, Q, R, P):
    """Ein Riccati-Schritt rückwärts in Brüchen: (K, P⁺) mit K = (R + B′PB)⁻¹B′PA und P⁺ = Q + A′P(A − BK)."""
    S = mat_add(R, mat_mul(transpose(B), mat_mul(P, B)))
    K = mat_mul(mat_inv(S), mat_mul(transpose(B), mat_mul(P, A)))
    Pn = mat_add(Q, mat_mul(transpose(A), mat_mul(P, mat_add(A, mat_mul(B, K), -1))))
    return K, Pn


def horizon_gains_exact(A, B, Q, R, Pf, n_max: int):
    """[(K_1, Π_1), …, (K_n_max, Π_n_max)] in Brüchen; K_j gehört zum Horizont j (Kostenmatrix nach j − 1 Schritten), Π_j nach j Schritten."""
    out, P = [], Pf
    for _ in range(n_max):
        K, P = riccati_step_exact(A, B, Q, R, P)
        out.append((K, P))
    return out


def closed_loop_exact(A, B, K):
    return mat_add(A, mat_mul(B, K), -1)


def trace_det(M) -> tuple[Fr, Fr]:
    """Spur und Determinante einer 2×2-Matrix."""
    return M[0][0] + M[1][1], M[0][0] * M[1][1] - M[0][1] * M[1][0]


def rho_from_trace_det(spur: Fr, det: Fr) -> float:
    """Spektralradius einer 2×2-Matrix aus Spur und Determinante: reelle Eigenwerte (s ± √(s² − 4d))/2, sonst √d (Betrag eines konjugiert komplexen Paars)."""
    disc = float(spur * spur - 4 * det)
    if disc >= 0.0:
        w = math.sqrt(disc)
        return max(abs(float(spur) + w), abs(float(spur) - w)) / 2.0
    return math.sqrt(float(det))


def fraction_text(x: Fr) -> str:
    """Bruch als Text: „2/7“, ganze Zahlen ohne Nenner."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
