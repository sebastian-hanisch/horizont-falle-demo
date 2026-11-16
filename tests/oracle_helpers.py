"""Unabhängige Rechenwege für die Orakel-Tests: nichts davon benutzt `hfa_dynamics.riccati_step`, `horizon_gains` oder `hfa_exact`.

- `batch_gain_float`: alle Stellwerte gemeinsam (Normalgleichungen H u = −G′Q̄F x₀), erste Zeile.
- `batch_gain_fractions`: dasselbe in Brüchen mit eigenem Gauß-Löser.
- `plan_cost` und `plan_gradient`: Kosten eines Stellplans durch Vorwärtssimulation, Gradient über die Adjungierte (Kozustandsrekursion); damit minimiert `brute_force_u0` per BFGS.
- `jury_rho`: Spektralradius einer 2×2-Matrix aus Spur und Determinante (Jury-Dreieck: beide Wurzeln im Einheitskreis ⇔ |det| < 1 und |spur| < 1 + det), hier als Betrag der größten Wurzel.
- `dare_fixpoint`: Fixpunkt der Riccati-Rekursion bis zur Konvergenz."""

from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize


def _stack(A, B, N):
    n, m = B.shape
    G = np.zeros((n * N, m * N))
    F = np.zeros((n * N, n))
    Ak = np.eye(n)
    powers = [Ak]
    for _ in range(N):
        powers.append(powers[-1] @ A)
    for k in range(1, N + 1):
        F[(k - 1) * n: k * n] = powers[k]
        for j in range(k):
            G[(k - 1) * n: k * n, j * m: (j + 1) * m] = powers[k - 1 - j] @ B
    return G, F


def batch_gain_float(A, B, Q, R, Pf, N):
    n, m = B.shape
    G, F = _stack(A, B, N)
    Qb = np.zeros((n * N, n * N))
    for k in range(1, N):
        Qb[(k - 1) * n: k * n, (k - 1) * n: k * n] = Q
    Qb[(N - 1) * n: N * n, (N - 1) * n: N * n] = Pf
    H = G.T @ Qb @ G + np.kron(np.eye(N), R)
    return np.linalg.solve(H, G.T @ Qb @ F)[:m]


def _solve_fractions(M, b):
    n = len(M)
    aug = [list(M[i]) + list(b[i]) for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c]
        aug[c] = [v / piv for v in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [aug[r][j] - f * aug[c][j] for j in range(len(aug[r]))]
    return [row[n:] for row in aug]


def batch_gain_fractions(A, B, Q, R, Pf, N):
    """Wie batch_gain_float, aber mit Brüchen; A, B, Q, R, Pf sind Listen von Listen (Fraction oder int)."""
    n, m = len(A), len(B[0])

    def mm(X, Y):
        return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))] for i in range(len(X))]

    powers = [[[Fr(1 if i == j else 0) for j in range(n)] for i in range(n)]]
    for _ in range(N):
        powers.append(mm(A, powers[-1]))
    G = [[Fr(0)] * (m * N) for _ in range(n * N)]
    F = [[Fr(0)] * n for _ in range(n * N)]
    for k in range(1, N + 1):
        for i in range(n):
            for j in range(n):
                F[(k - 1) * n + i][j] = powers[k][i][j]
        for j in range(k):
            blk = mm(powers[k - 1 - j], B)
            for i in range(n):
                for c in range(m):
                    G[(k - 1) * n + i][j * m + c] = blk[i][c]
    Qb = [[Fr(0)] * (n * N) for _ in range(n * N)]
    for k in range(1, N + 1):
        W = Pf if k == N else Q
        for i in range(n):
            for j in range(n):
                Qb[(k - 1) * n + i][(k - 1) * n + j] = Fr(W[i][j])
    Rb = [[Fr(0)] * (m * N) for _ in range(m * N)]
    for k in range(N):
        for i in range(m):
            for j in range(m):
                Rb[k * m + i][k * m + j] = Fr(R[i][j])
    Gt = [list(r) for r in zip(*G)]
    H = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mm(Gt, mm(Qb, G)), Rb)]
    sol = _solve_fractions(H, mm(Gt, mm(Qb, F)))
    return sol[:m]


def plan_cost(A, B, Q, R, Pf, x0, u):
    """Kosten Σ_(k<N) (x′Qx + u′Ru) + x_N′P_f x_N eines Stellplans u (Form N × m) durch Vorwärtssimulation."""
    x = np.array(x0, dtype=float)
    cost = 0.0
    for uk in u:
        cost += x @ Q @ x + uk @ R @ uk
        x = A @ x + B @ uk
    return float(cost + x @ Pf @ x)


def plan_gradient(A, B, Q, R, Pf, x0, u):
    """Gradient der Kosten nach u über die Adjungierte: λ_N = 2P_f x_N, λ_k = 2Q x_k + A′λ_(k+1), ∂J/∂u_k = 2R u_k + B′λ_(k+1)."""
    xs = [np.array(x0, dtype=float)]
    for uk in u:
        xs.append(A @ xs[-1] + B @ uk)
    lam = 2.0 * Pf @ xs[-1]
    grad = np.zeros_like(u)
    for k in range(len(u) - 1, -1, -1):
        grad[k] = 2.0 * R @ u[k] + B.T @ lam
        lam = 2.0 * Q @ xs[k] + A.T @ lam
    return grad


def brute_force_u0(A, B, Q, R, Pf, x0, N):
    """Erster Stellwert des besten Plans über N Schritte, numerisch (BFGS mit analytischem Gradienten) minimiert."""
    m = B.shape[1]
    f = lambda v: plan_cost(A, B, Q, R, Pf, x0, v.reshape(N, m))  # noqa: E731
    g = lambda v: plan_gradient(A, B, Q, R, Pf, x0, v.reshape(N, m)).ravel()  # noqa: E731
    res = minimize(f, np.zeros(N * m), jac=g, method="BFGS", options={"gtol": 1e-12, "maxiter": 2000})
    return res.x.reshape(N, m)[0]


def jury_rho(spur, det):
    """Betrag der größten Wurzel von z² − spur·z + det aus der Lösungsformel (zwei reelle Wurzeln) bzw. √det (konjugiert komplexes Paar)."""
    disc = spur * spur - 4.0 * det
    if disc >= 0:
        r = disc ** 0.5
        return max(abs((spur + r) / 2.0), abs((spur - r) / 2.0))
    return abs(det) ** 0.5


def jury_stable(spur, det):
    """Jury-Kriterium für n = 2: beide Wurzeln im offenen Einheitskreis ⇔ |det| < 1 und |spur| < 1 + det."""
    return abs(det) < 1.0 and abs(spur) < 1.0 + det


def dare_fixpoint(A, B, Q, R, iterations=20000):
    P = np.array(Q, dtype=float)
    for _ in range(iterations):
        S = R + B.T @ P @ B
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A)
        if np.max(np.abs(Pn - P)) < 1e-15 * max(1.0, np.max(np.abs(Pn))):
            return Pn
        P = Pn
    return P
