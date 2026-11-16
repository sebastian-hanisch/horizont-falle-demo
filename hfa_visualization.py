"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import hfa_constants as C
import hfa_dynamics as D


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_rho_curve(rho: np.ndarray, N_sel: int, rho_open: float, rho_inf: float, n_show: int):
    """ρ_N als Stäbe für N = 1 bis n_show: rot ab ρ ≥ 1 (nicht asymptotisch stabil), grün darunter; Linie bei 1, gestrichelt ρ der offenen Strecke und ρ_∞; der gewählte Horizont hat einen Rahmen."""
    n_show = min(n_show, len(rho))
    xs = list(range(1, n_show + 1))
    ys = [float(v) for v in rho[:n_show]]
    colors = [C.COLOR_BAD if v >= 1.0 - D.TOL else C.COLOR_RHO for v in ys]
    widths = [3 if x == N_sel else 0 for x in xs]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker=dict(color=colors, line=dict(color="black", width=widths)), name="ρ des Gesetzes K_N", hovertemplate="N = %{x}: ρ = %{y:.4f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_BAD, width=2), annotation_text="ρ = 1 (Grenze)", annotation_position="top left")
    fig.add_hline(y=rho_inf, line=dict(color=C.COLOR_LINE, dash="dash", width=1.5), annotation_text=f"N → ∞: {rho_inf:.3f}".replace(".", ","), annotation_position="bottom right")
    if abs(rho_open - rho_inf) > 0.01 and abs(rho_open - 1.0) > 0.01:
        fig.add_hline(y=rho_open, line=dict(color=C.COLOR_GREY, dash="dot", width=1.5), annotation_text=f"offene Strecke: {rho_open:.3f}".replace(".", ","), annotation_position="top right")
    fig.update_xaxes(title_text="Horizont N")
    fig.update_yaxes(title_text="Spektralradius ρ (Betrag des größten Pols)", range=[0, max(1.15, max(ys) * 1.08)])
    fig.update_layout(showlegend=False)
    return _base(fig)


def build_poles(poles_sel: np.ndarray, poles_inf: np.ndarray, poles_open: np.ndarray, N: int):
    """Pole in der komplexen Ebene mit Einheitskreis: offene Strecke (grau), Gesetz des Horizonts N (rot außerhalb, grün innerhalb), unendlicher Horizont (blau). Gleicher Maßstab über scaleanchor."""
    th = np.linspace(0, 2 * np.pi, 181)
    fig = go.Figure(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color="rgba(120,120,120,0.9)", width=1.5), hoverinfo="skip", name="Einheitskreis"))
    sel_color = C.COLOR_BAD if float(np.max(np.abs(poles_sel))) >= 1.0 - D.TOL else C.COLOR_RHO
    for pts, name, color, symbol, size in ((poles_open, "offene Strecke", C.COLOR_GREY, "circle-open", 12), (poles_inf, "N → ∞", C.COLOR_LINE, "diamond", 10), (poles_sel, f"Gesetz N = {N}", sel_color, "x", 12)):
        fig.add_trace(go.Scatter(x=[float(p.real) for p in pts], y=[float(p.imag) for p in pts], mode="markers", name=name, marker=dict(color=color, symbol=symbol, size=size, line=dict(width=2)),
                                 hovertemplate=name + ": %{x:.4f} %{y:+.4f}i<extra></extra>"))
    allp = np.concatenate([poles_sel, poles_inf, poles_open])
    lim = max(1.25, float(np.max(np.abs(allp))) * 1.12)
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))   # erzwingt den Rand
    fig.update_xaxes(title_text="Realteil", zeroline=True)
    fig.update_yaxes(title_text="Imaginärteil", scaleanchor="x", zeroline=True)
    return _base(fig, height=420)


def build_trajectory(norm_sel: np.ndarray, norm_inf: np.ndarray, N: int):
    """|x_k| des Regelkreises (logarithmische Achse) unter dem Gesetz K_N und unter dem unendlichen Gesetz, Anfangszustand e₁."""
    ks = list(range(len(norm_sel)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=np.maximum(norm_sel, 1e-300), mode="lines+markers", name=f"Gesetz N = {N}", line=dict(color=C.COLOR_ORANGE, width=3), marker=dict(size=4),
                             hovertemplate="k = %{x}: |x| = %{y:.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ks, y=np.maximum(norm_inf, 1e-300), mode="lines", name="Gesetz N → ∞", line=dict(color=C.COLOR_LINE, width=2, dash="dash"), hovertemplate="k = %{x}: |x| = %{y:.4g}<extra></extra>"))
    fig.update_xaxes(title_text="Schritt k")
    fig.update_yaxes(title_text="|x_k| (Euklid-Norm, logarithmisch)", type="log", exponentformat="power")
    return _base(fig)


def _runs(ns: list[int]) -> list[tuple[int, int]]:
    out = []
    for n in ns:
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def build_price(ratios: np.ndarray, N_sel: int):
    """Kostenverhältnis (schlechtester Anfangszustand) über N, logarithmische Achse; nicht stabile Horizonte sind rot hinterlegt, der gewählte Horizont ist markiert."""
    xs = list(range(1, len(ratios) + 1))
    fin = [(x, float(r)) for x, r in zip(xs, ratios) if np.isfinite(r)]
    fig = go.Figure()
    if fin:
        fig.add_trace(go.Scatter(x=[x for x, _ in fin], y=[r for _, r in fin], mode="lines+markers", line=dict(color=C.COLOR_RHO, width=3), marker=dict(size=6), name="Kostenverhältnis",
                                 hovertemplate="N = %{x}: %{y:.4f}<extra></extra>"))
    for a, b in _runs([x for x, r in zip(xs, ratios) if not np.isfinite(r)]):
        fig.add_vrect(x0=a - 0.5, x1=b + 0.5, fillcolor="rgba(178,58,42,0.18)", line_width=0)
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dot", width=1))
    if N_sel <= len(ratios) and np.isfinite(ratios[N_sel - 1]):
        fig.add_trace(go.Scatter(x=[N_sel], y=[float(ratios[N_sel - 1])], mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color="black", width=3)), name=f"gewähltes N = {N_sel}",
                                 hoverinfo="skip"))
    top = max([r for _, r in fin], default=10.0)
    fig.update_xaxes(title_text="Horizont N (rot: nicht stabil, Kosten nicht endlich)")
    fig.update_yaxes(title_text="Kosten ÷ optimale Kosten (logarithmisch)", type="log", range=[-0.02, np.log10(max(top, 1.2)) + 0.1], exponentformat="power")
    return _base(fig)


def build_map(first: list[list[int | None]], row_labels: list[str], col_labels: list[str], x_title: str, y_title: str, mark: tuple[str, str] | None = None):
    """Karte „kleinstes stabiles N“: Zahl in jeder Zelle, grün bei 1 (keine Falle), je größer die Zahl desto roter; `mark` = (Zeilenname, Spaltenname) der gewählten Zelle."""
    z = [[np.nan if v is None else float(v) for v in row] for row in first]
    text = [["–" if v is None else str(v) for v in row] for row in first]
    fig = go.Figure(go.Heatmap(z=z, x=col_labels, y=row_labels, text=text, texttemplate="%{text}", zmin=1, zmax=14,
                               colorscale=[[0.0, "#7fb685"], [0.08, "#f2e3a0"], [1.0, "#b23a2a"]], showscale=False, hovertemplate=y_title + " %{y}, " + x_title + " %{x}: kleinstes stabiles N = %{text}<extra></extra>"))
    if mark is not None and mark[0] in row_labels and mark[1] in col_labels:
        fig.add_trace(go.Scatter(x=[mark[1]], y=[mark[0]], mode="markers", marker=dict(size=34, color="rgba(0,0,0,0)", line=dict(color="black", width=3), symbol="square"), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title_text=x_title, type="category", side="bottom")
    fig.update_yaxes(title_text=y_title, type="category", autorange="reversed")
    return _base(fig, height=430 if len(row_labels) > 6 else 300)


def build_random(rows: list[dict], seed_sel: int):
    """Anteil der offen stabilen Zufallssysteme, bei denen das Gesetz N = 1 (P_f = Q) instabil ist, je Zufalls-Seed (Prozent); der gewählte Zufalls-Seed ist dunkel."""
    seeds = [r["seed"] for r in rows]
    vals = [100.0 * r["n1_unstable"] / r["open_stable"] for r in rows]
    colors = [C.COLOR_BAD if s == seed_sel else "rgba(178,58,42,0.35)" for s in seeds]
    fig = go.Figure(go.Bar(x=[str(s) for s in seeds], y=vals, marker_color=colors, hovertemplate="Zufalls-Seed %{x}: %{y:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Zufalls-Seed (je 6 000 Zufallssysteme)", type="category")
    fig.update_yaxes(title_text="N = 1 instabil, Anteil der offen stabilen Strecken (%)", rangemode="tozero")
    return _base(fig, height=300)


def build_phase(paths: dict, x0: tuple[float, float], endpoint: float):
    """Phasenbild des Wagens: Position (x) gegen Geschwindigkeit (y) unter den Gesetzen N = 1, 2, 3 und ∞; Stern: Anfangszustand, Raute: Endlage des Gesetzes N = 1 auf der Positionsachse."""
    palette = {"N = 1": C.COLOR_BAD, "N = 2": C.COLOR_ORANGE, "N = 3": C.COLOR_RHO, "N → ∞": C.COLOR_LINE}
    fig = go.Figure()
    for name, st in paths.items():
        fig.add_trace(go.Scatter(x=st[:, 0], y=st[:, 1], mode="lines+markers", name=name, line=dict(color=palette[name], width=2), marker=dict(size=4), hovertemplate=name + ": x₁ = %{x:.3f}, x₂ = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x0[0]], y=[x0[1]], mode="markers", name="Anfangszustand", marker=dict(symbol="star", size=14, color="black"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[endpoint], y=[0.0], mode="markers", name="Endlage bei N = 1", marker=dict(symbol="diamond", size=12, color=C.COLOR_BAD, line=dict(color="black", width=1)), hoverinfo="skip"))
    fig.update_xaxes(title_text="Position x₁")
    fig.update_yaxes(title_text="Geschwindigkeit x₂")
    return _base(fig, height=400)
