"""Die Falle des endlichen Horizonts – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der Linie „Regelung (von LQR bis MPC)“: ein Plan über N Schritte, in jedem Schritt neu gerechnet und nur im ersten Stellwert angewendet (MPC ohne Beschränkungen),
ergibt ein festes Gesetz u = −K_N x. Wann ist es stabil, wann nicht, und was ändern die Terminalkosten? Alles eigene Nachrechnung in diesem Modell (linear, ohne Beschränkungen,
ohne Störung); Kern in hfa_dynamics.py, Bruchrechnung in hfa_exact.py.

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import numpy as np
import streamlit as st

import hfa_constants as C
import hfa_dynamics as D
import hfa_evaluation as ev
from hfa_evaluation import Settings, de, fmt_int, pct, r_label
from hfa_layout import dunkles_design, kopf_css
from hfa_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)
from hfa_visualization import (
    build_map,
    build_phase,
    build_poles,
    build_price,
    build_random,
    build_rho_curve,
    build_trajectory,
)

st.set_page_config(page_title="Die Falle des endlichen Horizonts – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _terminal(settings):
    return ev.terminal_table(settings)


@st.cache_data(show_spinner=False)
def _price(settings):
    return ev.price_curve(settings)


@st.cache_data(show_spinner=False)
def _timevarying(settings):
    return ev.timevarying_rows(settings)


@st.cache_data(show_spinner="Rechne 13 Nullstellen bis N = 400 …")
def _null_rows():
    return ev.nullstelle_rows()


@st.cache_data(show_spinner="Rechne die Karte über Nullstelle und R (91 Zellen bis N = 400) …")
def _map_cR():
    return ev.map_c_R()


@st.cache_data(show_spinner="Rechne die Karte über ε und R (28 Zellen bis N = 400) …")
def _map_Re():
    return ev.map_R_eps()


@st.cache_data(show_spinner=False)
def _cell(c, eps, R):
    return ev.trap_cell(c, eps, R)


@st.cache_data(show_spinner=False)
def _wagen(R_text):
    return ev.wagen_rows(Fraction(R_text))


@st.cache_data(show_spinner=False)
def _wagen_inf(R_text):
    plant = D.build_plant("wagen", float(Fraction(R_text)))
    return D.rho_of_gain(plant, D.dare(plant)[1])


@st.cache_data(show_spinner=False)
def _wagen_paths(R_text, x0):
    return ev.wagen_paths(Fraction(R_text), x0)


@st.cache_data(show_spinner="Rechne das Vier-Tank-System …")
def _vt_curve(R, pf):
    return ev.vier_tank_curve(R, pf)


@st.cache_data(show_spinner="Rechne die Vier-Tank-Tabelle (7 Werte von R) …")
def _vt_table():
    return ev.vier_tank_table()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in hfa_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📉 Die Falle des endlichen Horizonts")
st.markdown(
    """
Ein **Plan über N Schritte**, in jedem Schritt neu gerechnet und nur im ersten Stellwert angewendet (das ist MPC ohne Beschränkungen), ergibt ein festes Regelgesetz **u = −K_N x**. Die Demo misst in diesem Modell
(linear, ohne Beschränkungen, ohne Störung), **wann dieses Gesetz stabil ist**: In Kalmans Gegenbeispiel zum endlichen Horizont destabilisiert der Plan eine offen stabile Strecke bei N = 1 bis 6 und ist erst ab N = 7 stabil.
Wählt man als **Terminalkosten** die Lösung der algebraischen Riccati-Gleichung, ist der Gewinn für jedes N gleich dem des unendlichen Horizonts. Das ist Stück 3 der Linie „Regelung (von LQR bis MPC)“; es setzt den
Gewinn K des unendlichen Horizonts aus [`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo) voraus. Alle Zahlen sind **eigene Nachrechnung** in diesem Modell; wo eine Quelle gelesen ist, steht sie in den Formeln unten.

Weiter unten: die **Kernfragen** mit Verdikt, sechs Experimente (🎯, 🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Plan über N Schritte", expanded=True):
    st.markdown(
        """
1. **Strecke und Kosten:** Der Zustand x ändert sich nach x⁺ = A x + B u. Der Plan wählt u₀ bis u_(N−1) so, dass die Kosten Σ (x′Qx + u′Ru) über N Schritte plus das **Terminalgewicht** x_N′ P_f x_N am kleinsten sind.
2. **Das Gesetz K_N:** Rückwärts gerechnet (Riccati-Rekursion) hängt der erste Stellwert linear vom Zustand ab, **u₀ = −K_N x**. MPC ohne Beschränkungen wendet nur diesen Stellwert an und rechnet im nächsten Schritt neu: das Gesetz ist über die Zeit **fest**.
3. **Stabil heißt ρ < 1:** Die **Pole** sind die Eigenwerte von A − B K_N, **ρ** ist der Betrag des größten Pols (dieselbe Größe wie das ρ in [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo)). ρ = 1 heißt grenzstabil, ρ > 1 instabil.
4. **Die Falle:** Optimal für N Schritte heißt nicht stabil, wenn der Plan wiederholt wird. Ob und bei welchem N das passiert, hängt hier an der Strecke, an R, Q und am Terminalgewicht (Messungen unten, nicht als Satz).
5. **Vorzeichen und Faktor:** Gesetz u = −K x (K positiv bei Rückführung); die Kosten haben keinen Faktor ½. Gewinn und Kostenmatrix hängen davon nicht ab.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Strecke**")
    system = st.selectbox("Strecke", list(C.SYSTEMS), format_func=lambda k: C.SYSTEM_LABELS[k], key="system_select",
                          help="Kalmans Gegenbeispiel (Rawlings/Mayne/Diehl, 1.3.4), der Wagen (Doppelintegrator, Abtastzeit 1) oder das linearisierte Vier-Tank-System als modellbasierte Entsprechung (Berberich u. a., Abschnitt V).")
    is_kalman = system == "kalman"
    c = st.select_slider("Nullstellenparameter c", options=list(C.C_OPTIONS), format_func=ev.c_label, key="c_select", disabled=not is_kalman,
                         help="Nur für Kalmans Gegenbeispiel: Ausgang C = (−c, 1), die Nullstelle der Strecke liegt bei 1/c. c = 2/3 ist das Beispiel der Quelle.")
    eps = st.select_slider("Zustandsgewicht ε", options=list(C.EPS_OPTIONS), format_func=r_label, key="eps_select", disabled=not is_kalman,
                           help="Nur für Kalmans Gegenbeispiel: Q = C′C + ε·I. Je kleiner ε, desto weniger zählt der Zustand außerhalb des Ausgangs.")
    st.caption({"kalman": "Q = C′C + ε·I, ein Eingang.", "wagen": "Q = I, ein Eingang (Beschleunigung).", "vier_tank": "Q = 3·C′C (Ausgang: die ersten zwei Zustände), R·I mit zwei Eingängen."}[system])
    st.markdown("**Gewichte und Horizont**")
    R = st.select_slider("Stellgewicht R", options=list(C.R_OPTIONS), format_func=r_label, key="r_select", help="Je kleiner R, desto billiger der Stelleingriff.")
    N = st.slider("Horizont N", *bounds("n_slider"), key="n_slider", help="Länge des Plans in Schritten; gerechnet wird für N = 1 bis 150.")
    pf = st.selectbox("Terminalgewicht P_f", list(C.PF_KINDS), format_func=lambda k: C.PF_LABELS[k], key="pf_select",
                      help="Bewertung des Zustands nach dem letzten Schritt des Plans: 0, Q, γ·I oder γ·P_∞ (P_∞ ist die Lösung der algebraischen Riccati-Gleichung).")
    uses_gamma = pf in ("gamma_i", "gamma_pinf")
    gamma = st.select_slider("Faktor γ", options=list(C.GAMMA_OPTIONS), format_func=lambda g: de(g, 1) if g < 10 else fmt_int(int(g)), key="gamma_select", disabled=not uses_gamma,
                             help="Nur für P_f = γ·I und P_f = γ·P_∞.")

N = int(N)
settings = Settings(system, float(c), float(eps), float(R), N, pf, float(gamma))
sync_query_params({"system_select": system, "c_select": float(c), "eps_select": float(eps), "r_select": float(R), "n_slider": N, "pf_select": pf, "gamma_select": float(gamma)})
a = _analysis(settings)
n_show = C.N_SHOW[system]

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Experiment 1: Das Gesetz des Horizonts N")
m1, m2, m3, m4, m5 = st.columns(5)
state = "instabil" if a.rho_sel >= 1.0 + D.TOL else ("grenzstabil" if a.rho_sel >= 1.0 - D.TOL else "stabil")
m1.metric(f"ρ des Gesetzes K_N (N = {N})", de(a.rho_sel, 3), delta=state, delta_color="off", help="Betrag des größten Pols von A − B K_N. Unter 1: stabil; 1: grenzstabil; darüber: instabil.")
m2.metric("ρ für N → ∞", de(a.rho_inf, 3), delta="unendlicher Horizont", delta_color="off", help="Gewinn K_∞ aus der algebraischen Riccati-Gleichung (SciPy).")
m3.metric("ρ der offenen Strecke", de(a.rho_open, 3), delta="ohne Regelung", delta_color="off", help="Betrag des größten Eigenwerts von A.")
m4.metric("Kleinstes stabiles N", str(a.first_stable) if a.first_stable is not None else "keines", delta=f"Terminalgewicht: {a.pf_label}", delta_color="off",
          help="Kleinster Horizont mit ρ < 1; danach können weitere instabile folgen (siehe die Zahl rechts).")
m5.metric(f"Instabile N (1 bis {C.N_MAX})", str(len(a.unstable)), delta=("danach keines bis N = 150" if a.first_stable is not None and a.later_unstable == 0 and a.unstable else ("danach wieder" if a.later_unstable else None)), delta_color="off",
          help="Zahl der Horizonte mit ρ ≥ 1 (grenzstabil zählt mit).")
col_l, col_r = st.columns([1.5, 1])
col_l.plotly_chart(build_rho_curve(a.rho, N, a.rho_open, a.rho_inf, n_show), width="stretch", key="rho_chart")
col_r.plotly_chart(build_poles(a.poles_sel, a.poles_inf, a.poles_open, N), width="stretch", key="poles_chart")
st.caption(f"Links: ρ des Gesetzes K_N für N = 1 bis {n_show}, rot ab ρ ≥ 1, der gewählte Horizont hat einen Rahmen. Rechts: die Pole von A − B K_N in der komplexen Ebene (kleine Wirkung bei N → ∞: blau).")
st.plotly_chart(build_trajectory(a.traj_sel, a.traj_inf, N), width="stretch", key="traj_chart")
st.caption("Verlauf von |x_k| im geschlossenen Kreis ohne Störung, Anfangszustand x₀ = (1, 0, …). Ein Gesetz mit ρ > 1 lässt |x| im Mittel um den Faktor ρ je Schritt wachsen (die Kurve kann am Anfang davon abweichen).")

st.markdown("## ❓ Kernfrage 1: Wie lang muss der Horizont sein, damit das Gesetz stabil ist?")
kind, text = ev.verdict_horizon_analysis(a)
getattr(st, kind)(text)
st.caption("Alle Zahlen: in diesem Modell, ohne Beschränkungen und ohne Störung, Gesetz u = −K_N x in jedem Schritt wiederholt; „stabil“ heißt ρ < 1 (Band 10⁻⁹), „grenzstabil“ ρ = 1.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2 (Kernfrage 2): Was ändern die Terminalkosten?")
st.caption("Das Terminalgewicht P_f bewertet den Zustand nach dem letzten Schritt des Plans. Gewählt wird es in der Seitenleiste (erste Zeile, ►); darunter zwölf feste Gewichte. Tabelle: ρ des Gesetzes K_N für N = 1 bis 12, "
           "▲ markiert ρ ≥ 1. Die letzte Zeile ist die harte Bedingung x(N) = 0 (Grenzfall P_f → ∞, ohne Beschränkungen; für N = 1 nicht erfüllbar).")
table = _terminal(settings)
st.dataframe(ev.terminal_frame(table), hide_index=True, width="stretch")
kind, text = ev.verdict_terminal(a.label, table)
getattr(st, kind)(text)
st.caption("Die Zählung der instabilen N reicht bis N = 150. Ein Zusammenhang „längerer Horizont ist stabiler“ gilt hier nicht allgemein; die benannten Fälle stehen im Verdikt.")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3 (Kernfrage 3): Was kostet ein Horizont, der stabil, aber kurz ist?")
st.caption("Kosten der unendlichen Zukunft unter dem Gesetz K_N im schlechtesten Anfangszustand, geteilt durch die optimalen Kosten (Lyapunov-Gleichung; größter verallgemeinerter Eigenwert von S gegen P_∞). Rot hinterlegt: Horizonte, bei denen das Gesetz nicht stabil ist.")
ratios = _price(settings)
st.plotly_chart(build_price(ratios, N), width="stretch", key="price_chart")
kind, text = ev.verdict_price(a.label, a.pf_label, N, a.ratio_sel, ratios, a.rho_sel)
getattr(st, kind)(text)
with st.expander("Der Plan über N Schritte als zeitvarianter Ablauf (K_N, K_(N−1), …, K_1)", expanded=False):
    st.markdown("So ist der Plan über N Schritte gedacht: nicht ein festes Gesetz, sondern der Reihe nach K_N, K_(N−1), …, K_1. Die Tabelle zeigt für dieses Beispiel, was das Produkt der N geschlossenen Matrizen tut.")
    tv = _timevarying(settings)
    st.dataframe(ev.timevarying_frame(tv), hide_index=True, width="stretch")
    kind, text = ev.verdict_timevarying(a.label, a.pf_label, tv)
    getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wann gibt es die Falle?")
st.caption("Kalmans Gegenbeispiel als Familie: Ausgang C = (−c, 1), die Nullstelle der Strecke liegt bei 1/c (c = 2/3: Nullstelle 1,5, das Beispiel der Quelle); Q = C′C + ε·I, P_f = Q. Die Zahl in jeder Zelle ist das **kleinste stabile N** "
           f"(1 heißt: kein instabiler Horizont bis N = {C.N_SEARCH}). Schwarzer Rahmen: der gewählte Fall der Seitenleiste, soweit er auf dem Raster liegt.")
rows_null = _null_rows()
mp = _map_cR()
c_labels = [("2/3" if x == 2 / 3 else de(x, 1)) + (f" (z₀ = {de(1 / x, 2)})" if x else " (keine)") for x in mp["c"]]
mark_c = c_labels[mp["c"].index(float(c))] if float(c) in mp["c"] else None
mark_R = r_label(float(R)) if float(R) in mp["R"] else None
st.markdown("**Nullstelle (c) gegen Stellgewicht R**, ε = 10⁻³")
st.plotly_chart(build_map(mp["first"], c_labels, [r_label(x) for x in mp["R"]], "Stellgewicht R", "c (Nullstelle z₀ = 1/c)", (mark_c, mark_R) if mark_c and mark_R and abs(float(eps) - ev.NULL_PLANT_EPS) < 1e-12 else None),
                width="stretch", key="map_cr_chart")
mq = _map_Re()
st.markdown("**Zustandsgewicht ε gegen Stellgewicht R**, c = 2/3 (Nullstelle 1,5)")
st.plotly_chart(build_map(mq["first"], [r_label(x) for x in mq["eps"]], [r_label(x) for x in mq["R"]], "Stellgewicht R", "Zustandsgewicht ε",
                          (r_label(float(eps)), r_label(float(R))) if float(eps) in mq["eps"] and float(R) in mq["R"] and abs(float(c) - 2 / 3) < 1e-12 else None), width="stretch", key="map_re_chart")
frozen_c = ev.load_frozen()["random_c"]
seed = st.select_slider("Zufalls-Seed der Zufallssysteme", options=list(C.SEEDS), key="seed_select", value=C.DEFAULT_SEED,
                        help="Je Zufalls-Seed sind 6 000 Zufallssysteme (n = 2, ein Eingang) vorgerechnet; die Zählung für alle 20 Zufalls-Seeds steht im Diagramm.")
rnd = ev.random_row(int(seed))
st.plotly_chart(build_random(ev.load_frozen()["random"], int(seed)), width="stretch", key="random_chart")
cell = _cell(float(c), float(eps), float(R)) if is_kalman else None
counts = ev.count_nullstelle(rows_null)
kind, text = ev.verdict_trap(counts, rnd, cell, float(c), float(R), float(eps), int(seed))
getattr(st, kind)(text)
rs = ev.random_summary()
st.caption(f"Vorgerechnet (`tools/make_frozen.py`, `data/hfa_frozen.json`): 20 Zufalls-Seeds × 6 000 Zufallssysteme; der Anteil der offen stabilen Strecken mit instabilem N = 1 liegt je Zufalls-Seed zwischen {pct(rs['share_min'])} und {pct(rs['share_max'])}. "
           f"Gegenprobe über {frozen_c['n']} Zufallswerte c in (0,05; 4): Falle bei Nullstelle außerhalb des Einheitskreises in {frozen_c['outside_trap']} von {frozen_c['outside']}, sonst in {frozen_c['rest_trap']} von {frozen_c['rest']}. "
           "Ob die Nullstelle die **Ursache** ist, folgt aus dieser Messung nicht (eine Familie, ein Q, ein paar R; keine andere Eingangsmatrix B, keine Mehrgrößenstrecken außer dem Vier-Tank-System).")
with st.expander("Tabelle der 13 Nullstellen (ε = 10⁻³, R = 10⁻³, P_f = Q)", expanded=False):
    st.dataframe(ev.nullstelle_frame(rows_null), hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Der Wagen")
st.caption("Der Wagen (Doppelintegrator, Abtastzeit 1) mit Q = I und P_f = Q; unabhängig von der Strecke der Seitenleiste. Die Gewinne der Tabelle sind in Brüchen gerechnet (exakt), das Phasenbild in Gleitkomma.")
w1, w2, w3 = st.columns(3)
R_text = w1.select_slider("Stellgewicht R (Wagen)", options=list(C.WAGEN_R_OPTIONS), key="wagen_r", value="1", help="Als Bruch, damit die Tabelle exakt rechnen kann.")
x1 = w2.slider("Anfangsposition x₁(0)", min(C.WAGEN_X0_POS), max(C.WAGEN_X0_POS), C.DEFAULT_WAGEN_X0[0], step=0.5, key="wagen_x1")
x2 = w3.slider("Anfangsgeschwindigkeit x₂(0)", -3.0, 3.0, C.DEFAULT_WAGEN_X0[1], step=0.5, key="wagen_x2")
Rw = Fraction(R_text)
rows_w = _wagen(R_text)
rho_w_inf = _wagen_inf(R_text)
x0w = (float(x1), float(x2))
st.dataframe(ev.wagen_frame(rows_w, rho_w_inf), hide_index=True, width="stretch")
st.plotly_chart(build_phase(_wagen_paths(R_text, x0w), x0w, ev.wagen_endpoint(Rw, x0w)), width="stretch", key="phase_chart")
kind, text = ev.verdict_wagen(Rw, rows_w, rho_w_inf, x0w)
getattr(st, kind)(text)
st.caption("N = 1: der Wagen bremst die Geschwindigkeit ab, aber nichts im Gesetz holt die Position zurück (K_1 hat keinen Anteil auf der Position). Ab N = 2 sind in den gemessenen Fällen R = 1 bis 10⁻³ beide Pole innerhalb des Einheitskreises.")

st.markdown("---")

# --- Experiment 6 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Das Vier-Tank-System (modellbasierte Entsprechung)")
st.caption("Linearisiertes Vier-Tank-System, Q = 3·C′C und R·I wie bei Berberich u. a. (arXiv 1906.04679, Abschnitt V). Die Matrizen sind aus der Textextraktion der Quelle gelesen, die Zeilenfolge von B ist nicht gegen das Seitenbild geprüft. "
           "Hier das **modellbasierte** Gesetz K_N (Riccati-Rekursion); das Schema der Quelle ist datenbasiert und regularisiert und ist nicht nachgerechnet. Unabhängig von der Strecke der Seitenleiste.")
v1, v2 = st.columns(2)
vt_R = v1.select_slider("Stellgewicht R (Vier-Tank)", options=list(C.R_DECADES), format_func=r_label, key="vt_r", value=1e-2)
vt_pf = v2.selectbox("Terminalgewicht (Vier-Tank)", ["q", "null", "gamma_pinf"], format_func=ev.pf_name_vier_tank, key="vt_pf")
cur = _vt_curve(float(vt_R), vt_pf)
st.plotly_chart(build_rho_curve(cur["rho"], 1, cur["rho_open"], cur["rho_inf"], C.N_MAX), width="stretch", key="vt_chart")
kind, text = ev.verdict_vier_tank(float(vt_R), vt_pf, cur)
getattr(st, kind)(text)
st.caption("Die Schwelle ist knapp: bei R = 10⁻⁴ und P_f = Q liegt ρ bei N = 55 nur etwa 0,3 % unter 1; sie hängt an den drei Stellen der Matrixeinträge der Quelle und ist nur gemessen (Band, nicht bitgleich über Plattformen).")
with st.expander("Tabelle: instabile Horizonte N ≤ 150 für sieben Werte von R", expanded=False):
    st.dataframe(ev.vier_tank_frame(_vt_table()), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gewinn des unendlichen Horizonts (K_∞) ist gegeben** | Hier ist K_∞ der Vergleich; seine Herleitung, die Riccati-Gleichung und ihre Konventionen stehen im LQR-Stück. | [`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo) |
| **Keine Beschränkungen** | „P_f = P_∞ gibt K_N = K_∞“ gilt nur ohne Stell- und Zustandsgrenzen. Mit Grenzen wird aus dem Plan ein QP, und Terminalmenge und Zulässigkeit entscheiden. | [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo), [`mpc-zulaessigkeit-demo`](https://github.com/sebastian-hanisch/mpc-zulaessigkeit-demo) |
| **Kein Rauschen, keine Störung** | Nur das nominelle Modell ist gerechnet; die Robustheit der Gesetze K_∞ und K_N gegen Modellfehler und Reserven ist nicht gemessen. | [`lqr-reserven-demo`](https://github.com/sebastian-hanisch/lqr-reserven-demo), [`plan-stoerung-demo`](https://github.com/sebastian-hanisch/plan-stoerung-demo) |
| **Rechnen ohne Zeit** | Die Rechenzeit des Plans zählt hier nicht; als Verzug von Perioden ist sie ein eigenes Stabilitätsproblem. | [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo) |
| **Eine Familie, ein Q** | Die Abhängigkeit von der Nullstelle ist an Kalmans Strecke mit Q = C′C + ε·I gemessen; Ursache, andere B und Mehrgrößenstrecken sind nicht gemessen. | – |
| **Terminalgewichte nur aus der Auswahl** | Ein Gewicht γ·P_f ohne Terminalbedingung (Rawlings, Kapitel 2, S. 144 bis 147) und ein Steuerhorizont m > 1 (Grüne u. a.) sind nicht nachgerechnet; die Mindest-N-Schranke von Grüne u. a. ist nicht verglichen. | – |
| **Vier-Tank nur modellbasiert** | Das Schema der Quelle (datenbasiert, regularisiert, mit Schlupfvariable) ist nicht nachgerechnet; die Matrizen stammen aus einer Textextraktion. | – |
| **Eine Rollende-Horizont-Demo ohne Zustandsraum** | [`rolling-horizon-demo`](https://github.com/sebastian-hanisch/rolling-horizon-demo) hat Quantilregeln auf einer AR(1)-Nachfrage, keinen Zustandsraum und keinen Stabilitätsbegriff; „Rolling“ meint dort etwas anderes als das Neu-Rechnen eines Optimierungsproblems hier. | [`rolling-horizon-demo`](https://github.com/sebastian-hanisch/rolling-horizon-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind eine lineare Strecke $x^+ = Ax + Bu$, Gewichte $Q \succeq 0$, $R \succ 0$, ein Terminalgewicht $P_f \succeq 0$ und ein Horizont $N \ge 1$; gesucht sind das Gesetz $u = -K_N x$ des ersten Stellwerts und die Frage, ob $A - BK_N$ stabil ist.

| Symbol | Bedeutung |
|---|---|
| $A$, $B$ | Zustands- und Eingangsmatrix der Strecke, $x \in \mathbb{R}^n$, $u \in \mathbb{R}^m$ |
| $Q$, $R$, $P_f$ | Gewicht des Zustands, der Stellgröße und des Endzustands |
| $\Pi_j$ | Kostenmatrix nach $j$ Schritten der Rückwärtsrechnung, $\Pi_0 = P_f$ |
| $K_N$, $K_\infty$ | Gewinn der ersten Stufe des Horizonts $N$, Gewinn des unendlichen Horizonts |
| $P_\infty$ | Lösung der algebraischen Riccati-Gleichung (DARE) |
| $\rho$ | Betrag des größten Eigenwerts von $A - BK$ (Spektralradius) |
| $c$, $z_0$ | Parameter des Ausgangs $C = (-c, 1)$ in Kalmans Beispiel, Nullstelle $z_0 = 1/c$ |

**Kosten und Gesetz.** Der Plan über $N$ Schritte minimiert (Konvention: $u = -Kx$, kein Faktor $\tfrac12$)

$$J_N = \sum_{k=0}^{N-1} \left( x_k^\top Q x_k + u_k^\top R u_k \right) + x_N^\top P_f\, x_N \quad \text{mit } x_{k+1} = A x_k + B u_k .$$

**Rückwärtsrechnung (Riccati).** Mit $\Pi_0 = P_f$ und für $j = 1, 2, \dots$

$$K_j = (R + B^\top \Pi_{j-1} B)^{-1} B^\top \Pi_{j-1} A, \qquad \Pi_j = Q + A^\top \Pi_{j-1} (A - B K_j) .$$

$K_N$ ist der Gewinn der ersten Stufe des Horizonts $N$. MPC ohne Beschränkungen wendet $u = -K_N x$ in jedem Schritt an; der Regelkreis ist stabil, wenn $\rho_N = \rho(A - BK_N) < 1$ ist.

**Unendlicher Horizont.** $P_\infty$ löst $P = Q + A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A$ und $K_\infty = (R + B^\top P_\infty B)^{-1} B^\top P_\infty A$.

**Terminalgewicht gleich der Riccati-Lösung.** Gilt $\Pi_0 = P_f = P_\infty$, dann ist $\Pi_j = P_\infty$ für alle $j$ und deshalb $K_N = K_\infty$ für jedes $N$ (Induktion über $j$; eigene Nachrechnung, in Rawlings, Abschnitt 2.5.1, S. 132, für $K_N$ unabhängig von $N$ in eigener Fassung gelesen). Für $P_f \ne P_\infty$ gilt das nicht.

**Hart gesetzte Endbedingung.** Verlangt man $x_N = 0$ statt eines Gewichts, ist das der Grenzfall $P_f \to \infty$; die Stellwerte $u_0, \dots, u_{N-1}$ lösen ein Gleichungssystem (KKT), und das Gesetz ist aus dem ersten Block gewonnen. Für Kalmans Beispiel ist $N = 2$ ein Deadbeat-Gesetz.

**Zeitvarianter Plan.** Der Plan über $N$ Schritte wendet $K_N, K_{N-1}, \dots, K_1$ nacheinander an; sein $N$-Schritt-Produkt ist $\prod_{j} (A - BK_j)$ in dieser Reihenfolge.

**Kosten des kurzen Horizonts.** Unter dem festen Gesetz $K$ gilt $x'Sx$ mit $(A - BK)^\top S (A - BK) - S = -(Q + K^\top R K)$ als Kosten der unendlichen Zukunft; das Verhältnis zu den optimalen Kosten im schlechtesten Anfangszustand ist der größte Eigenwert von $P_\infty^{-1} S$.

**Kalmans Gegenbeispiel (Rawlings/Mayne/Diehl, Abschnitt 1.3.4, S. 21–22).** $A = \begin{pmatrix} 4/3 & -2/3 \\ 1 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, Ausgang $C = (-2/3,\ 1)$, $Q = C^\top C + \varepsilon I$, $R = 10^{-3}$ und $P_f = Q$; die Quelle nennt dort die Eigenwerte für $N = 5$ und $N = 7$, die hier bestätigt sind. Die Familie mit $C = (-c, 1)$ ist eigene Wahl.

**Wagen, Handrechnung.** Mit $Q = I$, $P_f = Q$ ist $K_1 = (0,\ 1/(R+1))$ und $A - BK_1 = \begin{pmatrix} 1 & 1 \\ 0 & R/(R+1) \end{pmatrix}$ mit den Eigenwerten $1$ und $R/(R+1)$. Der Vektor $(1,\ R+1)$ ist Linkseigenvektor zum Eigenwert 1; deshalb endet der Wagen bei $x_1(0) + (R+1)\,x_2(0)$.

**Literatur und Quellenstand.** Gelesen: Rawlings/Mayne/Diehl, *Model Predictive Control: Theory, Computation, and Design*, 1.3.4 (S. 21–22), 2.5.1 (S. 132) und Kapitel 2, S. 144–147 (Terminalgewicht $\gamma$ ohne Terminalbedingung, hier nicht nachgerechnet); nur Gleichungen und Befunde in eigenen Worten. Grüne/Pannek/Seehafer/Worthmann, arXiv:1006.2529 (stabil für genügend großes $N$ unter Steuerbarkeitsannahme, Zulässigkeit ausgeklammert; ihre Mindest-$N$-Schranke ist hier nicht gemessen). Yoon/Kim, arXiv:2607.04006 (2026), nur Abstract und Einleitung gelesen: bei DARE-Terminalkosten stimme der erste Stellwert mit dem LQR für jeden Horizont überein; hier selbst nachgerechnet, ihr Beweis nicht geprüft. Berberich/Köhler/Müller/Allgöwer, arXiv:1906.04679, Abschnitt V (Vier-Tank-System, gelesener Satz). Alle übrigen Befunde sind **eigene Nachrechnung in diesem Modell**; eine Neuheitsaussage gibt es nicht.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
