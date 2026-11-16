"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler (Schieberegler,
Auswahl-Schieberegler, Auswahlfelder) an seine Grenzen bzw. auf jede Option und prüft, dass kein Python-Fehler auftritt (insbesondere `StreamlitDuplicateElementId` bei
st.plotly_chart ohne eindeutiges key= und Fehler bei ausgegrauten Reglern)."""

import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hfa_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title[0].value
    assert len(at.get("plotly_chart")) >= 8                          # Experiment 1 (3), 3, 4 (3), 5, 6


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


OPTIONEN = {   # echte Werte je Schlüssel (AppTest zeigt die formatierten Texte, set_value braucht den Wert)
    "c_select": C.C_OPTIONS, "eps_select": C.EPS_OPTIONS, "r_select": C.R_OPTIONS, "gamma_select": C.GAMMA_OPTIONS, "seed_select": C.SEEDS, "wagen_r": C.WAGEN_R_OPTIONS,
    "vt_r": C.R_DECADES, "system_select": C.SYSTEMS, "pf_select": C.PF_KINDS, "vt_pf": ("q", "null", "gamma_pinf"),
}


def test_every_select_slider_and_selectbox_on_every_option_does_not_raise():
    probe = _fresh_app()
    widgets = [("select_slider", w.key) for w in probe.select_slider] + [("selectbox", w.key) for w in probe.selectbox]
    assert {k for _, k in widgets} == set(OPTIONEN), "jeder Auswahlregler braucht seine Optionen in OPTIONEN"
    for kind, key in widgets:
        options = OPTIONEN[key]
        if getattr(probe, kind)(key=key).proto.disabled:
            continue                                                 # ausgegraute Regler lassen sich auch im Browser nicht ändern (AppTest lehnt es ab); sie kommen im nächsten Test dran
        for option in (options[0], options[len(options) // 2], options[-1]):
            at = _fresh_app()
            getattr(at, kind)(key=key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (kind, key, option, [str(e) for e in at.exception])


def test_every_system_with_every_terminal_weight_renders_all_sections():
    for system in ("kalman", "wagen", "vier_tank"):
        for pf in ("null", "q", "gamma_i", "gamma_pinf"):
            at = _fresh_app()
            at.sidebar.selectbox(key="system_select").set_value(system).run(timeout=TIMEOUT)
            at.sidebar.selectbox(key="pf_select").set_value(pf).run(timeout=TIMEOUT)
            assert not at.exception, (system, pf, [str(e) for e in at.exception])
            assert len(at.dataframe) >= 4 and len(at.subheader) >= 6
            if pf in ("gamma_i", "gamma_pinf"):
                for gamma in (C.GAMMA_OPTIONS[0], C.GAMMA_OPTIONS[-1]):                 # γ wirkt nur hier, sonst ausgegraut
                    at.sidebar.select_slider(key="gamma_select").set_value(gamma).run(timeout=TIMEOUT)
                    assert not at.exception, (system, pf, gamma, [str(e) for e in at.exception])
