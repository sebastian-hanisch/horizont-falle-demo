"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Drei Regler sind je nach Strecke bzw. Terminalgewicht wirkungslos und stehen
dann ausgegraut (`disabled`): c und ε außerhalb von Kalmans Gegenbeispiel, γ ohne γ·I und γ·P_∞. Ihr Wert wird beim Vergleich mit den Presets nur berücksichtigt, wenn sie wirken
(`hfa_presets.aktives_preset` ruft dafür `inaktive_regler`)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hfa_constants as C
import hfa_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _anderer_wert(regler, spec):
    if spec.options is not None:
        return next(o for o in spec.options if o != regler.value)
    return spec.hi if regler.value != spec.hi else spec.lo


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"


@pytest.mark.parametrize("state_key", ["system_select", "r_select", "n_slider", "pf_select"])
def test_jede_aenderung_eines_stets_wirksamen_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(regler, spec)).run()
    assert not at.exception
    assert "primary" not in _typen(at).values()
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()           # erneutes Anklicken markiert wieder
    assert _typen(at)[name] == "primary"


@pytest.mark.parametrize("state_key", ["c_select", "eps_select"])
def test_c_und_eps_heben_auf_bei_kalman_und_sind_beim_wagen_wirkungslos(state_key):
    at = _app()                                                      # Standard: Kalmans Gegenbeispiel, die beiden Regler wirken
    regler = _regler(at, state_key)
    assert not regler.proto.disabled
    regler.set_value(_anderer_wert(regler, P.SETTING_SPECS[state_key])).run()
    assert "primary" not in _typen(at).values()
    at = _app()
    at.sidebar.button(key="preset_Wagen (N = 1 grenzstabil)").click().run()
    assert _regler(at, state_key).proto.disabled                     # ausgegraut: im Browser nicht änderbar
    at.session_state[state_key] = _anderer_wert(_regler(at, state_key), P.SETTING_SPECS[state_key])   # nur der Zustand: der Wert zählt beim Vergleich nicht
    at.run()
    assert _typen(at)["Wagen (N = 1 grenzstabil)"] == "primary"


def test_gamma_hebt_auf_bei_gamma_pinf_und_ist_sonst_wirkungslos():
    at = _app()
    assert _regler(at, "gamma_select").proto.disabled               # Standard: P_f = Q
    at.session_state["gamma_select"] = 10.0
    at.run()
    assert _typen(at)[next(iter(C.PRESETS))] == "primary"
    at.sidebar.button(key="preset_Terminalkosten = Riccati-Lösung").click().run()
    assert not _regler(at, "gamma_select").proto.disabled
    _regler(at, "gamma_select").set_value(10.0).run()
    assert "primary" not in _typen(at).values()
