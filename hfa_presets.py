"""Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben). Die Regler der Seitenleiste sind Strecke, Nullstellenparameter c,
Zustandsgewicht ε, Stellgewicht R, Horizont N und Terminalgewicht (Art und γ). Die Demo hat keinen Zufalls-Seed in der Seitenleiste (der Zufalls-Seed der Zufallssysteme
steht beim Experiment 4)."""

import math
from dataclasses import dataclass

import streamlit as st

import hfa_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # erlaubte Werte (Auswahlregler); Zahlen rasten auf den nächsten ein, Texte müssen enthalten sein


SETTING_SPECS = {
    "system_select": SettingSpec("strecke", str, "kalman", options=C.SYSTEMS),
    "c_select": SettingSpec("c", float, C.DEFAULT_C, min(C.C_OPTIONS), max(C.C_OPTIONS), C.C_OPTIONS),
    "eps_select": SettingSpec("eps", float, C.DEFAULT_EPS, min(C.EPS_OPTIONS), max(C.EPS_OPTIONS), C.EPS_OPTIONS),
    "r_select": SettingSpec("r", float, C.DEFAULT_R, min(C.R_OPTIONS), max(C.R_OPTIONS), C.R_OPTIONS),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "pf_select": SettingSpec("pf", str, C.DEFAULT_PF, options=C.PF_KINDS),
    "gamma_select": SettingSpec("gamma", float, C.DEFAULT_GAMMA, min(C.GAMMA_OPTIONS), max(C.GAMMA_OPTIONS), C.GAMMA_OPTIONS),
}
PRESET_KEYS = {"system": "system_select", "c": "c_select", "eps": "eps_select", "R": "r_select", "N": "n_slider", "pf": "pf_select", "gamma": "gamma_select"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(spec: SettingSpec, value):
    """Wert auf den erlaubten Bereich bringen: Zahlenoptionen rasten auf den nächsten Wert ein, Zahlen ohne Optionen werden begrenzt; None, wenn ein Text nicht erlaubt ist."""
    if spec.caster is str:
        return value if value in spec.options else None
    if not math.isfinite(value):
        return None
    if spec.options is not None:
        return min(spec.options, key=lambda o: abs(o - value))
    return max(spec.lo, min(spec.hi, value))


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, begrenzt bzw. auf erlaubte Stufen eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = _snap(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def _gleich(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def inaktive_regler() -> set:
    """Regler, die beim aktuellen Stand wirkungslos sind (ausgegraut): c und ε außerhalb von Kalmans Gegenbeispiel, γ ohne γ·I und γ·P_∞. Ihr Wert zählt beim Preset-Vergleich nicht."""
    out = set()
    if st.session_state.get("system_select", SETTING_SPECS["system_select"].default) != "kalman":
        out |= {"c_select", "eps_select"}
    if st.session_state.get("pf_select", SETTING_SPECS["pf_select"].default) not in ("gamma_i", "gamma_pinf"):
        out.add("gamma_select")
    return out


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen (Gleitkommawerte relativ auf 1e-9), sonst None.

    Zustandslos: verglichen wird jeder wirksame Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    tests/test_layout.py prüft für jeden Regler, dass seine Änderung die Markierung aufhebt."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    inaktiv = inaktive_regler()
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items() if state_key not in inaktiv
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )
