"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen, Einrasten); jedes Preset ist rechnerisch gültig."""

import hfa_constants as C
import hfa_evaluation as ev
import hfa_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_options():
    for name, preset in C.PRESETS.items():
        assert preset["system"] in C.SYSTEMS and preset["pf"] in C.PF_KINDS, name
        assert preset["c"] in C.C_OPTIONS and preset["eps"] in C.EPS_OPTIONS and preset["R"] in C.R_OPTIONS and preset["gamma"] in C.GAMMA_OPTIONS, name
        assert C.N_MIN <= preset["N"] <= C.N_MAX, name


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert ev.Settings(first["system"], first["c"], first["eps"], first["R"], first["N"], first["pf"], first["gamma"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_preset_computes_without_error_and_with_finite_numbers():
    for name, p in C.PRESETS.items():
        a = ev.analyse(ev.Settings(p["system"], p["c"], p["eps"], p["R"], p["N"], p["pf"], p["gamma"]))
        assert a.rho_sel > 0 and a.rho_inf > 0, name


def test_snap_limits_numbers_and_rejects_unknown_texts():
    assert P._snap(P.SETTING_SPECS["n_slider"], 999) == C.N_MAX and P._snap(P.SETTING_SPECS["n_slider"], -5) == C.N_MIN
    assert P._snap(P.SETTING_SPECS["r_select"], 0.0021) == 3e-3
    assert P._snap(P.SETTING_SPECS["c_select"], 0.66) == 2 / 3
    assert P._snap(P.SETTING_SPECS["system_select"], "wagen") == "wagen" and P._snap(P.SETTING_SPECS["system_select"], "x") is None
    assert P._snap(P.SETTING_SPECS["r_select"], float("nan")) is None
