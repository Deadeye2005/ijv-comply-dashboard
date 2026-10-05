"""
IJV-Comply dashboard: patient-parameterised jugular-vein testbed (autoregulated outflow, separate outlets, slit-lumen hydraulics).

Run:
    pip install -r requirements.txt
    streamlit run app.py

Needs `ijv_math_engine.py` in the same folder. Optional: copy `config.toml` to `.streamlit/config.toml`.

Workflow (mirrors the engine):
    1. PatientProfile(A0, K, P_ext, Q_ref, ...) from the sidebar
    2. calibrate_network(patient)                 -> recalculated whenever a sidebar value changes (cached)
    3. run_scenario(name, patient, network, ...)  -> prescribed total outflow and two outlet pressures in;
                                                     flow split (jugular flow may reverse) and P_in out (cached)

Layout: sidebar = patient + overrides + advanced model assumptions | top = scenario buttons and their inputs |
middle = solved readout and animated flow | then the exact reversal-threshold equation with a live numerical check |
bottom = regime map over (P_ext, P_out_ijv) with the threshold line, solved-P_in contours and a "You are here" marker.

Modes: "Guided Narrative" (no sidebar; four states, one change per step: baseline Earth, 6 deg head-down tilt, Valsalva at the
jugular outlet with valve closure, Valsalva reaching both outlets) and "Clinical Sandbox" (every control, the NASA risk factors, the map).
Mode and step live in the URL (?mode=guided&step=3) so a refresh restores the place. The 3D view is Plotly Mesh3d: offline, no CDN.

Scientific status: exploratory and not validated. Upright diversion is a calibration target; the pressure-drop share across
the jugular/collateral pair is an assumption that sets the reversal threshold; regime thresholds are modelling assumptions;
the venous valve is a switch (competence varies between people).
"""

from __future__ import annotations

import html
import inspect
import json
import math
from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.interpolate import RegularGridInterpolator

st.set_page_config(page_title="IJV-Comply | Jugular vein testbed", layout="wide")

try:
    from ijv_math_engine import (
        PatientProfile, RegimeThresholds, ScenarioError, RESPIRATION_PHASES, apply_respiration, calibrate_network,
        effective_gravity_for_tilt, get_scenario_conditions, list_scenarios, nasa_protocol, reversal_threshold_mmHg,
        run_scenario, solve_steady_state,
    )
except ImportError as _exc:  # pragma: no cover - environment problem
    st.error(f"Could not import `ijv_math_engine`: {_exc}. Keep `ijv_math_engine.py` in the same folder as `app.py`.")
    st.stop()

# =====================================================================================
# Constants and look
# =====================================================================================
INK, SLATE, BLUE, TEAL, GRID = "#0E2438", "#5B6B7C", "#1F5FA8", "#0B7F86", "#E3E8EE"

BUTTON_LABELS = {
    "supine_1g": "Supine (1 g)",
    "upright_1g": "Upright (1 g)",
    "microgravity_0g": "Microgravity (0 g)",
    "microgravity_asymmetric_obstruction": "Microgravity + Brachiocephalic Obstruction",
}
FLIGHT_KEYS = {"microgravity_0g", "microgravity_asymmetric_obstruction"}

# Regime map categories and the thresholds that define them (MODELLING ASSUMPTIONS, not clinical cut-offs).
THR = RegimeThresholds()
# ONE colour language for every view. Bright set: dark surfaces, the 3D vein and the map. Darker set: white-text badges and the light 2D canvas.
STATE_COLORS = {"forward": "#14B8A6", "stagnant": "#F59E0B", "reversed": "#8B5CF6", "trapped": "#EF4444"}
STATE_ON_LIGHT = {"forward": "#0F8F82", "stagnant": "#B26A00", "reversed": "#6D45D6", "trapped": "#C62828"}
REGIMES = [("Reversed", STATE_COLORS["reversed"]), ("Stagnant", STATE_COLORS["stagnant"]), ("Diverted", "#8DB3DE"),
           ("Normal", STATE_COLORS["forward"]), ("Collapsed", "#3B4A5E"), ("Trapped (valve closed)", STATE_COLORS["trapped"])]
V_CUT = THR.v_stasis_cm_s                  # |mean velocity| below this is "stagnant"
DIVERSION_SHARE = THR.diversion_share      # IJV share below this is "diverted"
COLLAPSE_FRACTION = 0.5                    # share of vein length at P_tm < 0 that counts as "collapsed"
MAP_RES = {"Fast": (9, 9), "Standard": (13, 14), "Fine": (19, 20)}
X_RANGE, Y_RANGE = (-5.0, 10.0), (-5.0, 30.0)       # P_ext (x) and P_out_ijv (y), mmHg
FINE_NX, FINE_NY = 91, 76

DEFAULT = PatientProfile.population_placeholder()
K_OPTIONS = sorted(set(np.round(np.geomspace(0.1, 5.0, 41), 2).tolist() + [float(DEFAULT.K_mmHg)]))

st.markdown(
    f"""
    <style>
    /* no web-font import: the page must run fully offline; Barlow is used if installed, else Segoe UI / Arial */
    .stApp, .stMarkdown, h1, h2, h3, h4, label {{ font-family: 'Barlow', 'Segoe UI', Arial, sans-serif; }}
    .block-container {{ padding-top: 1.8rem; max-width: 1500px; }}
    h1 {{ font-weight: 700; letter-spacing: -0.01em; color: {INK}; margin-bottom: 0; }}
    .ijv-sub {{ font-size: 1.15rem; font-weight: 500; color: {BLUE}; margin: 0.1rem 0 0.7rem 0; }}
    .ijv-hook {{ font-size: 0.98rem; line-height: 1.55; max-width: 66rem; color: {INK}; }}
    .ijv-eyebrow {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
                    color: {SLATE}; margin: 1.3rem 0 0.35rem 0; border-bottom: 1px solid {GRID}; padding-bottom: 0.25rem; }}
    .ijv-status {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.6rem 0 0.2rem 0; }}
    .ijv-chip {{ font-size: 0.8rem; color: {INK}; background: #EEF2F7; border: 1px solid #D5DCE5; border-radius: 999px;
                 padding: 0.15rem 0.65rem; }}
    .ijv-chip b {{ color: {BLUE}; font-weight: 600; }}
    [data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid #D5DCE5; border-left: 5px solid {BLUE};
                               border-radius: 4px; padding: 0.55rem 0.85rem; }}
    [data-testid="stMetricLabel"] {{ color: {SLATE}; }}
    [data-testid="stMetricValue"] {{ font-weight: 600; color: {INK}; }}
    .ijv-flags {{ background: #FFFFFF; border: 1px solid #D5DCE5; border-left: 5px solid {TEAL}; border-radius: 4px;
                  padding: 0.5rem 0.85rem; min-height: 5.1rem; }}
    .ijv-flags .lbl {{ color: {SLATE}; font-size: 0.85rem; margin-bottom: 0.3rem; }}
    .ijv-badge {{ display: inline-block; color: #FFFFFF; font-weight: 600; font-size: 0.8rem; border-radius: 3px;
                  padding: 0.16rem 0.48rem; margin: 0 0.3rem 0.3rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def stretch(fn) -> dict:
    """Full-width keyword that works across Streamlit versions (`use_container_width` was replaced by `width`)."""
    params = inspect.signature(fn).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="ijv-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


# =====================================================================================
# Cached engine calls (keys are plain numbers/strings/tuples so Streamlit hashes them cheaply)
# =====================================================================================
OBS_KEY = "microgravity_asymmetric_obstruction"


def make_patient(pk) -> PatientProfile:
    """
    Step 1 of the engine workflow. `pk` = (A0, K, P_ext, Q_ref, pair-drop fraction, supine IJV share, slit lumen on/off,
    competent valve on/off): every value comes from the sidebar; nothing is hardcoded here.
    """
    A0, K, p_ext, q_ref, f_par, share, slit, valve = pk
    return PatientProfile(
        A0_cm2=float(A0), K_mmHg=float(K), p_ext_mmHg=float(p_ext), q_ref_ijv_mL_s=float(q_ref),
        parallel_drop_fraction=float(f_par), ijv_share_ref=float(share), lumen_model="slit" if slit else "circular",
        valve_competent=bool(valve),
        sources={"A0_cm2": "user input (pre-flight ultrasound)", "K_mmHg": "user input (tilt-table)",
                 "p_ext_mmHg": "user input (assumed)", "q_ref_ijv_mL_s": "user input (pre-flight Doppler)",
                 "parallel_drop_fraction": "model assumption", "ijv_share_ref": "literature (Doepp 2004, via Lan 2021)"},
    )


def scenario_inputs(key: str, ovr) -> dict:
    """Scenario inputs after the optional override (q_total_factor, p_out_ijv, p_out_collat) of the active scenario."""
    cond = get_scenario_conditions(key)
    for name, value in (ovr or {}).items():
        cond[name] = float(value)
    return cond


@st.cache_data(show_spinner=False, max_entries=64)
def cached_calibration(pk):
    """Step 2: calibrate_network(patient). Returns (NetworkCalibration, None) or (None, message) if infeasible."""
    try:
        return calibrate_network(make_patient(pk)), None
    except ScenarioError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_result(key: str, ovr_qf, ovr_pout_ijv, ovr_pout_c, pk, p_ext_run: float, phase: str, coupling: float,
                  shifts, tilt):
    """
    Step 3: run_scenario(...) for the active scenario: prescribed Q_total and both outlets in, split and P_in out.
    Respiration phase, its collateral coupling, the (inspiration fall, Valsalva rise) magnitudes and the head-down tilt are
    plain numbers, so a slider move is one cheap closed-form/1D solve, never a sweep.
    """
    network, _err = cached_calibration(pk)
    overrides = {k: v for k, v in (("q_total_factor", ovr_qf), ("p_out_ijv", ovr_pout_ijv), ("p_out_collat", ovr_pout_c))
                 if v is not None}
    table = {"Deep Inspiration": -float(shifts[0]), "Valsalva": float(shifts[1])}
    return run_scenario(key, make_patient(pk), network, overrides=overrides or None, p_ext=p_ext_run, respiration=phase,
                        collateral_coupling=float(coupling), respiration_table=table, tilt_deg=tilt)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_threshold_check(key: str, q_factor: float, p_out_c: float, pk, p_ext_run: float):
    """
    Live proof that the reversal boundary is exact. Runs the full solver at the predicted threshold and 0.2 mmHg either
    side of it, for three vein stiffnesses (0.3x, 1x, 3x the patient's K). Expected: Q_IJV = 0 at the threshold for every
    stiffness, forward below it and reversed above it.
    """
    network, _err = cached_calibration(pk)
    if network is None:
        return None
    base = replace(make_patient(pk), valve_competent=False)       # the check shows the mathematical threshold, valve off
    thr = reversal_threshold_mmHg(network, q_factor * network.q_total_ref)
    if not math.isfinite(thr):
        return None
    rows = []
    for kf in (0.3, 1.0, 3.0):
        pat = base.with_stiffness_factor(kf)
        qs = []
        for d in (-0.2, 0.0, 0.2):
            r = run_scenario(key, pat, network, p_ext=p_ext_run,
                             overrides={"q_total_factor": q_factor, "p_out_collat": p_out_c, "p_out_ijv": p_out_c + thr + d})
            qs.append(r.q_ijv)
        rows.append((pat.K_mmHg, qs))
    return {"threshold": thr, "rows": rows}


def classify_codes(v, frac, share, trapped=None):
    """
    Regime code per cell: 0 Reversed, 1 Stagnant, 2 Diverted, 3 Normal, 4 Collapsed, 5 Trapped (competent valve closed).
    Priority: trapped > reversed > stagnant > collapsed > diverted > normal ('diverted' needs forward flow below the share cut-off).
    """
    v, frac, share = (np.asarray(a, dtype=float) for a in (v, frac, share))
    code = np.full(v.shape, 3, dtype=np.int8)
    code[(share >= 0) & (share < DIVERSION_SHARE)] = 2
    code[frac >= COLLAPSE_FRACTION] = 4
    code[np.abs(v) < V_CUT] = 1
    code[v <= -V_CUT] = 0
    if trapped is not None:
        code[np.asarray(trapped, dtype=bool)] = 5
    return code


@st.cache_data(show_spinner=False, max_entries=32)
def cached_map(key: str, q_factor: float, p_out_c: float, pk, res: str):
    """
    Regime map for one scenario: the engine is solved on a coarse (P_ext x P_out_ijv) grid with the prescribed Q_total,
    the collateral outlet P_out_collat, g_eff and the calibrated network held fixed. Mean velocity, collapse fraction,
    IJV share and the SOLVED P_in are interpolated to a fine grid; regimes are classified there and P_in is drawn as
    contour lines. The reversal boundary itself is exact and drawn separately.
    """
    network, _err = cached_calibration(pk)
    if network is None:
        return None
    patient = make_patient(pk)
    nx, ny = MAP_RES[res]
    xs, ys = np.linspace(*X_RANGE, nx), np.linspace(*Y_RANGE, ny)
    cond = dict(get_scenario_conditions(key))
    cond["q_total_factor"], cond["p_out_collat"] = float(q_factor), float(p_out_c)
    fields = {name: np.zeros((ny, nx)) for name in ("v", "frac", "share", "p_in")}
    failures, fallbacks = 0, 0
    for j, p_out in enumerate(ys):
        c = dict(cond)
        c["p_out_ijv"] = float(p_out)
        for i, pe in enumerate(xs):
            try:
                r = solve_steady_state(c, patient, network, p_ext=float(pe), n_pts=25, report_unconstrained=False)
                fields["v"][j, i], fields["frac"][j, i] = r.velocity_mean_cm_s, r.frac_collapsed
                fields["share"][j, i], fields["p_in"][j, i] = r.ijv_share, r.p_in
                fallbacks += r.method_used == "0d"
            except Exception:           # a failed node is counted and shown as open rather than crashing the page
                failures += 1
                fields["share"][j, i], fields["p_in"][j, i] = 1.0, float(p_out)
    xf, yf = np.linspace(*X_RANGE, FINE_NX), np.linspace(*Y_RANGE, FINE_NY)
    pts = np.array([(y, x) for y in yf for x in xf])
    fine = {k: RegularGridInterpolator((ys, xs), f)(pts).reshape(FINE_NY, FINE_NX) for k, f in fields.items()}
    # With a competent valve the trapped region is EXACT (P_out,IJV above P_out,collat + Q_total R_collat), not interpolated.
    thr_line = p_out_c + reversal_threshold_mmHg(network, q_factor * network.q_total_ref)
    trap = np.repeat((yf > thr_line)[:, None], FINE_NX, axis=1) if (patient.valve_competent and math.isfinite(thr_line)) else None
    codes = classify_codes(fine["v"], fine["frac"], fine["share"], trap)
    # Hover text is built ONCE here (it is the slowest part of drawing the background), so moving a slider only rebuilds
    # the target marker, not the background.
    hover = np.char.add(np.char.add(np.array([r[0] for r in REGIMES])[codes], "<br>"), np.char.mod("P_in %.1f mmHg (solved)", fine["p_in"]))
    hover = np.char.add(np.char.add(hover, "<br>"), np.char.mod("IJV share %.0f%%", 100 * fine["share"]))
    hover = np.char.add(np.char.add(hover, "<br>"), np.char.mod("mean velocity %.1f cm/s", fine["v"]))
    return {"x": xf, "y": yf, "codes": codes, "hover": hover, "v": fine["v"], "share": fine["share"], "p_in": fine["p_in"],
            "shares": [float(np.mean(codes == k)) for k in range(len(REGIMES))], "v_min": float(fields["v"].min()),
            "v_max": float(fields["v"].max()), "failures": failures, "fallbacks": fallbacks, "n_solves": nx * ny}


# =====================================================================================
# Visuals
# =====================================================================================
FLAG_COLORS = [
    ("valve closed", STATE_ON_LIGHT["trapped"]), ("critical low shear", "#C62828"), ("low shear", "#D9822B"), ("severe collapse", "#44307A"),
    ("partial collapse", "#1F5FA8"), ("diverted", "#4F7CAC"), ("flow-limitation", "#44307A"), ("reversed", STATE_ON_LIGHT["reversed"]),
    ("stagnant", STATE_ON_LIGHT["stagnant"]), ("forward", STATE_ON_LIGHT["forward"]),
]


def badges_html(flags) -> str:
    out = []
    for flag in flags:
        color = next((c for needle, c in FLAG_COLORS if needle in flag.lower()), SLATE)
        out.append(f'<span class="ijv-badge" style="background:{color}">{html.escape(flag)}</span>')
    return "".join(out)


FLOW_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent; }
  canvas { display: block; box-sizing: border-box; border: 1px solid #D5DCE5; border-radius: 6px; background: #F8FAFC; }
</style></head>
<body><div id="wrap"><canvas id="c" aria-label="Animated blood flow through the jugular vein"></canvas></div>
<script>
(function () {
  "use strict";
  var D = __DATA__;
  var cv = document.getElementById("c"), ctx = cv.getContext("2d");
  var W = 900, H = 300, MX = 74, parts = [], last = null;

  // Linear interpolation of a profile array (uniform in x, inlet -> outlet) at position xcm.
  function lerp(arr, xcm) {
    var n = arr.length, f = (xcm / D.L) * (n - 1);
    if (f <= 0) return arr[0];
    if (f >= n - 1) return arr[n - 1];
    var i = Math.floor(f), t = f - i;
    return arr[i] * (1 - t) + arr[i + 1] * t;
  }
  // One scale for both axes: the 15 cm vein spans the canvas, so the lumen is drawn at true proportion.
  function ppc() { return (W - 2 * MX) / D.L; }
  function radius(area) { return Math.max(3, Math.sqrt(area / Math.PI) * ppc()); }

  function fit() {
    var w = (cv.parentElement && cv.parentElement.clientWidth) || 900;
    var dpr = window.devicePixelRatio || 1;
    W = Math.max(480, w);
    cv.style.width = W + "px"; cv.style.height = H + "px";
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Sample a position along the vein with probability proportional to the local area. In steady flow blood cells are
  // uniformly concentrated, so number density per unit length follows A(x); with that distribution the average particle
  // velocity is exactly Q / mean(A), the engine's reported mean velocity.
  function sampleX() {
    var n = D.area.length, cum = [0], i;
    for (i = 1; i < n; i++) cum.push(cum[i - 1] + 0.5 * (D.area[i] + D.area[i - 1]));
    var u = Math.random() * cum[n - 1];
    for (i = 1; i < n - 1 && cum[i] < u; i++) {}
    var seg = cum[i] - cum[i - 1], t = seg > 0 ? (u - cum[i - 1]) / seg : 0;
    return ((i - 1 + t) / (n - 1)) * D.L;
  }

  function init() {
    parts = [];
    for (var i = 0; i < D.nParticles; i++) {
      var s = Math.sqrt(Math.random());                 // radial position, uniform over the circular cross-section
      parts.push({
        x: sampleX(), s: s, th: Math.random() * 2 * Math.PI,
        spd: 2 * (1 - s * s),                           // Poiseuille profile: fast at the centre, slow at the wall (mean = 1)
        r: 1.7 + Math.random() * 1.4,
        col: "hsl(" + (352 + Math.random() * 14) + ",72%," + (36 + Math.random() * 12) + "%)",
        ox: 0, oy: 0,                                    // Brownian offsets (cm) from the advected position: zero-mean, mean-reverting
        ph: Math.random() * 2 * Math.PI, om: (1.5 + Math.random() * 2) * (Math.random() < 0.5 ? -1 : 1)     // slow swirl in place
      });
    }
  }

  // Standard normal variate (Box-Muller).
  function gauss() { var u = 1 - Math.random(), v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  // Jitter amplitude: 1 when blood is trapped (|v| = 0), falling linearly to 0 at the stagnation cut-off and exactly 0 above it, so
  // flowing blood moves in clean lines and slow or trapped blood hovers. No pop at the threshold.
  function jitterAmp() { return Math.max(0, Math.min(1, 1 - Math.abs(D.vMean) / D.vCut)); }
  var T = 0;                                           // animation clock (s)

  // Net transport: each particle advances by its LOCAL velocity v(x) = Q/A(x) (signed), so the stream speeds up in narrow sections and
  // reverses when the mean velocity is negative. When Q_IJV = 0 every local velocity is 0, so net transport is exactly zero.
  // Trapped or slow blood additionally jitters: a zero-mean Ornstein-Uhlenbeck offset (it can never drift away) plus a small swirl.
  // The jitter is ILLUSTRATIVE (exaggerated); the model predicts no eddies, only the absence of net flow.
  function update(dt) {
    var amp = jitterAmp(), a = Math.exp(-D.jitRate * dt), sd = Math.sqrt(1 - a * a);
    T += dt;
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      p.x += lerp(D.vel, p.x) * p.spd * D.timeScale * dt;
      if (p.x < 0 || p.x >= D.L) p.x = ((p.x % D.L) + D.L) % D.L;      // wrap only when leaving the tube, so zero velocity means bit-for-bit zero motion
      if (amp > 0) { p.ox = a * p.ox + amp * D.jitAx * sd * gauss(); p.oy = a * p.oy + amp * D.jitTr * sd * gauss(); }
      else { p.ox = 0; p.oy = 0; }
    }
  }
  // Displayed offset of a particle from its advected position (cm): Brownian part plus swirl. Exactly (0, 0) when blood is flowing.
  function offsetOf(p) {
    var amp = jitterAmp();
    return [p.ox + amp * D.swirl * Math.cos(p.ph + p.om * T), p.oy + amp * D.swirl * Math.sin(p.ph + p.om * T)];
  }

  function draw() {
    var cy = 150, n = D.area.length, k = ppc();
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#F8FAFC"; ctx.fillRect(0, 0, W, H);

    // dashed outline = reference lumen (A0) so collapse or engorgement is visible at a glance
    var rRef = radius(D.A0);
    ctx.setLineDash([6, 5]); ctx.strokeStyle = "#8A94A0"; ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(MX, cy - rRef); ctx.lineTo(W - MX, cy - rRef);
    ctx.moveTo(MX, cy + rRef); ctx.lineTo(W - MX, cy + rRef);
    ctx.stroke(); ctx.setLineDash([]);

    // lumen: height follows the area profile from the 1D solver (its mean is the reported mean area)
    var i, x, r;
    ctx.beginPath();
    for (i = 0; i < n; i++) { x = MX + (i / (n - 1)) * (W - 2 * MX); r = radius(D.area[i]); if (i === 0) ctx.moveTo(x, cy - r); else ctx.lineTo(x, cy - r); }
    for (i = n - 1; i >= 0; i--) { x = MX + (i / (n - 1)) * (W - 2 * MX); r = radius(D.area[i]); ctx.lineTo(x, cy + r); }
    ctx.closePath();
    ctx.fillStyle = "#FBE3E0"; ctx.fill();
    ctx.strokeStyle = "#1F5FA8"; ctx.lineWidth = 3; ctx.stroke();

    // blood particles
    for (i = 0; i < parts.length; i++) {
      var p = parts[i], off = offsetOf(p), xd = Math.min(D.L, Math.max(0, p.x + off[0])), rr = radius(lerp(D.area, xd));
      var yd = p.s * Math.cos(p.th) * rr * 0.86 + off[1] * k;
      yd = Math.max(-0.92 * rr, Math.min(0.92 * rr, yd));          // jitter never leaves the lumen
      ctx.fillStyle = p.col;
      ctx.beginPath();
      ctx.arc(MX + (xd / D.L) * (W - 2 * MX), cy + yd, p.r, 0, 2 * Math.PI);
      ctx.fill();
    }

    // labels
    var dirCol = D.vMean >= D.vCut ? "#0F8F82" : (D.vMean <= -D.vCut ? "#6D45D6" : "#B26A00");
    var chev = Math.min(4, Math.max(1, Math.ceil(Math.abs(D.vMean) / 4)));
    var arrows = ""; for (i = 0; i < chev; i++) arrows += (D.vMean < 0 ? "\\u25C0" : "\\u25B6");
    var word = D.vMean >= D.vCut ? "forward" : (D.vMean <= -D.vCut ? "reversed" : "stagnant");
    if (D.frozen) { dirCol = "#C62828"; arrows = "\\u25A0"; word = "valve closed: blood trapped, no net flow"; }

    // venous valve at the outlet end: closed = solid red bar across the lumen, open = green leaflets pushed to the wall
    if (D.valve !== "none") {
      var xv = W - MX, rv = radius(D.area[n - 1]);
      ctx.lineWidth = 4; ctx.lineCap = "round";
      if (D.valve === "closed") { ctx.strokeStyle = "#C62828"; ctx.beginPath(); ctx.moveTo(xv, cy - rv - 6); ctx.lineTo(xv, cy + rv + 6); ctx.stroke(); }
      else { ctx.strokeStyle = "#0F8F82"; ctx.beginPath(); ctx.moveTo(xv, cy - rv - 6); ctx.lineTo(xv - 9, cy - rv * 0.55); ctx.moveTo(xv, cy + rv + 6); ctx.lineTo(xv - 9, cy + rv * 0.55); ctx.stroke(); }
      ctx.lineCap = "butt"; ctx.textAlign = "center"; ctx.font = "600 12px Segoe UI, Arial, sans-serif";
      ctx.fillStyle = D.valve === "closed" ? "#C62828" : "#0F8F82";
      ctx.fillText(D.valve === "closed" ? "valve closed" : "valve open", xv, cy - rv - 12);
    }
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#0E2438"; ctx.font = "600 16px Segoe UI, Arial, sans-serif"; ctx.textAlign = "left";
    ctx.fillText(D.title, MX, 28);
    ctx.fillStyle = "#5B6B7C"; ctx.font = "13px Segoe UI, Arial, sans-serif";
    ctx.fillText(D.sub, MX, 48);
    ctx.fillStyle = dirCol; ctx.font = "600 15px Segoe UI, Arial, sans-serif"; ctx.textAlign = "right";
    ctx.fillText(arrows + "  v\\u0304 = " + D.vMean.toFixed(2) + " cm/s (" + word + ")", W - MX, 28);
    ctx.fillStyle = "#5B6B7C"; ctx.font = "13px Segoe UI, Arial, sans-serif";
    ctx.fillText("Q_IJV = " + D.q.toFixed(2) + " mL/s    mean area = " + D.areaMean.toFixed(2) + " cm\\u00B2 (" + Math.round(100 * D.areaMean / D.A0) + "% of A\\u2080)", W - MX, 48);

    ctx.textAlign = "left"; ctx.fillText("Inlet (skull base)", MX, H - 14);
    ctx.textAlign = "right"; ctx.fillText("Outlet (thorax)", W - MX, H - 14);
    ctx.textAlign = "center"; ctx.fillText("dashed = reference lumen (A\\u2080)", W / 2, H - 14);
    if (jitterAmp() > 0) { ctx.fillStyle = dirCol; ctx.fillText("slow or trapped blood: random jitter only, no net transport (motion exaggerated)", W / 2, H - 32); ctx.fillStyle = "#5B6B7C"; }
    ctx.strokeStyle = "#0E2438"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(MX, H - 38); ctx.lineTo(MX + k, H - 38); ctx.stroke();
    ctx.textAlign = "left"; ctx.fillStyle = "#0E2438"; ctx.fillText("1 cm", MX + k + 8, H - 34);
  }

  function frame(ts) {
    if (last === null) last = ts;
    var dt = Math.min(0.05, Math.max(0, (ts - last) / 1000)); last = ts;
    update(dt); draw();
    window.requestAnimationFrame(frame);
  }

  fit(); init();
  window.addEventListener("resize", fit);
  window.__ijv = { particles: parts, update: update, draw: draw, data: D, offsetOf: offsetOf, jitterAmp: jitterAmp };   // test hook, harmless in production
  window.requestAnimationFrame(frame);
})();
</script></body></html>
"""


def flow_animation_html(res, patient: PatientProfile, title: str, sub: str, time_scale: float) -> str:
    """Inject the engine result into the animation. Tube height <- area profile; particle speed/direction <- velocity."""
    n = 61
    xs = np.linspace(float(res.x_cm[0]), float(res.x_cm[-1]), n)
    data = {
        "title": title, "sub": sub, "L": float(patient.L_cm), "A0": float(patient.A0_cm2),
        "area": [round(float(a), 5) for a in np.interp(xs, res.x_cm, res.area_cm2)],
        "vel": [round(float(v), 4) for v in np.interp(xs, res.x_cm, res.velocity_cm_s)],
        "vMean": float(res.velocity_mean_cm_s), "vCut": float(V_CUT), "areaMean": float(res.area_mean_cm2),
        "q": float(res.q_ijv), "timeScale": float(time_scale), "nParticles": 170,
        # illustrative Brownian jitter of trapped / slow blood: stationary sd (cm), relaxation rate (1/s), swirl radius (cm)
        "jitAx": 0.12, "jitTr": 0.07, "jitRate": 3.0, "swirl": 0.05,
        # speed is bound to Q_IJV through v = Q/A; when Q_IJV is exactly 0 the animation is frozen outright
        "frozen": bool(res.valve_closed or res.q_ijv == 0.0),
        "valve": "closed" if res.valve_closed else ("open" if patient.valve_competent else "none"),
    }
    return FLOW_HTML.replace("__DATA__", json.dumps(data))


def build_map_figure(m, here_x: float, here_y: float, title: str, threshold_y: float) -> go.Figure:
    """Regime heatmap + exact reversal threshold + solved-P_in contour lines + legend + 'You are here' marker."""
    k = len(REGIMES)
    scale = []
    for i, (_name, color) in enumerate(REGIMES):
        scale += [[i / k, color], [(i + 1) / k, color]]
    hover = m["hover"]                                   # prebuilt and cached with the map
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=m["x"], y=m["y"], z=m["codes"], zmin=-0.5, zmax=k - 0.5, colorscale=scale, showscale=False, text=hover,
        hovertemplate="P_ext %{x:.1f} mmHg | P_out,IJV %{y:.1f} mmHg<br>%{text}<extra></extra>"))
    fig.add_trace(go.Contour(
        x=m["x"], y=m["y"], z=m["p_in"], ncontours=10, showscale=False, hoverinfo="skip",
        colorscale=[[0, "#FFFFFF"], [1, "#FFFFFF"]], line=dict(width=1.3),
        contours=dict(coloring="lines", showlabels=True, labelfont=dict(size=11, color="#FFFFFF")),
        name="Solved P_in (mmHg)", showlegend=True))
    if Y_RANGE[0] <= threshold_y <= Y_RANGE[1]:
        # Exact, not interpolated: at zero jugular flow the collateral carries everything, so the boundary is
        # P_out,IJV = P_out,collat + Q_total x R_collat for every tissue pressure.
        fig.add_trace(go.Scatter(
            x=list(X_RANGE), y=[threshold_y, threshold_y], mode="lines", name="Exact reversal threshold",
            line=dict(color=INK, width=2.5, dash="dash"),
            hovertemplate=f"Reversal threshold: P_out,IJV = {threshold_y:.2f} mmHg<extra></extra>"))
    for name, color in REGIMES:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=14, color=color, symbol="square")))
    off_scale = here_y > Y_RANGE[1] or here_y < Y_RANGE[0]
    y_plot = min(max(here_y, Y_RANGE[0]), Y_RANGE[1])
    label = f"You are here (P_out,IJV {here_y:.1f}: off scale)" if off_scale else "You are here"
    fig.add_trace(go.Scatter(
        x=[here_x], y=[y_plot], mode="markers+text", name="You are here", text=[label],
        textposition="bottom center" if y_plot > Y_RANGE[1] - 2 else "top center",
        textfont=dict(size=14, color=INK), cliponaxis=False,
        marker=dict(symbol="star", size=26, color="#FFFFFF", line=dict(color="#000000", width=2.5)),
        hovertemplate=f"Current state<br>P_ext {here_x:.1f} mmHg<br>effective P_out,IJV {here_y:.1f} mmHg<extra></extra>"))
    axis = dict(showline=True, linecolor=INK, linewidth=1.5, gridcolor=GRID, zeroline=False, ticks="outside")
    fig.update_xaxes(range=list(X_RANGE), title_text="Tissue pressure, P_ext (mmHg)", **axis)
    fig.update_yaxes(range=list(Y_RANGE), title_text="Jugular outlet pressure, P_out,IJV (mmHg)", **axis)
    fig.update_layout(
        template="plotly_white", height=600, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=80, r=210, t=70, b=70),
        font=dict(family="Barlow, Segoe UI, Arial, sans-serif", size=14, color=INK),
        title=dict(text=title, x=0.0, xanchor="left", y=0.97, yanchor="top", font=dict(size=16, color=INK)),
        legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.95)",
                    bordercolor=GRID, borderwidth=1, font=dict(size=13)))
    return fig


def show_plotly(fig: go.Figure, filename: str, key: str = None) -> None:
    config = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
              "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 3}}
    kw = stretch(st.plotly_chart)
    if key is not None and "key" in inspect.signature(st.plotly_chart).parameters:
        kw["key"] = key                      # stable element identity so the 3D camera is not rebuilt on every rerun
    st.plotly_chart(fig, theme=None, config=config, **kw)


# =====================================================================================
# One state vocabulary for every view (cards, badges, 2D animation, 3D vein, map)
# =====================================================================================
DARK_BG, NEUTRAL_LINE, GHOST_LINE = "#08121C", "#7FA3C4", "#4A5E72"
VENOUS_WSS_PA = (0.1, 0.6)        # typical venous wall shear stress, Pa (arterial low-shear threshold 0.4 Pa is NOT applied to veins)
RAMPS = {"forward": [[0.0, "#0E6B63"], [1.0, "#6EF0DE"]], "stagnant": [[0.0, "#9A5F0A"], [1.0, "#FFD27A"]],
         "reversed": [[0.0, "#4B2FA3"], [1.0, "#C4B0FF"]]}
CAMERAS = {"Oblique": dict(eye=dict(x=1.55, y=-1.45, z=0.45), center=dict(x=0.0, y=0.0, z=-0.08), up=dict(x=0, y=0, z=1)),
           "Front": dict(eye=dict(x=0.0, y=-2.3, z=0.25), center=dict(x=0.0, y=0.0, z=-0.08), up=dict(x=0, y=0, z=1)),
           "Side": dict(eye=dict(x=2.3, y=0.0, z=0.25), center=dict(x=0.0, y=0.0, z=-0.08), up=dict(x=0, y=0, z=1))}


def flow_state(res) -> str:
    """trapped (valve closed / Q_IJV exactly 0) > stagnant (|mean v| below the cutoff) > reversed > forward."""
    if res.valve_closed or res.q_ijv == 0.0:
        return "trapped"
    if abs(res.velocity_mean_cm_s) < V_CUT:
        return "stagnant"
    return "reversed" if res.q_ijv < 0 else "forward"


# =====================================================================================
# 3D anchor: schematic cranial venous network as Plotly Mesh3d + Scatter3d (bundled with Streamlit: works offline)
# =====================================================================================
def tube_mesh(center, radius, n_theta=20):
    """Closed tube around a polyline: vertices (n*n_theta, 3) and triangles (m, 3)."""
    n = len(center)
    t = np.gradient(center, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ref = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    ref[np.abs(t @ np.array([0.0, 1.0, 0.0])) > 0.9] = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(t, u)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ring = center[:, None, :] + radius[:, None, None] * (np.cos(th)[None, :, None] * u[:, None, :] + np.sin(th)[None, :, None] * v[:, None, :])
    a = np.arange(n - 1)[:, None] * n_theta + np.arange(n_theta)[None, :]
    b = np.arange(n - 1)[:, None] * n_theta + (np.arange(n_theta)[None, :] + 1) % n_theta
    tri = np.concatenate([np.stack([a, b, a + n_theta], -1).reshape(-1, 3), np.stack([b, b + n_theta, a + n_theta], -1).reshape(-1, 3)])
    return ring.reshape(-1, 3), tri


def _lines(segs, color, width, dash=None):
    xs, ys, zs = [], [], []
    for s in segs:
        for p in s:
            xs.append(float(p[0])); ys.append(float(p[1])); zs.append(float(p[2]))
        xs.append(None); ys.append(None); zs.append(None)          # gap between segments
    line = dict(color=color, width=width)
    if dash:
        line["dash"] = dash
    return ("Scatter3d", dict(x=xs, y=ys, z=zs, mode="lines", line=line, hoverinfo="skip", showlegend=False))


def _label(p, text, color, size=12, pos="top center", marker=None):
    kw = dict(x=[float(p[0])], y=[float(p[1])], z=[float(p[2])], mode="markers+text" if marker else "text", text=[text],
              textposition=pos, textfont=dict(size=size, color=color), hoverinfo="skip", showlegend=False)
    if marker:
        kw["marker"] = marker
    return ("Scatter3d", kw)


def _head_outline(c=(0.0, 0.0, 7.5), ax=(7.0, 8.6, 7.8), n_lat=5, n_lon=8, m=48):
    """Skull cap as line art (no translucent surface: WebGL transparency sorts badly)."""
    segs, t = [], np.linspace(0, 2 * np.pi, m)
    for phi in np.linspace(-0.55 * np.pi / 2, 0.85 * np.pi / 2, n_lat):
        segs.append(np.stack([c[0] + ax[0] * np.cos(phi) * np.cos(t), c[1] + ax[1] * np.cos(phi) * np.sin(t), np.full(m, c[2] + ax[2] * np.sin(phi))], 1))
    s = np.linspace(-0.55 * np.pi / 2, np.pi / 2, m)
    for lam in np.linspace(0, np.pi, n_lon, endpoint=False):
        for sign in (1, -1):
            segs.append(np.stack([c[0] + sign * ax[0] * np.cos(s) * np.cos(lam), c[1] + sign * ax[1] * np.cos(s) * np.sin(lam), c[2] + ax[2] * np.sin(s)], 1))
    return segs


# Flow glyphs. They must sit ON the vein surface: anything drawn on the centreline is hidden inside the opaque Mesh3d tube by WebGL
# depth testing. Three sides, 120 degrees apart, one facing the preset camera, so at least one is within 60 degrees of any view.
GLYPH_LEVELS = (("#0A1F2E", 5), ("#7FA3C4", 6), ("#FFFFFF", 7))      # comet tail -> head (colour, line width): brightness ramp reads as direction
N_COMETS, N_SIDES, N_COMET_PTS = 5, 3, 10


def _surface_points(center, radius, s_frac, az, lift: float = 1.10, pad: float = 0.05):
    """Points on the vein surface at fractional position s (0 skull base .. 1 outlet) on the side facing azimuth az (radians)."""
    idx = np.arange(len(center))
    f = np.clip(np.asarray(s_frac, dtype=float), 0.0, 1.0) * (len(center) - 1)
    c = np.stack([np.interp(f, idx, center[:, k]) for k in range(3)], -1)
    rr = np.interp(f, idx, radius) * lift + pad
    return c + np.stack([np.cos(az) * rr, np.sin(az) * rr, np.zeros_like(rr)], -1)


def _gaps(polylines):
    xs, ys, zs = [], [], []
    for pl in polylines:
        for q in pl:
            xs.append(round(float(q[0]), 2)); ys.append(round(float(q[1]), 2)); zs.append(round(float(q[2]), 2))
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def flow_glyphs(center, radius, state: str, vmean: float, camera: str = "Oblique", phase: float = 0.0, n: int = N_COMETS, sides: int = N_SIDES):
    """
    Static-or-animated flow cue as a fixed list of Scatter3d traces.
    forward / reversed: comet tails with a dark -> white brightness ramp and a V head, brighter end = where the blood is going; tail length grows
    with speed. trapped: white crosses on the surface (jittered when phase > 0, zero-mean). stagnant: small amber dots (no direction).
    phase in [0, 1/n) slides the comets by up to one spacing, so a cycle of phases loops seamlessly.
    """
    az0 = math.atan2(CAMERAS[camera]["eye"]["y"], CAMERAS[camera]["eye"]["x"])
    azs = [az0 + 2 * math.pi * j / sides for j in range(sides)]
    if state in ("trapped", "stagnant"):
        rng = np.random.default_rng(1000 + int(round(phase * 1e5)))
        pts = np.concatenate([_surface_points(center, radius, (np.arange(8) + 0.5) / 8, az) for az in azs])
        if state == "trapped" and phase > 0:
            pts = pts + rng.normal(0.0, 0.06, pts.shape)                    # illustrative Brownian jitter, zero-mean
        marker = dict(symbol="x", size=6, color="#FFFFFF") if state == "trapped" else dict(symbol="circle", size=3, color=STATE_COLORS["stagnant"])
        return [("Scatter3d", dict(x=np.round(pts[:, 0], 2), y=np.round(pts[:, 1], 2), z=np.round(pts[:, 2], 2), mode="markers", marker=marker,
                                    hoverinfo="skip", showlegend=False))]
    d = 1.0 if vmean > 0 else -1.0
    tail = float(np.clip(0.035 + 0.012 * abs(vmean), 0.04, 0.14))
    t = np.linspace(0.0, 1.0, N_COMET_PTS)
    levels = [[], [], []]
    heads = []
    for az in azs:
        side = np.array([math.sin(az), -math.cos(az), 0.0])                  # horizontal, tangent to the surface
        for k in range(n):
            s_head = ((k + 0.5) / n + d * phase) % 1.0          # half-spacing offset: no comet sits exactly on the clipped ends
            pts = _surface_points(center, radius, s_head - d * tail * (1.0 - t), az)
            for j, (lo, hi) in enumerate(((0, 4), (3, 7), (6, 10))):
                levels[j].append(pts[lo:hi])
            flow_dir = np.array([0.0, 0.0, -d])                              # s increases toward the outlet, i.e. z decreases
            head = pts[-1]
            heads.append(np.stack([head - flow_dir * 0.45 + side * 0.33, head, head - flow_dir * 0.45 - side * 0.33]))
    out = []
    for (col, wid), pls in zip(GLYPH_LEVELS, levels):
        xs, ys, zs = _gaps(pls)
        out.append(("Scatter3d", dict(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=wid), hoverinfo="skip", showlegend=False)))
    xs, ys, zs = _gaps(heads)
    out.append(("Scatter3d", dict(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#FFFFFF", width=GLYPH_LEVELS[-1][1]), hoverinfo="skip", showlegend=False)))
    return out


def glyph_frames(args: dict, seconds: float = 5.0, cycle_frames: int = 8, max_frames: int = 60):
    """
    Frames for the OPTIONAL animation: only the glyph traces change, the mesh is untouched. One cycle moves the comets by exactly one spacing (seamless);
    cycles repeat to fill `seconds`. Plotly cannot loop forever without custom JavaScript, so the run is bounded and the Play button restarts it.
    Returns (list of trace-spec lists, frame duration in ms).
    """
    state, vmean = args["state"], float(args["vmean"])
    if state == "stagnant":
        return [], 0
    if state == "trapped":
        dur, nf = 90, min(max_frames, int(seconds * 1000 / 90))
        return [flow_glyphs(args["center"], args["radius"], state, vmean, args["camera"], phase=(i % 8 + 1) / 1000.0 + i * 1e-6) for i in range(nf)], dur
    spacing_cm = float(abs(args["center"][-1, 2] - args["center"][0, 2])) / N_COMETS
    cycle_s = 2.0 * spacing_cm / max(abs(vmean), 0.5)                       # 2x slow motion, so fast flow still reads
    dur = int(np.clip(1000.0 * cycle_s / cycle_frames, 40, 250))
    nf = min(max_frames, int(math.ceil(seconds * 1000.0 / dur)))
    frames = [flow_glyphs(args["center"], args["radius"], state, vmean, args["camera"], phase=((i % cycle_frames) / cycle_frames) / N_COMETS) for i in range(nf)]
    return frames, dur


def network_scene(res, patient, q_total_ref: float, camera: str = "Oblique", exag: float = 1.6, n_st: int = 41, n_th: int = 20):
    """
    Plain description of the 3D scene: ([(trace type, kwargs), ...], layout info). Jugular radius follows the solved 1D area profile
    (x exag); colour follows the flow state (solid red when Q_IJV is exactly 0); collateral radius follows its flow.
    A schematic, not patient anatomy; the contralateral side is drawn as a ghost because it is not simulated.
    """
    state = flow_state(res)
    x_cm = np.asarray(res.x_cm, dtype=float)
    xs = np.linspace(x_cm[0], x_cm[-1], n_st)
    z = -xs
    area = np.interp(xs, x_cm, np.asarray(res.area_cm2))
    vel = np.interp(xs, x_cm, np.asarray(res.velocity_cm_s))
    r = np.sqrt(area / math.pi) * exag
    center = np.stack([2.6 + 0.35 * np.sin(z / 5.0), 0.25 * np.cos(z / 7.0), z], 1)
    V, T = tube_mesh(center, r, n_th)
    mesh = dict(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=T[:, 0], j=T[:, 1], k=T[:, 2], flatshading=False, showscale=False, hoverinfo="skip",
                name="Jugular vein", lighting=dict(ambient=0.5, diffuse=0.85, specular=0.35, roughness=0.45, fresnel=0.2),
                lightposition=dict(x=80, y=-60, z=100))
    if state == "trapped":
        mesh["color"] = STATE_COLORS["trapped"]                                  # the anchor: Q_IJV == 0 -> red
    else:
        mesh.update(intensity=np.repeat(np.abs(vel), n_th), intensitymode="vertex", colorscale=RAMPS[state], cmin=0.0,
                    cmax=max(float(np.abs(vel).max()), 1.0) * 1.15)
    L = float(patient.L_cm)
    tor = np.array([0.0, -6.5, 4.2])
    sss = np.stack([np.zeros(30), np.linspace(8.5, -6.5, 30), 4.2 + 9.0 * np.sin(np.linspace(0.05, 0.95, 30) * np.pi)], 1)
    ts_r = np.stack([np.linspace(0.0, 2.6, 16), np.linspace(-6.5, 0.25, 16), np.linspace(4.2, 0.0, 16)], 1)
    specs = [_lines(_head_outline(), GHOST_LINE, 2), _lines([sss], NEUTRAL_LINE, 6),
             _lines([ts_r], NEUTRAL_LINE if state == "trapped" else STATE_COLORS["forward"], 6),
             _lines([ts_r * np.array([-1.0, 1.0, 1.0])], GHOST_LINE, 5, dash="dot"), ("Mesh3d", mesh)]
    zl = np.linspace(0, -L, 12)
    specs.append(_lines([np.stack([-2.6 - 0.35 * np.sin(zl / 5.0), 0.25 * np.cos(zl / 7.0), zl], 1)], GHOST_LINE, 5, dash="dot"))
    zc = np.linspace(0.5, -L, 30)
    centc = np.stack([np.zeros(30), np.full(30, -2.6), zc], 1)
    rc = float(np.clip(0.13 * math.sqrt(max(abs(res.q_collateral), 1e-9) / max(0.34 * q_total_ref, 1e-9)), 0.06, 0.42))
    Vc, Tc = tube_mesh(centc, np.full(30, rc), 12)
    specs.append(("Mesh3d", dict(x=Vc[:, 0], y=Vc[:, 1], z=Vc[:, 2], i=Tc[:, 0], j=Tc[:, 1], k=Tc[:, 2], flatshading=False, showscale=False,
                                  color=STATE_COLORS["reversed"] if res.q_collateral < 0 else STATE_COLORS["forward"], hoverinfo="skip",
                                  name="Collateral", lighting=dict(ambient=0.5, diffuse=0.8, specular=0.25, roughness=0.5),
                                  lightposition=dict(x=80, y=-60, z=100))))
    glyph_start = len(specs)
    glyphs = flow_glyphs(center, r, state, float(res.velocity_mean_cm_s), camera)
    specs += glyphs
    if patient.valve_competent:
        vc = STATE_COLORS["trapped"] if res.valve_closed else STATE_COLORS["forward"]
        specs.append(_label(center[-4] + np.array([1.7, 0.0, 0.0]), "valve closed" if res.valve_closed else "valve open", vc, 11, "middle right",
                            marker=dict(symbol="diamond" if res.valve_closed else "diamond-open", size=9, color=vc, line=dict(color="#FFFFFF", width=1))))
    pin = dict(symbol="circle", size=7, color="#E6EEF5")
    specs += [_label(tor + np.array([0.0, 0.0, 1.2]), f"cranial venous pressure P_in {res.p_in:.1f} mmHg", "#E6EEF5", 12, "top center", marker=pin),
              _label(center[-1] + np.array([0.0, 0.0, -1.2]), f"jugular outlet {res.p_out_ijv:.1f} mmHg", "#E6EEF5", 12, "bottom right", marker=pin),
              _label(np.array([0.0, -2.6, -L - 1.2]), f"collateral outlet {res.p_out_collat:.1f} mmHg", "#E6EEF5", 12, "bottom left", marker=pin),
              _label(np.array([-2.6, 0.0, -4.5]), "contralateral side: not simulated", "#8FA6BA", 10, "middle left"),
              _label(np.array([0.0, -2.6, -10.0]), "vertebral collateral", "#8FA6BA", 10, "middle left")]
    banner = {"trapped": f"TRAPPED  Q_IJV = {res.q_ijv:.2f} mL/s", "reversed": f"REVERSED  Q_IJV = {res.q_ijv:+.2f} mL/s",
              "stagnant": f"STAGNANT  Q_IJV = {res.q_ijv:+.2f} mL/s", "forward": f"FORWARD  Q_IJV = {res.q_ijv:+.2f} mL/s"}[state]
    return specs, dict(state=state, camera=CAMERAS[camera], banner=banner, exag=exag, glyph_start=glyph_start, glyph_n=len(glyphs),
                       glyph_args=dict(center=center, radius=r, state=state, vmean=float(res.velocity_mean_cm_s), camera=camera))


def build_network_figure(res, patient, q_total_ref: float, camera: str = "Oblique", height: int = 540, animate: bool = False) -> go.Figure:
    """
    3D anchor. Default: static surface flow glyphs (zero client cost). animate=True adds Plotly frames for the glyph traces only plus Play / Pause
    buttons: the mesh is never re-sent. gl3d frames need redraw=True, so every frame redraws the scene: keep this opt-in and rehearse on the stage laptop.
    """
    specs, lay = network_scene(res, patient, q_total_ref, camera)
    fig = go.Figure()
    for typ, kw in specs:
        fig.add_trace(getattr(go, typ)(**kw))
    axis = dict(visible=False, showbackground=False)
    extra = {}
    if animate:
        frames, dur = glyph_frames(lay["glyph_args"])
        if frames:
            idx = list(range(lay["glyph_start"], lay["glyph_start"] + lay["glyph_n"]))
            fig.frames = [go.Frame(data=[getattr(go, typ)(**kw) for typ, kw in fr], traces=idx, name=f"f{i}") for i, fr in enumerate(frames)]
            extra["updatemenus"] = [dict(type="buttons", direction="left", x=0.02, y=0.09, xanchor="left", yanchor="bottom", showactive=False,
                                         bgcolor="#12263A", bordercolor="#2A4560", font=dict(color="#E6EEF5", size=12),
                                         buttons=[dict(label="\u25b6 Animate flow", method="animate",
                                                       args=[None, dict(frame=dict(duration=dur, redraw=True), transition=dict(duration=0),
                                                                        fromcurrent=True, mode="immediate")]),
                                                  dict(label="\u23f8 Pause", method="animate",
                                                       args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")])])]
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis=axis, yaxis=axis, zaxis=axis, bgcolor=DARK_BG, camera=lay["camera"]),
        paper_bgcolor=DARK_BG, margin=dict(l=0, r=0, t=0, b=0), height=height, showlegend=False, uirevision=f"ijv-3d-{camera}",
        annotations=[dict(text=f"<b>{lay['banner']}</b>", x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
                          showarrow=False, font=dict(size=20, color=STATE_COLORS[lay["state"]])),
                     dict(text=f"schematic network, jugular radius x{lay['exag']:g} for visibility", x=0.02, y=0.02, xref="paper", yref="paper",
                          xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=11, color="#8FA6BA"))], **extra)
    return fig


# =====================================================================================
# Dark clinical-monitor cards and telemetry alerts (own HTML: robust, no reliance on st.metric internals)
# =====================================================================================
def _card(label: str, value: str, unit: str, sub: str, cls: str, tip: str = "") -> str:
    hint = '<span class="mon-i" aria-hidden="true">i</span>' if tip else ""
    tip_html = f'<div class="mon-tip" role="tooltip"><b>{html.escape(label)}</b>{html.escape(tip)}</div>' if tip else ""
    return (f'<div class="mon-card c-{cls}" tabindex="0"><div class="mon-lab">{html.escape(label)}{hint}</div>'
            f'<div class="mon-val">{html.escape(value)}<span class="mon-unit">{html.escape(unit)}</span></div>'
            f'<div class="mon-sub">{sub}</div>{tip_html}</div>')


def metric_tips(network) -> dict:
    """What each card means and why it matters. Thresholds are quoted as MODEL cut-offs or literature ranges, never as clinical verdicts."""
    lo, hi = VENOUS_WSS_PA
    return {
        "IJV flow": f"Blood volume leaving through the jugular vein each second. Pre-flight supine reference for this patient: "
                    f"{network.q_ijv_ref:.1f} mL/s (a model input, not a population norm). Zero or negative means no net drainage toward the heart.",
        "Mean velocity": f"Average speed of blood along the vein. In this model |v| below {V_CUT:g} cm/s is classed as stagnant "
                         f"(an assumed cut-off, not a clinical standard). Slow flow is one contributor to clot risk.",
        "Residence time": "How long blood stays in the vein (vein volume divided by flow). Infinite means no exchange: the blood is trapped. "
                          "Stasis is one part of thrombosis risk (Virchow's triad), not a prediction of a clot.",
        "Wall shear stress": f"Frictional force of flowing blood on the vein wall. Veins normally run about {lo:g}\u2013{hi:g} Pa. The 0.4 Pa "
                             f"low-shear figure linked to endothelial dysfunction comes from arteries. Zero means no flow-driven stimulus at all.",
        "Cranial venous P_in": "Pressure the venous side of the head needs to push the prescribed outflow through the network. "
                               "An output of the model, not an input.",
        "Collateral flow": "Outflow carried by the vertebral collateral route instead of the jugular vein. Rises when the jugular route "
                           "is obstructed or its valve closes.",
    }


def monitor_cards_html(res, network, cols: int = 3) -> str:
    """Six monitor cards with hover tooltips. Wall shear STRESS (Pa) is station-wise Carreau viscosity x slit-corrected shear rate."""
    state = flow_state(res)
    lo, hi = VENOUS_WSS_PA
    tips = metric_tips(network)
    if res.wss_mean_Pa == 0.0:
        wss_cls, wss_sub = "trapped", "NO WALL SHEAR &middot; no flow &middot; below venous range"
    elif res.wss_mean_Pa < lo:
        wss_cls, wss_sub = "stagnant", f"BELOW VENOUS RANGE ({lo:g}&ndash;{hi:g} Pa) &middot; &mu;<sub>eff</sub> {res.viscosity_mean_mPa_s:.1f} mPa&middot;s"
    elif res.wss_mean_Pa > hi:
        wss_cls, wss_sub = "stagnant", f"ABOVE VENOUS RANGE ({lo:g}&ndash;{hi:g} Pa) &middot; &mu;<sub>eff</sub> {res.viscosity_mean_mPa_s:.1f} mPa&middot;s"
    else:
        wss_cls = "forward"
        wss_sub = (f"IN VENOUS RANGE ({lo:g}&ndash;{hi:g} Pa) &middot; &mu;<sub>eff</sub> {res.viscosity_mean_mPa_s:.1f} mPa&middot;s Carreau "
                   f"({res.wss_carreau_over_newtonian:.2f}&times; a constant 4 mPa&middot;s)")
    q_sub = {"trapped": "VALVE CLOSED" + (f" &middot; would reverse to {res.q_ijv_unconstrained:+.2f}" if math.isfinite(res.q_ijv_unconstrained) else ""),
             "reversed": "REVERSED toward the head", "stagnant": f"STAGNANT &middot; |v| below {V_CUT:g} cm/s",
             "forward": f"{res.q_ijv / network.q_ijv_ref - 1:+.0%} vs pre-flight supine"}[state]
    trapped_t = math.isinf(res.residence_time_s)
    cards = [
        _card("IJV flow", f"{res.q_ijv:+.2f}", "mL/s", q_sub, state, tips["IJV flow"]),
        _card("Mean velocity", f"{res.velocity_mean_cm_s:+.2f}", "cm/s", "locally Q/A along the vein", state, tips["Mean velocity"]),
        _card("Residence time", "\u221e" if trapped_t else f"{res.residence_time_s:.1f}", "" if trapped_t else "s",
              "BLOOD TRAPPED &middot; no exchange" if trapped_t else f"vein volume {res.volume_mL:.1f} mL", "trapped" if trapped_t else "neutral",
              tips["Residence time"]),
        _card("Wall shear stress", f"{res.wss_mean_Pa:.3f}", "Pa", wss_sub, wss_cls, tips["Wall shear stress"]),
        _card("Cranial venous P_in", f"{res.p_in:.2f}", "mmHg", f"{res.p_in - network.p_in_ref:+.2f} vs pre-flight {network.p_in_ref:g}", "neutral",
              tips["Cranial venous P_in"]),
        _card("Collateral flow", f"{res.q_collateral:.2f}", "mL/s", f"{res.q_collateral / res.q_total:.0%} of total outflow" if res.q_total > 0 else "", "neutral",
              tips["Collateral flow"]),
    ]
    return f'<div class="mon-grid mon-cols-{2 if cols == 2 else 3}">' + "".join(cards) + "</div>"


def telemetry_html(level: str, text: str, basis: str, conditional: bool = False) -> str:
    """NASA flowchart branch as a high-contrast telemetry alert. Never flashes; the slow pulse stops under prefers-reduced-motion."""
    cls = "tele-cond" if conditional else ("tele-crit" if level == "error" else "tele-warn")
    head = ("IF SUSTAINED (not a ~15 s maneuver) &middot; NASA OCHMO-MTB-007 FLOWCHART BRANCH" if conditional
            else ("PRIORITY &middot; NASA OCHMO-MTB-007 PROTOCOL" if level == "error" else "ADVISORY &middot; NASA OCHMO-MTB-007 PROTOCOL"))
    return (f'<div class="tele {cls}"><div class="tele-head">{head}</div><div class="tele-body">{html.escape(text)}</div>'
            f'<div class="tele-foot">Triggered by: {html.escape(basis)}. A simulated state mapped onto the NASA in-flight flowchart for '
            f'illustration; this research prototype does not give medical advice.</div></div>')


# =====================================================================================
# Modes, guided steps and the pure state resolver
# =====================================================================================
GUIDED, SANDBOX = "Guided Narrative", "Clinical Sandbox"
GUIDED_STEPS = [
    dict(title="Baseline Earth", short="1  Baseline Earth", scenario="supine_1g", tilt=0.0, resp_phase="Resting", coupling=0.0, valve=True),
    dict(title="6\u00b0 head-down tilt", short="2  6\u00b0 head-down tilt", scenario="supine_1g", tilt=-6.0, resp_phase="Resting", coupling=0.0, valve=True),
    dict(title="Valsalva at the jugular outlet + valve closure", short="3  Valsalva + valve closure", scenario="supine_1g", tilt=-6.0,
         resp_phase="Valsalva", coupling=0.0, valve=True),
    dict(title="Valsalva reaching both outlets", short="4  Both outlets rise", scenario="supine_1g", tilt=-6.0, resp_phase="Valsalva",
         coupling=1.0, valve=True),
]
HANDOFF_KEYS = ("A0", "K", "p_ext", "q_ref", "f_par", "share", "slit", "valve", "resp_phase", "coupling", "tilt", "resp_fall", "resp_rise",
                "risk_family", "risk_thrombo", "risk_hormone", "bc_override", "flight_override")


def guided_state(step: int) -> dict:
    """Complete model state for a guided step: default patient, no overrides, no risk factors."""
    g = GUIDED_STEPS[step]
    s = dict(A0=float(DEFAULT.A0_cm2), K=float(DEFAULT.K_mmHg), p_ext=float(DEFAULT.p_ext_mmHg), q_ref=float(DEFAULT.q_ref_ijv_mL_s),
             f_par=float(DEFAULT.parallel_drop_fraction), share=float(DEFAULT.ijv_share_ref), slit=DEFAULT.lumen_model == "slit",
             resp_fall=4.0, resp_rise=20.0, risk_family=False, risk_thrombo=False, risk_hormone=False, bc_override=False,
             ovr_qf=1.0, ovr_pout_ijv=5.0, ovr_pout_c=5.0, flight_override=False, flight_p_ext=4.0)
    s.update(scenario=g["scenario"], tilt=float(g["tilt"]), resp_phase=g["resp_phase"], coupling=float(g["coupling"]), valve=bool(g["valve"]))
    return s


def resolve_state(mode: str, step: int, widgets: dict) -> dict:
    """Single source of truth: guided steps never read the sidebar; the sandbox never reads the step table."""
    if mode == GUIDED:
        s = guided_state(step)
        s.update(anim_speed=float(widgets.get("anim_speed", 1.0)), map_res=widgets.get("map_res", "Standard"))
        return s
    return dict(widgets)


def state_pk(s: dict):
    return (float(s["A0"]), float(s["K"]), float(s["p_ext"]), float(s["q_ref"]), float(s["f_par"]), float(s["share"]), bool(s["slit"]), bool(s["valve"]))


def state_run(s: dict):
    """Solve a resolved state through the cached engine call (used for the guided snapshots)."""
    active_ = s["scenario"]
    cnd = scenario_inputs(active_, None)
    tilt_ = float(s["tilt"]) if (active_ == "supine_1g" and float(s["tilt"]) != 0.0) else None
    pe = float(s["p_ext"])
    return cached_result(active_, None, None, None, state_pk(s), pe, s["resp_phase"], float(s["coupling"]),
                         (float(s["resp_fall"]), float(s["resp_rise"])), tilt_)


def step_html(i: int, R: list) -> str:
    """Narrative for step i with live numbers from the solved results R[0..i]."""
    r, r0 = R[i], R[0]
    thr = r.reversal_threshold_mmHg
    if i == 0:
        return (f'<p><b>Baseline Earth.</b> One subject lying supine at 1 g, the state the model is calibrated to. Total cranial outflow is held fixed '
                f'at {r.q_total:.2f} mL/s (autoregulation): {r.q_ijv:.2f} mL/s ({r.ijv_share:.0%}) leaves through the jugular vein and the rest through the '
                f'vertebral collateral. Cranial venous pressure is {r.p_in:.1f} mmHg and wall shear stress is {r.wss_mean_Pa:.2f} Pa, inside the '
                f'0.1&ndash;0.6 Pa range usually quoted for veins.</p>'
                f'<p class="muted">This walkthrough assumes a competent jugular valve and changes one thing per step. It is one steady-state model with '
                f'assumed parameters, not validated against flight data.</p>')
    if i == 1:
        return (f'<p><b>6&deg; head-down tilt, an Earth analog of the head-ward fluid shift.</b> Hydrostatic pressure now acts toward the head. The '
                f'model&rsquo;s answer is deliberately unspectacular: cranial venous pressure rises by {r.p_in - r0.p_in:+.2f} mmHg (the hydrostatic head '
                f'across the vein), the vein distends {r.area_mean_cm2 / r0.area_mean_cm2 - 1:+.1%}, and jugular flow is unchanged '
                f'({r0.q_ijv:.2f} &rarr; {r.q_ijv:.2f} mL/s). Gravity acts equally on both drainage paths, so it shifts pressure, not the flow split.</p>'
                f'<p class="muted">Tilt is only an analog: the model&rsquo;s own microgravity scenario has lower, not higher, outlet pressures than supine.</p>')
    if i == 2:
        unc = f"{r.q_ijv_unconstrained:+.2f} mL/s" if math.isfinite(r.q_ijv_unconstrained) else "a negative value"
        return (f'<p><b>Valsalva with the pressure rise at the jugular outlet only (transient, about 15 s).</b> The strain raises the jugular outlet pressure by '
                f'{r.respiration_shift_mmHg:.0f} mmHg while the collateral outlet stays put. The outlet difference ({r.p_out_difference_mmHg:+.1f} mmHg) now exceeds '
                f'the reversal threshold ({thr:.2f} mmHg = Q<sub>total</sub> &times; R<sub>collat</sub>), so without a valve jugular flow would reverse to {unc}. '
                f'A competent valve closes instead: jugular flow is exactly 0, the collateral carries all {r.q_total:.2f} mL/s, residence time is infinite and wall '
                f'shear stress is 0 Pa. The vein turns red: trapped blood.</p>'
                f'<p><b>Assumption to know:</b> this step applies the pressure rise to the jugular outlet only. A real strain acts on the whole chest; the next step '
                f'shows what happens when it reaches both outlets.</p>'
                f'<p class="muted">A Valsalva lasts seconds, so &ldquo;infinite residence time&rdquo; is a steady-state idealisation, not a prediction of clot formation.</p>')
    return (f'<p><b>The same Valsalva, but the pressure reaches both outlets equally.</b> The jugular and collateral outlets each rise by '
            f'{r.respiration_shift_mmHg:.0f} mmHg, so the outlet difference stays at {r.p_out_difference_mmHg:+.1f} mmHg, below the {thr:.2f} mmHg threshold. '
            f'The valve stays open, jugular flow is {r.q_ijv:.2f} mL/s and nothing is trapped. Cranial venous pressure rises by the full '
            f'{r.respiration_shift_mmHg:.0f} mmHg ({R[1].p_in:.1f} &rarr; {r.p_in:.1f} mmHg) because the collateral transmits it.</p>'
            f'<p><b>What this shows, in this model:</b> reversal or valve closure needs an imbalance between the two outlet pressures; a uniform thoracic pressure '
            f'rise does not create one. Whether the anatomy of astronauts with abnormal jugular flow provides such an imbalance is the open question this model '
            f'cannot answer.</p>')


def snapshot_html(r) -> str:
    rt = "&infin;" if math.isinf(r.residence_time_s) else f"{r.residence_time_s:.1f} s"
    return (f'<div class="snap">Q<sub>IJV</sub> {r.q_ijv:+.2f} mL/s &middot; P<sub>in</sub> {r.p_in:.2f} mmHg &middot; '
            f'WSS {r.wss_mean_Pa:.3f} Pa &middot; residence {rt}</div>')


def embed_html(markup: str, height: int) -> None:
    """Embed a self-contained HTML page. st.components.v1.html is deprecated from Streamlit 1.56, so prefer st.iframe when present."""
    if hasattr(st, "iframe"):
        try:
            st.iframe(markup, height=height)
            return
        except Exception:
            pass
    components.html(markup, height=height)


# =====================================================================================
# Session state, modes and URL persistence
# =====================================================================================
state_defaults = {
    "scenario": "supine_1g", "A0": DEFAULT.A0_cm2, "K": float(DEFAULT.K_mmHg), "p_ext": DEFAULT.p_ext_mmHg,
    "q_ref": DEFAULT.q_ref_ijv_mL_s, "slit": DEFAULT.lumen_model == "slit", "share": float(DEFAULT.ijv_share_ref),
    "f_par": float(DEFAULT.parallel_drop_fraction), "bc_override": False, "ovr_qf": 1.0, "ovr_pout_ijv": 5.0,
    "ovr_pout_c": 5.0, "flight_override": False, "flight_p_ext": 4.0, "map_res": "Standard", "anim_speed": 1.0,
    "valve": bool(DEFAULT.valve_competent), "resp_phase": "Resting", "coupling": 0.0, "tilt": 0.0, "resp_fall": 4.0,
    "resp_rise": 20.0, "risk_family": False, "risk_thrombo": False, "risk_hormone": False, "cam": "Oblique", "anim3d": False,
}
for _k, _v in state_defaults.items():
    st.session_state.setdefault(_k, _v)
if st.session_state["scenario"] not in BUTTON_LABELS:
    st.session_state["scenario"] = "supine_1g"


def _qp(name: str, default=None):
    try:
        v = st.query_params.get(name, default)
    except Exception:
        return default
    return v[0] if isinstance(v, (list, tuple)) and v else v


# First load only: restore mode and step from the URL, so a browser refresh mid-talk does not lose the place.
if "ui_mode" not in st.session_state:
    st.session_state["ui_mode"] = SANDBOX if str(_qp("mode", "guided")).lower() == "sandbox" else GUIDED
if "g_step" not in st.session_state:
    try:
        st.session_state["g_step"] = min(max(int(_qp("step", 1)) - 1, 0), len(GUIDED_STEPS) - 1)
    except (TypeError, ValueError):
        st.session_state["g_step"] = 0


def set_scenario(key: str) -> None:
    st.session_state["scenario"] = key


def _go(delta: int) -> None:
    st.session_state["g_step"] = min(max(st.session_state["g_step"] + delta, 0), len(GUIDED_STEPS) - 1)


def _restart() -> None:
    st.session_state["g_step"] = 0


def _open_in_sandbox() -> None:
    """Explicit hand-off (a callback, so it may write widget keys): copy the current guided state into the sandbox controls."""
    g = guided_state(st.session_state["g_step"])
    for k in HANDOFF_KEYS:
        st.session_state[k] = g[k]
    st.session_state["scenario"] = g["scenario"]
    st.session_state["ui_mode"] = SANDBOX


def _sync_url(mode_: str, step_: int) -> None:
    for k, v in (("mode", "guided" if mode_ == GUIDED else "sandbox"), ("step", str(step_ + 1))):
        try:
            if st.query_params.get(k) != v:
                st.query_params[k] = v
        except Exception:
            pass


st.title("IJV-Comply")
st.markdown('<div class="ijv-sub">Patient-parameterized jugular vein testbed: autoregulated cranial outflow, '
            'compliant-vein mechanics and collateral drainage under simulated spaceflight</div>', unsafe_allow_html=True)
mode = st.radio("Mode", [GUIDED, SANDBOX], key="ui_mode", horizontal=True, label_visibility="collapsed")

# Clinical-telemetry styling. Plain <style> injection; classes are ours, so nothing depends on Streamlit's internal class names.
st.markdown(
    f"""
    <style>
    .c-forward {{ --c: {STATE_COLORS['forward']}; }} .c-stagnant {{ --c: {STATE_COLORS['stagnant']}; }}
    .c-reversed {{ --c: {STATE_COLORS['reversed']}; }} .c-trapped {{ --c: {STATE_COLORS['trapped']}; }} .c-neutral {{ --c: #7FA3C4; }}
    .mon-grid {{ display: grid; gap: 10px; margin: 0.2rem 0 0.6rem 0; }}
    .mon-cols-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .mon-cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .mon-card {{ background: {DARK_BG}; border: 1px solid #1E3347; border-left: 6px solid var(--c); border-radius: 6px;
                 padding: 0.65rem 0.85rem 0.6rem 0.85rem; color: #E6EEF5; font-family: 'Barlow','Segoe UI',Arial,sans-serif; }}
    .mon-lab {{ font-size: 0.7rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: #8FA6BA; }}
    .mon-val {{ font-size: 2.05rem; font-weight: 700; line-height: 1.15; font-variant-numeric: tabular-nums; color: var(--c); }}
    .mon-unit {{ font-size: 0.95rem; font-weight: 500; color: #9FB4C7; margin-left: 0.35rem; }}
    .mon-sub {{ font-size: 0.76rem; color: #A9BCCD; margin-top: 0.1rem; line-height: 1.3; }}
    /* hover / keyboard-focus tooltip: an overlay INSIDE the card, so no parent container can clip it */
    .mon-card {{ position: relative; min-height: 6.9rem; cursor: help; outline: none; }}
    .mon-i {{ display: inline-block; margin-left: 0.45rem; width: 1.15em; height: 1.15em; line-height: 1.1em; text-align: center; font-size: 0.7rem;
              font-weight: 700; font-style: italic; border: 1px solid #5E7A94; border-radius: 50%; color: #9FB4C7; letter-spacing: 0; text-transform: none; }}
    .mon-tip {{ position: absolute; top: 0; right: 0; bottom: 0; left: 0; box-sizing: border-box; padding: 0.5rem 0.8rem; background: #0F2438;
                border: 1px solid var(--c); border-radius: 6px; color: #E6EEF5; font-size: 0.74rem; line-height: 1.35; overflow: auto;
                opacity: 0; visibility: hidden; transition: opacity 0.15s ease; z-index: 2; }}
    .mon-tip b {{ display: block; font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--c); margin-bottom: 0.15rem; }}
    .mon-card:hover .mon-tip, .mon-card:focus .mon-tip, .mon-card:focus-within .mon-tip {{ opacity: 1; visibility: visible; }}
    @media (prefers-reduced-motion: reduce) {{ .mon-tip {{ transition: none; }} }}
    .tele {{ border: 3px solid #EF4444; background: #1B0508; color: #FFFFFF; border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0;
             font-family: 'Barlow','Segoe UI',Arial,sans-serif; }}
    .tele-head {{ font-size: 0.74rem; font-weight: 800; letter-spacing: 0.16em; color: #FF9A9A; }}
    .tele-body {{ font-size: 1.15rem; font-weight: 800; line-height: 1.35; margin: 0.25rem 0; }}
    .tele-foot {{ font-size: 0.74rem; color: #E7B8B8; }}
    .tele-warn {{ border-color: #F59E0B; background: #1A1204; }} .tele-warn .tele-head {{ color: #FFC766; }} .tele-warn .tele-foot {{ color: #EBCB93; }}
    .tele-cond {{ border-style: dashed; border-color: #F59E0B; background: #1A1204; }} .tele-cond .tele-head {{ color: #FFC766; }}
    .tele-cond .tele-foot {{ color: #EBCB93; }}
    @keyframes telepulse {{ 0%, 100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0); }} 50% {{ box-shadow: 0 0 0 7px rgba(239,68,68,0.35); }} }}
    .tele-crit {{ animation: telepulse 2s ease-in-out infinite; }}
    @media (prefers-reduced-motion: reduce) {{ .tele-crit {{ animation: none; }} }}
    .stepper {{ display: flex; gap: 8px; margin: 0.3rem 0 0.5rem 0; }}
    .step-pill {{ flex: 1; border: 1px solid #D5DCE5; border-radius: 6px; padding: 0.45rem 0.7rem; background: #F4F7FA; color: {SLATE};
                  font-size: 0.88rem; font-weight: 600; }}
    .step-pill.on {{ background: {INK}; color: #FFFFFF; border-color: {INK}; }} .step-pill.done {{ background: #E7F3F1; color: {INK}; }}
    .step-card {{ background: #FFFFFF; border: 1px solid #D5DCE5; border-left: 5px solid {BLUE}; border-radius: 6px; padding: 0.7rem 1rem;
                  margin: 0.45rem 0; color: {INK}; }}
    .step-card.past {{ opacity: 0.72; border-left-color: #B8C4D2; }} .step-card p {{ margin: 0.2rem 0 0.45rem 0; line-height: 1.5; }}
    .step-card .muted {{ color: {SLATE}; font-size: 0.9rem; }}
    .step-card .snap {{ margin-top: 0.3rem; font-size: 0.8rem; color: {SLATE}; border-top: 1px dashed #D5DCE5; padding-top: 0.3rem; }}
    .block-container {{ max-width: 1600px; }}
    </style>
    """,
    unsafe_allow_html=True,
)
if mode == GUIDED:       # Guided mode: no sidebar at all (all known Streamlit selectors; the widgets stay rendered so state survives)
    st.markdown("<style>section[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], [data-testid='collapsedControl'], "
                "[data-testid='stSidebarCollapseButton'] { display: none !important; }</style>", unsafe_allow_html=True)


# =====================================================================================
# Sidebar: patient parameterisation, overrides, model assumptions
# =====================================================================================
with st.sidebar:
    st.header("Patient parameterization")
    st.caption("Pre-flight inputs for one astronaut. Any change recalibrates the network.")
    A0 = st.slider("Reference area, A₀ (cm²)", 0.30, 2.00, step=0.01, key="A0",
                   help="Area at zero transmural pressure. NOT the measured supine area: a supine vein sits at positive "
                        "transmural pressure and is larger than A₀.")
    K = st.select_slider("Wall stiffness, K (mmHg, log scale)", options=K_OPTIONS, key="K", format_func=lambda v: f"{v:.2f}",
                         help="Shapiro stiffness. It sets how readily the vein collapses upright; it does not move the "
                              "reversal threshold.")
    p_ext = st.slider("Tissue pressure, P_ext (mmHg)", -5.0, 10.0, step=0.1, key="p_ext",
                      help="Perivascular tissue pressure; unmeasured in flight. Also the pre-flight value used for calibration.")
    q_ref = st.slider("Reference supine IJV flow, Q_ref (mL/s)", 1.0, 10.0, step=0.1, key="q_ref",
                      help="Pre-flight supine Doppler flow in this vein. With the supine IJV share it sets the reference total "
                           "cranial outflow that every scenario scales.")

    st.markdown("**Anatomy and maneuvers**")
    valve = st.toggle("Competent Venous Valve Enabled", key="valve",
                      help="A valve is present in about 90 % of people, but competence varies by study (insufficiency on Valsalva "
                           "reported in roughly 29 % to 90 % of healthy subjects depending on method), so there is no 'typical' "
                           "setting. On: if the pressure gradient would reverse the jugular flow, the valve closes and Q_IJV is "
                           "forced to exactly 0; the collateral carries all outflow and residence time becomes infinite. Off: no "
                           "valve, or an incompetent one (reversal allowed).")
    resp_phase = st.select_slider("Respiration Phase", options=list(RESPIRATION_PHASES), key="resp_phase",
                                  help="Thoracic pump. Shifts the jugular outlet pressure: deep inspiration lowers it, Valsalva raises "
                                       "it. The magnitudes are set under 'Model assumptions'. NASA surveillance scans include "
                                       "breathing maneuvers.")
    coupling = st.slider("Share of the thoracic pressure change reaching the collateral outlet", 0.0, 1.0, step=0.05, key="coupling",
                         format="%.2f",
                         help="0 = respiration shifts the jugular outlet only (as specified). 1 = both outlets shift together, which "
                              "is closer to the physiology because the azygos and vertebral systems also drain inside the thorax. With "
                              "equal shifts the reversal threshold does not move; only cranial venous pressure P_in rises.")
    tilt_ok = st.session_state["scenario"] == "supine_1g"
    tilt = st.slider("Head-down tilt angle (°)", -6.0, 0.0, step=0.5, key="tilt", disabled=not tilt_ok, format="%.1f",
                     help="Earth analog, supine scenario only. 0° = supine, −6° = NASA head-down bed-rest standard. Sets g_eff = 9.81 × "
                          "sin(angle). Hydrostatic head acts on both drainage paths equally, so it raises cranial venous pressure P_in "
                          "and distends the vein slightly but does not move the flow split or the reversal threshold.")

    with st.expander("Scenario overrides"):
        bc_override = st.checkbox("Override the active scenario's inputs", key="bc_override")
        ovr_qf = st.slider("Total outflow factor (× reference)", 0.2, 1.5, step=0.05, key="ovr_qf", disabled=not bc_override)
        ovr_pout_ijv = st.slider("Jugular outlet pressure, P_out,IJV (mmHg)", 0.0, 20.0, step=0.1, key="ovr_pout_ijv",
                                 disabled=not bc_override)
        ovr_pout_c = st.slider("Collateral outlet pressure, P_out,collat (mmHg)", 0.0, 15.0, step=0.1, key="ovr_pout_c",
                               disabled=not bc_override)
    with st.expander("Scenario options"):
        flight_override = st.checkbox("Use a different tissue pressure in microgravity (edema hypothesis)", key="flight_override",
                                      help="Applies to the two microgravity scenarios. The network stays calibrated to the pre-flight P_ext.")
        flight_p_ext = st.slider("Microgravity tissue pressure (mmHg)", -5.0, 10.0, step=0.1, key="flight_p_ext",
                                 disabled=not flight_override)
    with st.expander("NASA OCHMO-MTB-007 Risk Factors"):
        risk_family = st.checkbox("Family History", key="risk_family")
        risk_thrombo = st.checkbox("Thrombophilia", key="risk_thrombo")
        risk_hormone = st.checkbox("High-Risk Hormones", key="risk_hormone")
        st.caption("Selects the branch of the NASA in-flight flowchart applied when the simulated state shows stasis. A reference "
                   "mapping for illustration, not medical advice.")
    with st.expander("Model assumptions (advanced)"):
        slit = st.checkbox("Slit-lumen resistance correction", key="slit",
                           help="A collapsing vein flattens into a constant-perimeter ellipse instead of shrinking as a circle, "
                                "which raises resistance (about 5× at 74 % area loss, 12× at 90 %). Untick to see the "
                                "circular-equivalent model.")
        share = st.slider("Supine jugular share of cranial outflow", 0.50, 0.90, step=0.01, key="share", format="%.2f",
                          help="Default 0.66 (Doepp 2004, as cited by Lan 2021).")
        f_par = st.slider("Supine pressure drop across the jugular/collateral pair", 0.05, 0.90, step=0.05, key="f_par",
                          format="%.2f",
                          help="Fraction of the supine cranial-to-outlet drop across the jugular vein and collateral. An "
                               "assumption: together with the share it sets the collateral resistance and so the reversal "
                               "threshold.")
        resp_fall = st.slider("Deep-inspiration fall in CVP (mmHg)", 1.0, 8.0, step=0.5, key="resp_fall", format="%.1f",
                              help="Right atrial pressure falls by 'several mmHg' on inspiration (working value 4).")
        resp_rise = st.slider("Valsalva rise in CVP (mmHg)", 5.0, 40.0, step=1.0, key="resp_rise", format="%.0f",
                              help="A standard 40 mmHg strain raised central venous pressure by about 40 mmHg (J Appl Physiol 2000). "
                                   "The default 20 is a moderate strain.")
        st.caption("These calibration assumptions are not patient measurements. The defaults give the default patient a "
                   "collateral-majority upright state; changing them moves that and the reversal threshold.")
    with st.expander("Display and speed"):
        anim_speed = st.select_slider("Animation speed", options=[0.25, 0.5, 1.0, 2.0], key="anim_speed",
                                      format_func=lambda v: f"{v:g}× real time")
        map_res = st.select_slider("Map resolution", options=list(MAP_RES), key="map_res",
                                   help="Solved nodes before interpolation. Higher is slower on the first draw, then cached.")
        precompute = st.button("Pre-compute all four maps", help="Warms the cache so scenario switches are instant on stage.")

# ---- single source of truth: guided steps never read the sidebar, the sandbox never reads the step table ----
_widgets = dict(A0=A0, K=K, p_ext=p_ext, q_ref=q_ref, f_par=f_par, share=share, slit=slit, valve=valve, resp_phase=resp_phase,
                coupling=coupling, tilt=tilt, resp_fall=resp_fall, resp_rise=resp_rise, risk_family=risk_family,
                risk_thrombo=risk_thrombo, risk_hormone=risk_hormone, bc_override=bc_override, ovr_qf=ovr_qf,
                ovr_pout_ijv=ovr_pout_ijv, ovr_pout_c=ovr_pout_c, flight_override=flight_override, flight_p_ext=flight_p_ext,
                anim_speed=anim_speed, map_res=map_res, scenario=st.session_state["scenario"])
g_step = st.session_state["g_step"]
S = resolve_state(mode, g_step, _widgets)
(A0, K, p_ext, q_ref, f_par, share, slit, valve, resp_phase, coupling, tilt, resp_fall, resp_rise, risk_family, risk_thrombo,
 risk_hormone, bc_override, ovr_qf, ovr_pout_ijv, ovr_pout_c, flight_override, flight_p_ext, anim_speed, map_res) = (
    S[k] for k in ("A0", "K", "p_ext", "q_ref", "f_par", "share", "slit", "valve", "resp_phase", "coupling", "tilt", "resp_fall",
                   "resp_rise", "risk_family", "risk_thrombo", "risk_hormone", "bc_override", "ovr_qf", "ovr_pout_ijv", "ovr_pout_c",
                   "flight_override", "flight_p_ext", "anim_speed", "map_res"))

pk = (float(A0), float(K), float(p_ext), float(q_ref), float(f_par), float(share), bool(slit), bool(valve))
network, cal_error = cached_calibration(pk)
with st.sidebar:
    if network is not None:
        st.success(f"Calibrated ({network.method.upper()}) from supine P_in {network.p_in_ref:g} → outlets "
                   f"{network.p_out_ref:g} mmHg. Reference total outflow {network.q_total_ref:.2f} mL/s. "
                   f"R_up {network.r_up:.3f}, R_term {network.r_term:.3f}, R_collat {network.r_collat:.3f} mmHg·s/mL.")

if network is None:
    st.error(f"**This combination can't be calibrated.** {cal_error}")
    st.info("Try a lower tissue pressure, a lower reference flow, a larger reference area, or a different pressure-drop "
            "share. Calibration needs the supine gradient to exceed the drop across the jugular branch at the reference flow.")
    st.stop()

patient = make_patient(pk)
active = S["scenario"]
ovr = {"q_total_factor": float(ovr_qf), "p_out_ijv": float(ovr_pout_ijv), "p_out_collat": float(ovr_pout_c)} if bc_override else None
cond = scenario_inputs(active, ovr)

# Effective conditions after the thoracic pump and head-down tilt (closed-form shifts of the outlets and of g_eff).
resp_table = {"Deep Inspiration": -float(resp_fall), "Valsalva": float(resp_rise)}
tilt_run = float(tilt) if (active == "supine_1g" and float(tilt) != 0.0) else None
cond_eff = apply_respiration(cond, resp_phase, float(coupling), resp_table)
g_run = effective_gravity_for_tilt(tilt_run) if tilt_run is not None else float(cond["g_eff"])
p_oi_nom, p_oc_nom = float(cond["p_out_ijv"]), float(cond["p_out_collat"])        # nominal: the static map background

if precompute:
    bar = st.sidebar.progress(0.0)
    for i, key in enumerate(list_scenarios()):
        c_k = scenario_inputs(key, ovr if key == active else None)
        cached_map(key, float(c_k["q_total_factor"]), float(c_k["p_out_collat"]), pk, map_res)
        bar.progress((i + 1) / len(list_scenarios()))
    st.sidebar.caption("All four maps cached.")

# ---- solve once, for whichever mode is active ----
use_flight_pext = bool(flight_override) and active in FLIGHT_KEYS
p_ext_run = float(flight_p_ext) if use_flight_pext else float(p_ext)
try:
    res = cached_result(active, ovr["q_total_factor"] if ovr else None, ovr["p_out_ijv"] if ovr else None,
                        ovr["p_out_collat"] if ovr else None, pk, p_ext_run, resp_phase, float(coupling),
                        (float(resp_fall), float(resp_rise)), tilt_run)
except Exception as exc:
    st.error(f"The engine could not solve this state: {exc}")
    st.stop()
_sync_url(mode, g_step)

# =====================================================================================
# Guided Narrative: a linear walk through four states, one change at a time; no sidebar, no map
# =====================================================================================
if mode == GUIDED:
    step, n_steps = g_step, len(GUIDED_STEPS)
    R = [state_run(guided_state(i)) for i in range(step + 1)]
    st.markdown('<div class="stepper">' + "".join(
        f'<div class="step-pill {"on" if i == step else ("done" if i < step else "")}">{html.escape(g["short"])}</div>'
        for i, g in enumerate(GUIDED_STEPS)) + "</div>", unsafe_allow_html=True)
    nav = st.columns([1, 1, 1, 2.4])
    nav[0].button("\u25c0 Back", key="g_back", on_click=_go, args=(-1,), disabled=step == 0)
    nav[1].button("Next \u25b6", key="g_next", on_click=_go, args=(1,), disabled=step == n_steps - 1, type="primary")
    nav[2].button("\u21ba Restart", key="g_restart", on_click=_restart)
    nav[3].button("Explore this state in the Clinical Sandbox \u25b6", key="g_open", on_click=_open_in_sandbox)
    left, right = st.columns([1.25, 1])
    with left:
        anim3d = st.toggle("Animate 3D flow (experimental)", key="anim3d",
                      help="Static glyphs are the default and cost nothing. Animating re-draws the whole 3D scene on every frame in your browser, so test it on "
                           "the presentation laptop first. The run is bounded (about 5 s); press Play again to repeat it. Any control change stops it.")
        show_plotly(build_network_figure(res, patient, network.q_total_ref, "Oblique", animate=bool(anim3d)), "ijv_comply_network", key="net3d")
        g_sub = (f"Q_total {res.q_total:.2f} mL/s (prescribed)  |  P_in {res.p_in:.1f} (solved)  |  outlets: IJV {res.p_out_ijv:g}, "
                 f"collateral {res.p_out_collat:g} mmHg  |  {resp_phase}" + (f"  |  tilt {tilt_run:g}\u00b0" if tilt_run is not None else ""))
        embed_html(flow_animation_html(res, patient, GUIDED_STEPS[step]["title"], g_sub, 1.0), height=300)
    with right:
        st.markdown(monitor_cards_html(res, network, cols=2), unsafe_allow_html=True)
        st.markdown(f'<div class="ijv-flags"><div class="lbl">Regime flags</div>{badges_html(res.regime_flags)}</div>', unsafe_allow_html=True)
        proto = nasa_protocol(res.stasis, False, False, False)
        if proto is not None:
            st.markdown(telemetry_html(proto[0], proto[1], res.stasis_basis, conditional=True), unsafe_allow_html=True)
    eyebrow("What you are seeing")
    for i in range(step + 1):
        st.markdown(f'<div class="step-card {"" if i == step else "past"}">{step_html(i, R)}{snapshot_html(R[i])}</div>', unsafe_allow_html=True)
    st.caption("Exploratory research prototype: one steady-state model with assumed parameters, a schematic (not patient-specific) network, "
               "not validated against flight data and not medical advice. Wall shear stress = Carreau viscosity \u00d7 slit-corrected shear rate.")
    st.stop()

# =====================================================================================
# Header
# =====================================================================================
st.markdown(
    '<div class="ijv-hook">A 2020 case report described the first left internal jugular vein thrombus in an ISS astronaut '
    '(Auñón-Chancellor et al., NEJM). NASA expert panels judged stagnant flow the most concerning pre-thrombotic indicator, '
    'yet its causes in microgravity remain not well understood (NASA OCHMO-MTB-007). Here, total cranial outflow is '
    'prescribed (autoregulation); the jugular vein and the vertebral collateral drain to separate outlets; and the model solves '
    'how the flow splits, whether the jugular vein reverses, and what cranial venous pressure is needed. Exploratory and not '
    'validated.</div>',
    unsafe_allow_html=True,
)
q_total_in = float(cond["q_total_factor"]) * network.q_total_ref
thr_now = reversal_threshold_mmHg(network, q_total_in)
st.markdown(
    '<div class="ijv-status">'
    '<span class="ijv-chip">Solver <b>1D steady + 0D fallback</b></span>'
    '<span class="ijv-chip">Boundary <b>prescribed outflow, two outlets, solved P_in</b></span>'
    f'<span class="ijv-chip">Blood <b>{html.escape(patient.rheology.title())}</b></span>'
    f'<span class="ijv-chip">Collapsed lumen <b>{"slit (constant-perimeter ellipse)" if slit else "circular equivalent"}</b></span>'
    f'<span class="ijv-chip">Jugular reverses at <b>ΔP_out &gt; {thr_now:.1f} mmHg</b></span>'
    f'<span class="ijv-chip">Patient <b>A₀ {A0:.2f} cm² · K {K:.2f} mmHg · P_ext {p_ext:g} mmHg · Q_ref {q_ref:.1f} mL/s</b></span>'
    '</div>', unsafe_allow_html=True)

# =====================================================================================
# Top: scenario buttons and the inputs they prescribe
# =====================================================================================
eyebrow("Scenario")
btn_cols = st.columns(4)
for col, key in zip(btn_cols, list_scenarios()):
    col.button(BUTTON_LABELS.get(key, key), key=f"btn_{key}", type="primary" if key == active else "secondary",
               on_click=set_scenario, args=(key,), **stretch(st.button))

use_flight_pext = bool(flight_override) and active in FLIGHT_KEYS
p_ext_run = float(flight_p_ext) if use_flight_pext else float(p_ext)
p_oi_in, p_oc_in = float(cond_eff["p_out_ijv"]), float(cond_eff["p_out_collat"])       # effective (with respiration)
shift_ijv, shift_col = p_oi_in - p_oi_nom, p_oc_in - p_oc_nom

b1, b2, b3, b4, b5, b6 = st.columns(6)
b1.metric("Total cranial outflow (prescribed)", f"{q_total_in:.2f} mL/s", f"×{float(cond['q_total_factor']):g} of reference",
          delta_color="off", help="Driving input. Autoregulation holds total outflow; the model solves where it goes.")
b2.metric("P_out,IJV (jugular outlet)", f"{p_oi_in:g} mmHg",
          f"{p_oi_nom:g} {shift_ijv:+g} ({resp_phase})" if shift_ijv else ("override" if bc_override else None), delta_color="off",
          help="Pressure where the jugular vein drains (brachiocephalic side), including the respiration shift.")
b3.metric("P_out,collat (collateral outlet)", f"{p_oc_in:g} mmHg", f"{p_oc_nom:g} {shift_col:+g}" if shift_col else None, delta_color="off",
          help="Pressure where the vertebral collateral drains (azygos/vertebral side). A superior vena cava rise would raise both outlets.")
b4.metric("ΔP_out = IJV − collat", f"{p_oi_in - p_oc_in:+g} mmHg", delta_color="off",
          help="Compared with the reversal threshold below.")
b5.metric("g_eff (axial)", f"{g_run:.2f} m/s²", f"head-down tilt {tilt_run:g}°" if tilt_run is not None else None, delta_color="off")
b6.metric("P_ext used", f"{p_ext_run:g} mmHg", "flight override" if use_flight_pext else None, delta_color="off")
st.caption(f"{cond['description']}" + ("  Active-scenario override is on." if bc_override else ""))

# Middle: solved readout (3D anchor + monitor cards), animated flow
# =====================================================================================
eyebrow("Solved state")
left, right = st.columns([1.25, 1])
with left:
    cam = st.radio("View", list(CAMERAS), key="cam", horizontal=True, label_visibility="collapsed")
    anim3d = st.toggle("Animate 3D flow (experimental)", key="anim3d",
                      help="Static glyphs are the default and cost nothing. Animating re-draws the whole 3D scene on every frame in your browser, so test it on "
                           "the presentation laptop first. The run is bounded (about 5 s); press Play again to repeat it. Any control change stops it.")
    show_plotly(build_network_figure(res, patient, network.q_total_ref, cam, animate=bool(anim3d)), "ijv_comply_network", key="net3d")
with right:
    st.markdown(monitor_cards_html(res, network, cols=2), unsafe_allow_html=True)
    st.markdown(f'<div class="ijv-flags"><div class="lbl">Regime flags</div>{badges_html(res.regime_flags)}</div>', unsafe_allow_html=True)
    st.caption(f"Mean area {res.area_mean_cm2:.2f} cm\u00b2 ({res.area_mean_cm2 / A0:.0%} of A\u2080) \u00b7 collapse resistance penalty "
               f"\u00d7{res.shape_penalty_max:.1f}" + (f" (lumen aspect ratio {res.aspect_ratio_max:.0f}:1)" if res.shape_penalty_max > 1.001 else "")
               + f" \u00b7 wall shear rate {res.shear_mean_1_s:.0f} s\u207b\u00b9")
for note in res.warnings:
    st.warning(note)
if res.valve_closed:
    st.info(f"Competent venous valve closed. The pressure gradient would have reversed the jugular flow to "
            f"{res.q_ijv_unconstrained:+.2f} mL/s; the valve holds it at exactly 0. The collateral carries all {res.q_total:.2f} mL/s, "
            f"the trapped column has infinite residence time in this steady-state model, and cranial venous pressure P_in is "
            f"{res.p_in:.2f} mmHg." if math.isfinite(res.q_ijv_unconstrained) else
            f"Competent venous valve closed: Q_IJV held at exactly 0, collateral carries all {res.q_total:.2f} mL/s, P_in {res.p_in:.2f} mmHg.")
protocol = nasa_protocol(res.stasis, risk_family, risk_thrombo, risk_hormone)
if protocol is not None:
    st.markdown(telemetry_html(protocol[0], protocol[1], res.stasis_basis), unsafe_allow_html=True)

anim_title = BUTTON_LABELS.get(active, cond["label"])
anim_sub = (f"Q_total {res.q_total:.2f} mL/s (prescribed)  |  P_in {res.p_in:.1f} (solved)  |  outlets: IJV {res.p_out_ijv:g}, "
            f"collateral {res.p_out_collat:g} mmHg  |  g_eff {res.g_eff:.2f}  |  P_ext {p_ext_run:g}  |  {resp_phase}"
            + (f"  |  tilt {tilt_run:g}\u00b0" if tilt_run is not None else ""))
embed_html(flow_animation_html(res, patient, anim_title, anim_sub, float(anim_speed)), height=320)
st.caption("Drawn to scale (15 cm vein). Lumen height follows the 1D area profile; particles move at the local velocity Q/A with "
           "a parabolic radial profile, and their density follows the local area, so their average speed equals the reported "
           "mean velocity. They run backward when jugular flow reverses and stop dead when the valve closes. The dashed outline is the reference lumen A\u2080.")

# =====================================================================================
# =====================================================================================
# The exact reversal threshold
# =====================================================================================
eyebrow("Reversal threshold: exact, not a numerical effect")
st.latex(r"P_{out,\mathrm{IJV}} \;-\; P_{out,\mathrm{collat}} \;=\; Q_{total} \times R_{collat}")
d_out = p_oi_in - p_oc_in
if math.isfinite(thr_now):
    margin = d_out - thr_now
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Outlet difference", f"{d_out:+.2f} mmHg", f"{p_oi_in:g} − {p_oc_in:g}", delta_color="off",
              help="P_out,IJV − P_out,collat for the active scenario.")
    t2.metric("Q_total × R_collat (threshold)", f"{thr_now:.2f} mmHg", f"{q_total_in:.2f} mL/s × {network.r_collat:.3f} mmHg·s/mL",
              delta_color="off", help="Reversal threshold: P_out,IJV − P_out,collat = Q_total × R_collat.")
    t3.metric("Margin", f"{margin:+.2f} mmHg", "above threshold: reversed" if margin > 0 else "below threshold: forward",
              delta_color="off")
    t4.metric("Resulting IJV flow", f"{res.q_ijv:+.2f} mL/s",
              "valve closes at this threshold" if res.valve_closed else ("reversed" if res.q_ijv < 0 else "forward"), delta_color="off")
    st.caption(
        "Why it is exact: when jugular flow is zero there is no viscous drop along the jugular branch, so the collateral must carry "
        "the entire prescribed outflow, Q_total = (P_out,IJV − P_out,collat) / R_collat. Reversal therefore starts exactly when the "
        "outlet difference exceeds Q_total × R_collat, whatever the vein's stiffness, area or the tissue pressure. "
        "R_collat comes from calibration against the pre-flight supine state.")
    with st.expander("Check it numerically with the full solver"):
        chk = cached_threshold_check(active, float(cond["q_total_factor"]), p_oc_in, pk, p_ext_run)
        if chk is not None:
            lines = ["| Vein stiffness K | ΔP_out = threshold − 0.2 mmHg | ΔP_out = threshold | ΔP_out = threshold + 0.2 mmHg |",
                     "|---|---|---|---|"]
            for k_val, qs in chk["rows"]:
                lines.append(f"| {k_val:.2f} mmHg | IJV flow {qs[0]:+.3f} mL/s | **{qs[1]:+.3f} mL/s** | IJV flow {qs[2]:+.3f} mL/s |")
            st.markdown("\n".join(lines))
            st.caption(f"Full 1D solver at the predicted threshold ({chk['threshold']:.3f} mmHg) for the active scenario, with the "
                       f"vein 0.3×, 1× and 3× as stiff. Jugular flow is zero at the threshold for every stiffness, forward below "
                       f"it and reversed above it.")
else:
    st.info("There is no collateral pathway in this configuration, so the jugular vein carries all of the prescribed outflow and cannot reverse.")

# =====================================================================================
# Bottom: regime map over (P_ext, P_out_ijv)
# =====================================================================================
eyebrow("Flow-regime map")
q_factor_map = float(cond["q_total_factor"])
with st.spinner("Computing flow-regime map (cached after the first draw)…"):
    fmap = cached_map(active, q_factor_map, p_oc_nom, pk, map_res)

if fmap is None:
    st.warning("The map needs a calibrated network.")
else:
    thr_map = p_oc_nom + reversal_threshold_mmHg(network, q_factor_map * network.q_total_ref)
    map_title = (f"{BUTTON_LABELS.get(active, cond['label'])}: Q_total {q_factor_map * network.q_total_ref:.2f} mL/s, "
                 f"P_out,collat {p_oc_nom:g} mmHg, g_eff {float(cond['g_eff']):g} held fixed. White lines = solved P_in (mmHg)")
    # The background is computed once and cached; only this single target point is recomputed when a slider moves. Its y is the
    # jugular outlet pressure relative to the nominal collateral outlet, so it sits on the correct side of the threshold line.
    target_x, target_y = res.phase_x_mmHg, res.phase_y_mmHg - (res.p_out_collat - p_oc_nom)
    show_plotly(build_map_figure(fmap, target_x, target_y, map_title, thr_map), "ijv_comply_regime_map")
    here = int(classify_codes(res.velocity_mean_cm_s, res.frac_collapsed, res.ijv_share if math.isfinite(res.ijv_share) else 1.0,
                                 trapped=res.valve_closed))
    st.caption(
        f"Active state ({BUTTON_LABELS.get(active, active)}, P_ext {p_ext_run:g}, P_out,IJV {p_oi_in:g} mmHg): "
        f"**{REGIMES[here][0]}**. Map shares: " + ", ".join(f"{n} {s:.0%}" for (n, _c), s in zip(REGIMES, fmap["shares"])) +
        f". The dashed line is the exact reversal threshold, P_out,IJV = {thr_map:.2f} mmHg; it does not depend on tissue "
        f"pressure or stiffness. Regimes: reversed = mean velocity ≤ −{V_CUT:g} cm/s; stagnant = |mean velocity| < {V_CUT:g} cm/s; "
        f"collapsed = at least {COLLAPSE_FRACTION:.0%} of the vein at P_tm < 0; diverted = forward IJV share < {DIVERSION_SHARE:.0%}; "
        f"otherwise normal; trapped = a competent valve has closed (jugular flow exactly 0). These cut-offs are modeling assumptions."
        + (" The map background is computed for the nominal scenario; the target moves with respiration, overrides and tissue "
           "pressure. Head-down tilt raises P_in but does not move the target." if (shift_ijv or tilt_run is not None) else "")
        + (f" {fmap['fallbacks']} of {fmap['n_solves']} nodes used the 0D fallback." if fmap["fallbacks"] else "")
        + (f" {fmap['failures']} of {fmap['n_solves']} nodes failed and were treated as open." if fmap["failures"] else ""))

# =====================================================================================
# Notes
# =====================================================================================
with st.expander("How to read this"):
    st.markdown(
        "- **Inputs and outputs.** Total cranial outflow is prescribed (autoregulation) and the jugular vein and vertebral "
        "collateral drain to separate outlets. The model solves the split and the cranial venous pressure P_in.\n"
        "- **Jugular reversal has an exact threshold** (equation above). Vein stiffness and tissue pressure do not move it; the "
        "collateral resistance does.\n"
        "- **Collapse flattens the lumen.** A collapsing vein keeps its wall length, so the lumen becomes an ellipse and then a "
        "slit rather than a smaller circle. That multiplies the viscous resistance (the card above shows by how much) and raises "
        "wall shear. The correction is geometric and has no tunable constant.\n"
        "- **Upright diversion is a calibration target, not a prediction.** With the default calibration the default patient "
        "sends a little over half of the upright flow through the collateral, consistent with the vertebral plexus becoming the "
        "main upright pathway (Doepp 2004, via Lan 2021). That result depends on the two calibration assumptions, which only "
        "work together over a narrow range. What the model does predict is the patient dependence: a stiffer or larger vein "
        "keeps more flow in the jugular vein.\n"
        "- **Which outlet rises matters.** A superior vena cava rise raises both outlets, because the azygos system also drains "
        "into it. A rise at the jugular outlet alone corresponds to a more local obstruction, for example of the brachiocephalic vein.\n"
        "- **Two modes.** Guided Narrative walks through four states, one change at a time, with no sidebar; Clinical Sandbox exposes every "
        "control. Mode and step are kept in the page URL, so a refresh does not lose your place.\n"
        "- **The 3D view is a schematic** of the modelled vein and its collateral, not patient anatomy. Jugular radius follows the solved area "
        "profile (x1.6 for visibility). Colour: red = jugular flow exactly zero (valve closed), violet = reversed, amber = stagnant, teal = forward.\n"
        "- **Wall shear stress** is in Pa: station-wise Carreau viscosity times the slit-corrected shear rate, compared with the 0.1-0.6 Pa range "
        "quoted for veins. The 0.4 Pa low-shear figure comes from the arterial literature and is not applied to veins.\n"
        "- **Venous valve switch.** With a competent valve on, a gradient that would reverse the jugular flow closes the valve "
        "instead: Q_IJV is held at exactly 0, the collateral carries all outflow, residence time is infinite and P_in follows in closed "
        "form. The jugular reversal then appears as trapped blood, not backflow. Competence varies widely between people and studies, "
        "so neither setting is 'typical'.\n"
        "- **Respiration.** Deep inspiration lowers and Valsalva raises the jugular outlet pressure by the amounts in 'Model "
        "assumptions'. In reality the thoracic pressure change reaches both outlets; the collateral-share slider sets how much of it "
        "does. With equal shifts the reversal threshold does not move; cranial venous pressure P_in rises instead.\n"
        "- **Head-down tilt.** Raises cranial venous pressure P_in by the hydrostatic head across the vein and distends it slightly. It "
        "acts on both drainage paths equally, so it does not move the flow split or the reversal threshold.\n"
        "- **NASA flowchart messages** map a simulated state with stasis (a closed valve, or |mean velocity| below the cutoff) onto "
        "the OCHMO-MTB-007 decision branch. Reversed flow is not stasis in that brief. This is an illustration, not medical advice.\n"
        "- **Limits:** most of "
        "the jugular branch resistance is the unmodelled outlet region; steady state with no pulsatile or respiratory reversal; "
        "one symmetric vein; rigid lumped collateral; clean elliptical collapse. The 0D fallback uses one uniform vein, so it "
        "overestimates jugular flow in partly collapsed states by roughly a third, and the page says so when it runs."
    )

with st.expander("Model card"):
    st.markdown(
        "**Equations.** Shapiro tube law P_tm = K[(A/A₀)^10 − (A/A₀)^−1.5] (fsolve, brentq fallback, area ≥ 5 % A₀); axial "
        "momentum dP/dx = −8πμ(γ)φ(A)Q/A² + ρg_eff (Poiseuille, inertia neglected; speed index reported) with the slit-lumen "
        "factor φ = (λ + 1/λ)/2 for a constant-perimeter ellipse of aspect ratio λ, and wall shear 4v/r × φ√(A/A₀); Carreau-Yasuda "
        "blood (Cho & Kensey 1991). Network: P_in → R_up → node → {jugular segment + R_term → P_out,IJV} ∥ {R_collat → "
        "P_out,collat}. Closure Q_IJV + (P_out,IJV − P_out,collat + D_IJV)/R_collat = Q_total; P_in = P_node + R_up·Q_total.\n\n"
        "**Calibration.** Pre-flight supine state (P_in, both outlets, reference jugular flow, supine jugular share 0.66, "
        "pressure-drop share 0.6) fixes R_up, R_term and R_collat once per patient. Reversal threshold at reference flow = "
        "drop share × (P_in,ref − P_out,ref) / (1 − jugular share).\n\n"
        "**Added state features (all closed-form).** Valve: Q_IJV = 0 and P_N = P_out,collat + Q_total·R_collat − ρgL when the valve is "
        "competent and Q_total − (P_out,IJV − P_out,collat)/R_collat < 0. Respiration: outlet shifts of −4 (inspiration), 0, +20 (Valsalva) "
        "mmHg by default. Tilt: g_eff = 9.81·sin(angle). No time-stepping solver is used for any of these.\n\n"
        "**Scenario values.** Central venous pressure falls below supine in microgravity (Buckey et al. 1996); cardiac output rises "
        "on long-duration missions (Norsk et al. 2015), so cerebral outflow is held at 1.0×. The brachiocephalic obstruction "
        "scenario is a hypothetical stress case.\n\n"
        "**Status.** Exploratory research prototype; not validated against flight data and not a clinical tool."
    )

with st.expander("Engine diagnostics and patient notes"):
    st.markdown(
        f"- Method used: **{res.method_used.upper()}**" + (f" (fallback: {res.fallback_reason})" if res.fallback_reason else "")
        + (" (closed-form hydrostatic column: valve closed)" if res.method_used == "static" else "") + "\n"
        f"- Stasis: {res.stasis_basis if res.stasis else 'none'} | respiration {res.respiration_phase} ({res.respiration_shift_mmHg:+g} mmHg at the jugular outlet)\n"
        f"- Flow closure error: {res.closure_error:.1e} mL/s | speed index (v/c) max: {res.speed_index_max:.2f}\n"
        f"- Area: mean {res.area_mean_cm2:.2f}, min {res.area_min_cm2:.2f} cm² | P_tm range {res.ptm_min_mmHg:+.1f} to "
        f"{res.ptm_max_mmHg:+.1f} mmHg | collapsed length {res.frac_collapsed:.0%}\n"
        f"- Collapse hydraulics: resistance penalty up to ×{res.shape_penalty_max:.1f}, lumen aspect ratio up to {res.aspect_ratio_max:.0f}:1\n"
        f"- Pressures: node {res.p_node_mmHg:.2f} mmHg, vein outlet {res.p_vein_outlet_mmHg:.2f} mmHg, jugular outlet "
        f"{res.p_out_ijv:g}, collateral outlet {res.p_out_collat:g} mmHg\n"
        f"- Calibration: R_up {network.r_up:.3f}, R_term {network.r_term:.3f}, R_collat {network.r_collat:.3f} mmHg·s/mL; pair drop "
        f"{network.drop_pair_ref:.2f} mmHg (vein segment {network.drop_segment_ref:.3f}) ({network.method.upper()})")
    st.markdown("**Plausibility notes from the engine**")
    for note in patient.validate():
        st.markdown(f"- {note}")
