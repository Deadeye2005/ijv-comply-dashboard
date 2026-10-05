"""
ijv_math_engine.py  (autoregulated outflow, separate outlets, slit-lumen hydraulics)
====================================================================================
Core math engine for IJV-Comply: a steady-state, patient-parameterised model of one internal jugular vein (IJV) in
parallel with a collateral pathway (vertebral venous plexus), each draining to ITS OWN outlet pressure. Pure
numpy/scipy, no UI imports.

Quick start
-----------
    from ijv_math_engine import PatientProfile, calibrate_network, run_scenario

    patient = PatientProfile(A0_cm2=0.88, K_mmHg=0.66, p_ext_mmHg=0.0)     # pre-flight inputs for one astronaut
    network = calibrate_network(patient)                                   # once per patient (cache this)
    res = run_scenario("microgravity_asymmetric_obstruction", patient, network)   # one split solve (~5-50 ms)
    res.q_ijv, res.velocity_mean_cm_s, res.p_in, res.regime_flags

Physics
-------
x runs from the inlet (x = 0, skull base) to the jugular outlet (x = L); Q > 0 is forward (towards the heart).
Units: mmHg, mL/s, mmHg*s/mL, cm^2, cm.

  Transmural pressure     P_tm = P - P_ext
  Shapiro tube law        P_tm = K [ (A/A0)^n - (A/A0)^m ]        inverted numerically; A >= 5 % of A0
  Continuity / shear      v = Q / A,   gamma = 4 |v| / r,   r = sqrt(A / pi)
  Axial momentum (1D)     dP/dx = -8 pi mu(gamma) Q / A^2 + rho g_eff   (inertia neglected; see speed_index)

Network
-------
    P_in --[ R_up ]--> node N --+--[ IJV segment (1D) ]--[ R_term ]--> P_out_ijv     (brachiocephalic side)
                                +--[ R_collat ]------------------------> P_out_collat  (azygos/vertebral side)

  Inputs:   Q_total (prescribed; autoregulation), P_out_ijv, P_out_collat, g_eff, P_ext, patient A0 and K.
  Unknowns: the split Q_IJV / Q_collat (Q_IJV may be NEGATIVE), the vein profile, and P_in.

Both branches start at node N and span the same height, so with D_ijv(Q) = P_N + rho g L - P_out_ijv (the IJV branch
drop, signed like Q_IJV) the collateral carries Q_collat = (D_ijv + P_out_ijv - P_out_collat) / R_collat. Integrating
the momentum balance BACKWARD from the jugular outlet gives D_ijv(Q_IJV) in one pass, and the split closes with

        Q_IJV + (P_out_ijv - P_out_collat + D_ijv(Q_IJV)) / R_collat = Q_total.

With s = Q_total - (P_out_ijv - P_out_collat) / R_collat, the residual is -s at Q_IJV = 0 and has the opposite sign
at Q_IJV = s, so brentq always has the bracket [min(0, s), max(0, s)]. Then P_in = P_N + R_up * Q_total.

Exact reversal threshold
------------------------
At zero IJV flow there is no viscous drop in the IJV branch, so P_N = P_out_ijv - rho g L and the collateral carries
everything:   Q_IJV = 0   <=>   P_out_ijv - P_out_collat = Q_total * R_collat.
This holds for ANY vein stiffness, area or tissue pressure. The IJV reverses when its outlet pressure exceeds the
collateral outlet by more than Q_total * R_collat.

Valve, respiration, tilt and the NASA flowchart (all closed-form; no time stepping)
-----------------------------------------------------------------------------------
* Venous valve (PatientProfile.valve_competent). A competent valve blocks retrograde flow. The unconstrained solution
  reverses exactly when s = Q_total - (P_out,IJV - P_out,collat)/R_collat < 0 (the root of the closure lies on the side of
  zero where s lies), so the engine tests that sign FIRST. If the valve is competent and s < 0, Q_IJV is forced to 0, the
  collateral carries all of Q_total, and the node pressure follows algebraically: P_N = P_out,collat + Q_total R_collat
  - rho g L (a hydrostatic column in the isolated vein). No root-finder or ODE is run for the closed state.
  Residence time is infinite (no exchange). P_in is continuous across the closure threshold.
  Prevalence (searched, not assumed): a valve is present in about 90 % of people (Lepori; Valecchi; Silva 2002), but
  competence on Valsalva varies by study: insufficiency in 29 % of valves in 50 healthy volunteers (pressure-controlled
  Valsalva), about 40-51 % in other healthy groups, and about 90 % in one colour-Doppler series. Hence no default.
* Respiration (apply_respiration). A thoracic-pump phase shifts the jugular outlet pressure by a set amount: deep
  inspiration -4 mmHg (right atrial pressure 'falls by several mmHg'), resting 0, Valsalva +20 mmHg. A standard 40 mmHg
  strain raised central venous pressure by about 40 mmHg (J Appl Physiol 2000), so +20 is a moderate strain. The shift is
  applied to the jugular outlet; `collateral_coupling` (0-1) sets the fraction also applied to the collateral outlet. In
  reality the thoracic pressure change reaches both outlets (coupling near 1); with equal shifts the reversal threshold does
  not move, only P_in and the vein's distension do. The default 0 follows the project specification.
* Head-down tilt (effective_gravity_for_tilt). g_eff = 9.81 sin(tilt); 0 deg = supine, -6 deg = NASA head-down bed rest.
  Hydrostatic head acts on both parallel branches equally and cancels in the zero-flow condition, so tilt raises P_in and
  distends the vein (about +1.2 mmHg across a 15 cm vein at -6 deg) but does NOT move the reversal threshold or the flow split.
* NASA OCHMO-MTB-007 flowchart (nasa_protocol). For a simulated state with stasis (valve-closed Q_IJV = 0, or |mean velocity|
  below the stagnation cutoff): any of family history, thrombophilia or high-risk hormones -> prophylaxis message, otherwise
  increased monitoring and hydration. Reversed flow is not 'stasis' in the brief ('not considered a main contributor').
  Reference mapping of a model state onto a flowchart; research prototype, not medical advice.

Slit-lumen hydraulics
---------------------
A collapsing vein does not shrink into a smaller circle: its wall barely stretches, so the perimeter stays at 2*pi*r0
and the lumen flattens into an ellipse, then a slit. For a constant-perimeter ellipse of the same area the viscous
resistance is higher than the circular-equivalent value by phi = (lambda + 1/lambda)/2 (lambda = aspect ratio), and the
mean wall shear rate is higher by phi * sqrt(A/A0). The factor has no tunable constant and is exactly 1 for A >= A0:
A/A0 = 0.9 -> 1.15x, 0.5 -> 2.3x, 0.26 -> 4.6x, 0.10 -> 12x, 0.05 (the clamp) -> 25x (see slit_factors). It enters the
momentum balance, the 0D fallback, the shear rate and the shear-dependent viscosity. PatientProfile.lumen_model =
'circular' switches it off, which reproduces the previous circular-equivalent engine exactly.

Calibration and its two assumptions
-----------------------------------
The pre-flight SUPINE state (g = 0, both outlets at p_out_ref, P_in = p_in_ref, IJV flow q_ref, IJV share) fixes the
resistances. The supine vein is open (A >= A0), so calibration is unaffected by the slit correction.

1. ijv_share_ref = 0.66, the supine jugular share of cranial venous drainage (Doepp et al. 2004, as cited by Lan et al.
   2021, who also note that the vertebral plexus is the main pathway upright).
2. parallel_drop_fraction = 0.6, the fraction of the supine P_in -> P_out drop across the IJV/collateral pair. With a
   shared outlet only the RATIO of IJV to collateral resistance mattered; with separate outlets the ABSOLUTE resistance
   sets the reversal threshold. A bare 15 cm vein drops only ~0.08 mmHg, which would put the threshold at ~0.4 mmHg and
   the backflow at ~200 mL/s for a 13 mmHg outlet difference, so the pair is calibrated to a stated share of the drop.
   The IJV branch reaches it through a lumped terminal resistance R_term (jugular outlet/valve region) in series with
   the 1D segment; R_collat carries the same drop at the collateral reference flow. Threshold (q_total_factor 1):

        dP_reversal = parallel_drop_fraction * (p_in_ref - p_out_ref) / (1 - ijv_share_ref) = 0.6 * 3 / 0.34 = 5.3 mmHg

How the defaults were chosen, and what they do not prove
--------------------------------------------------------
Upright diversion and the size of the reversal both depend on R_collat: a low R_collat diverts upright flow to the
collateral but makes the reversal violent, and the reverse. With the slit correction there is a band of
(ijv_share_ref, parallel_drop_fraction) where BOTH (a) the default patient's upright jugular share is below 50 % and (b)
the high-outlet reversal flow does not exceed the total outflow. At the literature share of 0.66 that band is
parallel_drop_fraction 0.59-0.65; the default 0.6 gives upright 48 % jugular and reversal -8.5 mL/s against 8.8 mL/s
total. The band is narrow, and upright diversion is therefore a CALIBRATION TARGET, not an independent prediction.
What is a prediction: how it varies with the patient (a stiffer or larger vein keeps more flow in the jugular vein).

Limits worth stating
--------------------
* The venous valve is a switch (default OFF = incompetent or absent). Competence varies widely between people and studies.
* R_term, the unmodelled jugular outlet resistance, carries ~95 % of the jugular branch resistance when the vein is open.
* A pressure rise in the superior vena cava raises BOTH outlets (the azygos system also drains to it). A rise at the
  jugular outlet alone corresponds to a more local obstruction, e.g. of the brachiocephalic vein.
* The slit idealisation assumes a clean ellipse at constant perimeter; real veins may buckle or fold.
* Steady state: no pulsatile or respiratory reversal.

If the 1D integration fails or returns something non-physical, the engine falls back to a lumped (0D) uniform-vein
model with the same closure; `method_used` and `fallback_reason` report which path ran.

Scientific status: exploratory, literature-parameterised, steady-state, not validated against flight data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve
from scipy.special import ellipe

__all__ = [
    "PatientProfile", "ScenarioError", "NetworkCalibration", "SteadyStateResult", "RegimeThresholds",
    "TubeLaw", "build_tube_law", "solve_area", "effective_viscosity",
    "get_scenario_conditions", "list_scenarios", "calibrate_network", "solve_steady_state", "run_scenario",
    "classify_regime", "reversal_threshold_mmHg", "slit_factors",
    "G_EARTH", "RESPIRATION_PHASES", "RESPIRATION_SHIFT_MMHG", "effective_gravity_for_tilt", "respiration_shifts",
    "apply_respiration", "nasa_protocol", "NASA_PROPHYLAXIS_TEXT", "NASA_MONITORING_TEXT",
]

# ----------------------------------------------------------------------------------------------------------
# Constants and safeguards
# ----------------------------------------------------------------------------------------------------------
MMHG_TO_PA = 133.322          # Pa per mmHg
RHO_BLOOD = 1060.0            # kg/m^3
A_MIN_FRACTION = 0.05         # area is never allowed below 5 % of A0 (prevents division by zero and 1/A^n blow-ups)
ALPHA_MAX = 3.0               # table upper bound on A/A0 (3 x A0)
DEFAULT_N = 10.0              # Shapiro (1977) distension exponent: structural constant, overridable per patient
DEFAULT_M = -1.5              # Shapiro (1977) collapse exponent: structural constant, overridable per patient
GAMMA_FLOOR = 1e-3            # s^-1, shear floor inside the viscosity law (zero-flow states return a finite value)
Q_FLOOR = 1e-6                # mL/s, flow floor used in residence-time and share ratios
CARREAU = dict(mu0=0.056, mu_inf=0.00345, lam=3.313, n_c=0.3568, a=2.0)   # Cho & Kensey (1991), Pa*s, s
NEWTONIAN_MU = 0.004          # Pa*s
G_EARTH = 9.81                # m/s^2
ODE_RTOL, ODE_ATOL = 1e-7, 1e-9   # RK45 tolerances; tighter values change the pressure drop by < 1e-6 mmHg (benchmarked)
ROOT_TOL = 1e-9               # brentq tolerance on the IJV flow (mL/s); the root is identical to ~8 digits at 1e-8


class ScenarioError(ValueError):
    """Raised for unknown scenario names or an uncalibratable reference state."""


# ----------------------------------------------------------------------------------------------------------
# Slit-lumen hydraulics: a collapsing vein is not a smaller circle
# ----------------------------------------------------------------------------------------------------------
# The wall is nearly inextensible, so as the vein collapses its perimeter stays at the unstressed value 2*pi*r0
# (A0 = pi*r0^2) and the lumen flattens into an ellipse of semi-axes a >= b with a*b = A (a slit as b -> 0).
# With w = (b/a)^2 and E the complete elliptic integral of the second kind:
#       perimeter  P = 4 a E(1 - w) = 2 pi r0          =>   A/A0 = [pi / (2 E(1 - w))]^2 * sqrt(w)
# Poiseuille flow in an ellipse gives R = 4 mu L (a^2 + b^2) / (pi a^3 b^3), versus 8 pi mu L / A^2 for a circle of the
# SAME area, so the resistance penalty is
#       phi = R_ellipse / R_circle = (lambda + 1/lambda) / 2 = (1 + w) / (2 sqrt(w)),      lambda = a/b.
# Mean wall shear = tau_wall / mu with tau_wall = dP * A / (P * L), which gives the shear-rate factor phi * sqrt(A/A0)
# relative to the circular 4v/r. For A >= A0 the lumen stays circular (phi = 1). The correction has no tunable
# constant; for A/A0 << 1 it tends to phi ~ pi^2 / (8 A/A0). Examples: A/A0 = 0.9 -> 1.15x, 0.5 -> 2.3x, 0.26 -> 4.6x,
# 0.10 -> 12x, 0.05 (the clamp) -> 25x. Idealised: real collapsed veins may buckle or fold rather than form a clean ellipse.
def _build_slit_table(n: int = 4000):
    w = np.geomspace(1e-8, 1.0, n)                                     # w = (b/a)^2, ascending
    alpha = (math.pi / (2.0 * ellipe(1.0 - w))) ** 2 * np.sqrt(w)      # A / A0, ascending
    phi = (1.0 + w) / (2.0 * np.sqrt(w))
    return alpha, phi, phi * np.sqrt(alpha)


_SLIT_ALPHA, _SLIT_PHI, _SLIT_SHEAR = _build_slit_table()


def slit_factors(alpha):
    """(resistance penalty phi, wall-shear factor) for a constant-perimeter elliptical lumen at A/A0 = alpha; vectorised.
    Both are exactly 1 for alpha >= 1 (circular, or stretched circular)."""
    a = np.clip(np.asarray(alpha, dtype=float), _SLIT_ALPHA[0], _SLIT_ALPHA[-1])
    return np.interp(a, _SLIT_ALPHA, _SLIT_PHI), np.interp(a, _SLIT_ALPHA, _SLIT_SHEAR)


# ----------------------------------------------------------------------------------------------------------
# 1. Scenario conditions: prescribed outflow, two outlet pressures, axial gravity
# ----------------------------------------------------------------------------------------------------------
# Literature-anchored values: central venous pressure FALLS below supine in microgravity (Buckey et al. 1996,
# ~2 mmHg in flight) and cardiac output RISES on the ISS (Norsk et al. 2015), so cerebral outflow is held at 1.0x
# (autoregulation). g_eff is the axial component of gravity along inlet -> outlet: 0 supine and in microgravity,
# 9.81 upright. P_in is not an input; the solver computes it.
_SCENARIOS: Dict[str, Dict[str, object]] = {
    "supine_1g": dict(
        label="Supine (1 g)", q_total_factor=1.0, p_out_ijv=5.0, p_out_collat=5.0, g_eff=0.0,
        description="Horizontal posture; the pre-flight reference state used for calibration."),
    "upright_1g": dict(
        label="Upright (1 g)", q_total_factor=1.0, p_out_ijv=3.0, p_out_collat=3.0, g_eff=9.81,
        description="Standing. Autoregulation holds total outflow; the hydrostatic column collapses the upper vein."),
    "microgravity_0g": dict(
        label="Microgravity (0 g)", q_total_factor=1.0, p_out_ijv=2.0, p_out_collat=2.0, g_eff=0.0,
        description="Central venous pressure falls below supine in flight (Buckey 1996); outflow held by autoregulation."),
    "microgravity_asymmetric_obstruction": dict(
        label="Microgravity + Brachiocephalic Obstruction", q_total_factor=1.0, p_out_ijv=15.0, p_out_collat=2.0, g_eff=0.0,
        description="A localized pressure spike at the jugular outlet (e.g., brachiocephalic vein compression). Because it "
                    "does not affect the collateral outlet, it forces the autoregulated flow to stall and reverse."),
}
_ALIASES = {
    "supine": "supine_1g", "upright": "upright_1g", "standing": "upright_1g",
    "microgravity": "microgravity_0g", "0g": "microgravity_0g", "zero_g": "microgravity_0g",
    "asymmetric_obstruction": "microgravity_asymmetric_obstruction",
    "brachiocephalic_obstruction": "microgravity_asymmetric_obstruction",
    "microgravity_brachiocephalic_obstruction": "microgravity_asymmetric_obstruction",
    # previous names of the fourth scenario
    "microgravity_stiff_high_cvp": "microgravity_asymmetric_obstruction", "microgravity_stiff_wall_high_cvp": "microgravity_asymmetric_obstruction",
    "stiff_high_cvp": "microgravity_asymmetric_obstruction", "high_cvp": "microgravity_asymmetric_obstruction",
}


def _scenario_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    key = _ALIASES.get(key, key)
    if key not in _SCENARIOS:
        raise ScenarioError(f"Unknown scenario '{name}'. Available: {', '.join(_SCENARIOS)}.")
    return key


def list_scenarios() -> List[str]:
    """Canonical scenario keys, in presentation order."""
    return list(_SCENARIOS)


def get_scenario_conditions(scenario_name: str) -> Dict[str, object]:
    """
    Scenario inputs: q_total_factor (x patient reference total outflow), p_out_ijv and p_out_collat (mmHg), g_eff
    (m/s^2), plus key, label, description. Returns a new dict. Names are forgiving ('Microgravity (0G)', '0g').
    """
    key = _scenario_key(scenario_name)
    out = dict(_SCENARIOS[key])
    out["key"] = key
    return out


# ----------------------------------------------------------------------------------------------------------
# 2. Patient-specific parameters (nothing patient-specific is hardcoded globally)
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PatientProfile:
    """
    Inputs that describe ONE astronaut. A0_cm2, K_mmHg and p_ext_mmHg are required (no defaults) so they can never
    be silently inherited from a global constant. Use PatientProfile.population_placeholder() for demos.

    A0_cm2                 Reference area at zero transmural pressure (cm^2). NOT the measured supine area.
    K_mmHg                 Shapiro wall stiffness (mmHg).
    p_ext_mmHg             Perivascular tissue pressure (mmHg), used for every scenario unless overridden below.
    p_ext_by_scenario      Optional per-scenario overrides, e.g. {"microgravity_0g": 4.0} (edema hypothesis).
    q_ref_ijv_mL_s         Pre-flight supine IJV Doppler flow (mL/s).
    ijv_share_ref          IJV fraction of total cranial outflow supine (None = no collateral); default 0.66 (Doepp 2004).
    p_in_ref_mmHg          Pre-flight supine cranial venous (inlet) pressure for calibration (default 8).
    p_out_ref_mmHg         Pre-flight supine outlet pressure, both outlets (default 5; keep equal to the supine
                           scenario's outlets so the supine scenario reproduces the calibration exactly).
    parallel_drop_fraction Fraction of the supine P_in -> P_out drop occurring across the IJV/collateral pair
                           (0 < f < 1; ASSUMPTION, default 0.6). None = bare calibration (pair drop = vein drop).
    lumen_model            'slit' (default): collapsing lumen flattens into a constant-perimeter ellipse, raising
                           resistance and wall shear (see slit_factors). 'circular': circular-equivalent lumen.
    valve_competent        True: a competent jugular valve blocks retrograde flow (Q_IJV forced to 0 when the pressure
                           gradient would reverse it). False (default): no valve, or an incompetent one.
    sources                Free-text provenance per field ('pre-flight US', 'tilt-table', 'assumed').
    """
    A0_cm2: float
    K_mmHg: float
    p_ext_mmHg: float
    n: float = DEFAULT_N
    m: float = DEFAULT_M
    L_cm: float = 15.0
    q_ref_ijv_mL_s: float = 5.8
    ijv_share_ref: Optional[float] = 0.66
    p_in_ref_mmHg: float = 8.0
    p_out_ref_mmHg: float = 5.0
    parallel_drop_fraction: Optional[float] = 0.6
    rheology: str = "carreau"
    lumen_model: str = "slit"
    valve_competent: bool = False
    p_ext_by_scenario: Mapping[str, float] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("A0_cm2", "K_mmHg", "L_cm", "q_ref_ijv_mL_s"):
            val = getattr(self, name)
            if not (isinstance(val, (int, float)) and val > 0 and math.isfinite(val)):
                raise ValueError(f"{name} must be a positive finite number, got {val!r}.")
        for name in ("p_ext_mmHg", "p_in_ref_mmHg", "p_out_ref_mmHg"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.p_in_ref_mmHg <= self.p_out_ref_mmHg:
            raise ValueError("p_in_ref_mmHg must exceed p_out_ref_mmHg (the reference state needs forward flow).")
        if self.parallel_drop_fraction is not None and not (0.0 < self.parallel_drop_fraction < 1.0):
            raise ValueError("parallel_drop_fraction must be in (0, 1), or None.")
        if self.m >= 0 or self.n <= 0:
            raise ValueError("Shapiro exponents need n > 0 and m < 0 so the tube law is monotonic.")
        if self.ijv_share_ref is not None and not (0.0 < self.ijv_share_ref <= 1.0):
            raise ValueError("ijv_share_ref must be in (0, 1], or None for no collateral pathway.")
        if self.rheology.lower() not in ("newtonian", "carreau"):
            raise ValueError("rheology must be 'newtonian' or 'carreau'.")
        if not isinstance(self.valve_competent, (bool, np.bool_)):
            raise ValueError("valve_competent must be True or False.")
        if self.lumen_model not in ("slit", "circular"):
            raise ValueError("lumen_model must be 'slit' (constant-perimeter ellipse) or 'circular'.")

    @classmethod
    def population_placeholder(cls, **overrides) -> "PatientProfile":
        """Population-average PLACEHOLDERS (A0 and K from the project's literature table). Not a real patient."""
        base = dict(A0_cm2=0.88, K_mmHg=0.66, p_ext_mmHg=0.0,
                    sources={"A0_cm2": "placeholder (Lan 2021 MRI supine area)", "K_mmHg": "placeholder (back-calculated)",
                             "p_ext_mmHg": "assumed", "q_ref_ijv_mL_s": "assumed", "parallel_drop_fraction": "assumed"})
        base.update(overrides)
        return cls(**base)

    def p_ext_for(self, scenario: Optional[str]) -> float:
        if scenario is not None:
            try:
                key = _scenario_key(scenario)
            except ScenarioError:
                return float(self.p_ext_mmHg)
            if key in self.p_ext_by_scenario:
                return float(self.p_ext_by_scenario[key])
        return float(self.p_ext_mmHg)

    def with_stiffness_factor(self, factor: float) -> "PatientProfile":
        """Copy with K multiplied by `factor`, e.g. to test how the result depends on vein stiffness."""
        return replace(self, K_mmHg=self.K_mmHg * factor)

    def validate(self) -> List[str]:
        """Plausibility warnings (never raises for merely unusual values)."""
        w = []
        if not 0.2 <= self.A0_cm2 <= 3.0:
            w.append(f"A0 = {self.A0_cm2} cm^2 is outside the usual 0.2-3.0 cm^2 IJV range.")
        if not 0.05 <= self.K_mmHg <= 10.0:
            w.append(f"K = {self.K_mmHg} mmHg is outside the 0.05-10 mmHg range explored here.")
        if not 8.0 <= self.L_cm <= 25.0:
            w.append(f"L = {self.L_cm} cm is unusual for an extracranial IJV.")
        if self.q_ref_ijv_mL_s > 15:
            w.append("Reference IJV flow exceeds typical whole-brain flow (~12 mL/s).")
        if "A0_cm2" not in self.sources or "K_mmHg" not in self.sources:
            w.append("Provenance for A0 and K is not recorded in `sources`.")
        if self.ijv_share_ref is not None:
            implied = self.q_ref_ijv_mL_s / self.ijv_share_ref
            if 2.0 * implied > 15.0:
                w.append(f"Implied total outflow is {implied:.1f} mL/s for this vein's side ({2.0 * implied:.1f} mL/s for both sides), "
                         f"above typical whole-brain flow of about 12 mL/s. Lower the reference jugular flow if this vein is not "
                         f"representative; scenarios scale this total.")
        w.append("A0 and K are not separately identifiable from ultrasound area alone: P_tm needs the unmeasured P_ext.")
        w.append("parallel_drop_fraction and ijv_share_ref are calibration assumptions; they set the collateral resistance and "
                 "so the jugular reversal threshold.")
        if self.lumen_model == "slit":
            w.append("Collapse hydraulics assume a clean constant-perimeter elliptical lumen; real veins may buckle or fold.")
        return w


# ----------------------------------------------------------------------------------------------------------
# 3. Blood rheology
# ----------------------------------------------------------------------------------------------------------
def effective_viscosity(gamma, model: str = "carreau"):
    """
    Apparent viscosity (Pa*s) at wall shear rate gamma (1/s); vectorised.

    'carreau' (Carreau-Yasuda, a = 2): mu_inf + (mu0 - mu_inf) [1 + (lam gamma)^a]^((n - 1)/a), whole-blood fit of
    Cho & Kensey (1991); bounded at mu0 = 56 mPa*s for gamma -> 0. 'newtonian': constant 4 mPa*s.
    """
    g = np.maximum(np.asarray(gamma, dtype=float), GAMMA_FLOOR)
    if model.lower() == "newtonian":
        return np.full_like(g, NEWTONIAN_MU)
    c = CARREAU
    return c["mu_inf"] + (c["mu0"] - c["mu_inf"]) * (1.0 + (c["lam"] * g) ** c["a"]) ** ((c["n_c"] - 1.0) / c["a"])


# ----------------------------------------------------------------------------------------------------------
# 4. Shapiro tube law and its numerical inversion
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class TubeLaw:
    """Shapiro tube law P_tm = K [ (A/A0)^n - (A/A0)^m ] with a lookup table for fast vectorised inversion."""
    A0: float
    K: float
    n: float
    m: float
    alpha: np.ndarray
    ptm: np.ndarray
    slit: bool = True               # True: constant-perimeter elliptical lumen when collapsed; False: circular equivalent

    def shape_factors(self, area_cm2):
        """(resistance penalty, wall-shear factor) at this area; (1, 1) when the lumen model is circular."""
        a = np.asarray(area_cm2, dtype=float)
        if not self.slit:
            return np.ones_like(a), np.ones_like(a)
        return slit_factors(a / self.A0)

    def forward(self, alpha):
        a = np.asarray(alpha, dtype=float)
        return self.K * (a ** self.n - a ** self.m)

    def area(self, ptm):
        """Area (cm^2) for P_tm (mmHg), vectorised via the table; clamped to [5 % A0, 3 A0]."""
        alpha = np.interp(np.asarray(ptm, dtype=float), self.ptm, self.alpha)
        return np.clip(alpha, A_MIN_FRACTION, ALPHA_MAX) * self.A0

    def area_fsolve(self, ptm: float) -> float:
        """Area (cm^2) for one P_tm by fsolve in u = ln(A/A0), with a brentq fallback if fsolve is not verified."""
        ptm = float(ptm)
        if ptm <= self.ptm[0]:
            return A_MIN_FRACTION * self.A0
        if ptm >= self.ptm[-1]:
            return ALPHA_MAX * self.A0
        f = lambda u: self.K * (math.exp(self.n * u) - math.exp(self.m * u)) - ptm
        u0 = math.log(float(np.interp(ptm, self.ptm, self.alpha)))
        sol, _info, ier, _msg = fsolve(lambda u: f(float(u[0])), [u0], full_output=True, xtol=1e-12)
        alpha = math.exp(float(sol[0]))
        converged = ier == 1 and abs(f(float(sol[0]))) < 1e-9 * (1.0 + abs(ptm)) and A_MIN_FRACTION <= alpha <= ALPHA_MAX
        if not converged:
            alpha = brentq(lambda a: self.K * (a ** self.n - a ** self.m) - ptm, A_MIN_FRACTION, ALPHA_MAX, xtol=1e-13)
        return alpha * self.A0

    def speed_index(self, alpha, velocity_cm_s):
        """|v| / c with c^2 = (alpha dP/dalpha) / rho, the elastic wave speed. Near 1, flow limitation (choking) occurs."""
        a = np.maximum(np.asarray(alpha, dtype=float), A_MIN_FRACTION)
        c2 = self.K * MMHG_TO_PA * (self.n * a ** self.n - self.m * a ** self.m) / RHO_BLOOD
        return np.abs(np.asarray(velocity_cm_s, dtype=float)) / 100.0 / np.sqrt(c2)


def build_tube_law(A0_cm2: float, K_mmHg: float, n: float = DEFAULT_N, m: float = DEFAULT_M, n_grid: int = 6000,
                   lumen: str = "slit") -> TubeLaw:
    """Build the tube law and its inversion table. A0 and K are arguments, never globals. lumen: 'slit' or 'circular'."""
    if lumen not in ("slit", "circular"):
        raise ValueError("lumen must be 'slit' or 'circular'.")
    alpha = np.geomspace(A_MIN_FRACTION, ALPHA_MAX, n_grid)
    ptm = K_mmHg * (alpha ** n - alpha ** m)
    if not np.all(np.diff(ptm) > 0):
        raise ValueError("Tube law is not monotonic for these exponents.")
    return TubeLaw(A0=A0_cm2, K=K_mmHg, n=n, m=m, alpha=alpha, ptm=ptm, slit=(lumen == 'slit'))


def solve_area(ptm_mmHg: float, A0_cm2: float, K_mmHg: float, n: float = DEFAULT_N, m: float = DEFAULT_M) -> float:
    """Stand-alone helper: area (cm^2) for a transmural pressure, given patient A0 and K."""
    return build_tube_law(A0_cm2, K_mmHg, n, m).area_fsolve(ptm_mmHg)


# ----------------------------------------------------------------------------------------------------------
# 5. Vein models: 1D (backward ODE) and 0D (lumped algebraic fallback). Unchanged from the previous engine.
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _VeinState:
    """Internal: vein profile for a trial IJV flow. Arrays run inlet (index 0) to outlet (last)."""
    x_cm: np.ndarray
    p: np.ndarray               # mmHg
    area: np.ndarray            # cm^2
    drop: float                 # D = P_N + rho g L - P_out  (viscous drop, mmHg, signed like Q)
    method: str


def _hyd_mmHg_per_m(g_eff: float) -> float:
    return RHO_BLOOD * g_eff / MMHG_TO_PA


def _vein_1d(q, p_out, p_ext, g_eff, law, L_cm, rheology, n_pts) -> _VeinState:
    """Integrate dP/ds = 8 pi mu(gamma) phi(A) Q / A^2 - rho g from the outlet upstream (s = L - x)."""
    length = L_cm / 100.0
    q_m3 = q * 1e-6
    hyd = _hyd_mmHg_per_m(g_eff)

    def rhs(_s, y):
        area_cm2 = float(law.area(y[0] - p_ext))
        area_m2 = area_cm2 * 1e-4
        phi, sfac = law.shape_factors(area_cm2)             # slit-lumen resistance penalty and wall-shear factor
        r = math.sqrt(area_m2 / math.pi)
        gamma = 4.0 * abs(q_m3) / (area_m2 * r) * float(sfac)
        mu = float(effective_viscosity(gamma, rheology))
        return [8.0 * math.pi * mu * q_m3 * float(phi) / area_m2 ** 2 / MMHG_TO_PA - hyd]

    s_eval = np.linspace(0.0, length, n_pts)
    sol = solve_ivp(rhs, (0.0, length), [p_out], t_eval=s_eval, method="RK45", rtol=ODE_RTOL, atol=ODE_ATOL)
    if not sol.success or sol.y.shape[1] != n_pts:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    p = sol.y[0][::-1]
    if not np.all(np.isfinite(p)):
        raise RuntimeError("ODE returned non-finite pressures.")
    area = law.area(p - p_ext)
    drop = float(p[0] - p_out + hyd * length)
    return _VeinState(x_cm=np.linspace(0.0, L_cm, n_pts), p=p, area=area, drop=drop, method="1d")


def _vein_0d(q, p_out, p_ext, g_eff, law, L_cm, rheology, n_pts) -> _VeinState:
    """
    Lumped uniform-vein fallback. One area A set by the MEAN transmural pressure (P_N + P_out)/2 - P_ext; viscous drop
    D = 8 pi mu(gamma) phi(A) L Q / A^2, solved for the SMALLEST (open-vein) root with brentq. Cannot represent partial
    collapse, and cannot represent a partly collapsed head end under strong reversed flow.
    """
    length = L_cm / 100.0
    q_m3 = q * 1e-6
    hyd = _hyd_mmHg_per_m(g_eff)

    def drop_for_area(area_cm2):
        area_c = max(float(area_cm2), A_MIN_FRACTION * law.A0)
        area_m2 = area_c * 1e-4
        phi, sfac = law.shape_factors(area_c)
        r = math.sqrt(area_m2 / math.pi)
        mu = float(effective_viscosity(4.0 * abs(q_m3) / (area_m2 * r) * float(sfac), rheology))
        return 8.0 * math.pi * mu * length * q_m3 * float(phi) / area_m2 ** 2 / MMHG_TO_PA

    d_max = abs(drop_for_area(A_MIN_FRACTION * law.A0))
    if d_max == 0.0:
        d_val = 0.0
    else:
        def g(d_abs):
            d = math.copysign(d_abs, q)
            p_n = p_out - hyd * length + d
            return d_abs - abs(drop_for_area(float(law.area(0.5 * (p_n + p_out) - p_ext))))
        # g(0) < 0. For forward flow g is increasing, so the root is unique. For REVERSED flow a larger drop lowers the
        # mean pressure and collapses the vein, which raises the drop again: g can have a second, spurious root where the
        # vein has collapsed under its own suction (area at the clamp, a drop of d_max). The physical solution is the
        # SMALLEST root (the open vein, the branch the 1D solver also follows), so bracket upward from zero by doubling
        # and take the first sign change.
        d_lo, d_hi = 0.0, min(1e-4, d_max)
        while d_hi < d_max and g(d_hi) <= 0.0:
            d_lo, d_hi = d_hi, min(2.0 * d_hi, d_max)
        d_abs = brentq(g, d_lo, d_hi * (1.0 + 1e-9) + 1e-15, xtol=1e-13)
        d_val = math.copysign(d_abs, q)
    p_n = p_out - hyd * length + d_val
    area_u = float(law.area(0.5 * (p_n + p_out) - p_ext))
    return _VeinState(x_cm=np.linspace(0.0, L_cm, n_pts), p=np.linspace(p_n, p_out, n_pts),
                      area=np.full(n_pts, area_u), drop=float(d_val), method="0d")


def _vein(method, q, p_out, p_ext, g_eff, law, L_cm, rheology, n_pts) -> _VeinState:
    return (_vein_1d if method == "1d" else _vein_0d)(q, p_out, p_ext, g_eff, law, L_cm, rheology, n_pts)


# ----------------------------------------------------------------------------------------------------------
# 6. Network calibration (patient-specific, done once)
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NetworkCalibration:
    """Fixed resistances derived from the patient's pre-flight supine reference state."""
    r_up: float                       # upstream cerebral venous resistance, mmHg*s/mL (sets P_in only)
    g_collat: float                   # collateral conductance = 1/R_collat, mL/s per mmHg; 0 = no collateral
    r_term: float                     # lumped IJV terminal resistance (outlet/valve region), mmHg*s/mL
    q_total_ref: float                # reference total cranial outflow, mL/s (scenarios scale this)
    q_ijv_ref: float
    drop_pair_ref: float              # pressure drop across the IJV/collateral pair at the reference state, mmHg
    drop_segment_ref: float           # part of it across the 1D vein segment, mmHg
    parallel_drop_fraction: Optional[float]
    p_in_ref: float
    p_out_ref: float
    method: str                       # '1d' or '0d' used for the calibration solves

    @property
    def r_collat(self) -> float:
        return 1.0 / self.g_collat if self.g_collat > 0 else math.inf


def reversal_threshold_mmHg(network: NetworkCalibration, q_total: float) -> float:
    """
    Outlet-pressure difference P_out_ijv - P_out_collat at which IJV flow is exactly zero: Q_total * R_collat.
    Exact for any vein stiffness, area or tissue pressure (no viscous drop at zero flow). Infinite with no collateral.
    """
    return q_total / network.g_collat if network.g_collat > 0 else math.inf


def calibrate_network(patient: PatientProfile, method: str = "auto", n_pts: int = 61) -> NetworkCalibration:
    """
    Fix R_up, R_term and R_collat from the pre-flight SUPINE state (g = 0, both outlets at p_out_ref, P_in = p_in_ref,
    IJV flow q_ref, IJV share ijv_share_ref):

        D_pair   = parallel_drop_fraction * (p_in_ref - p_out_ref)      (or the bare vein drop if None)
        R_term   : R_term * q_ref + D_segment(q_ref) = D_pair            (brentq on a guaranteed bracket)
        R_collat = D_pair / Q_collat_ref
        R_up     = (p_in_ref - p_out_ref - D_pair) / Q_total_ref

    Call once per patient with the PRE-FLIGHT P_ext; reuse it across scenarios and P_ext sweeps.
    """
    law = build_tube_law(patient.A0_cm2, patient.K_mmHg, patient.n, patient.m, lumen=patient.lumen_model)
    p_ext = patient.p_ext_for("supine_1g")
    q_ref, p_out = patient.q_ref_ijv_mL_s, patient.p_out_ref_mmHg
    gradient = patient.p_in_ref_mmHg - p_out

    def seg_drop(m_: str, r_t: float) -> float:
        return _vein(m_, q_ref, p_out + r_t * q_ref, p_ext, 0.0, law, patient.L_cm, patient.rheology, n_pts).drop

    used, d_seg0 = None, None
    for m_try in (("1d", "0d") if method == "auto" else (method,)):
        try:
            d_seg0, used = seg_drop(m_try, 0.0), m_try
            break
        except Exception:
            continue
    if used is None:
        raise ScenarioError("Calibration solve failed in both 1D and 0D.")

    f = patient.parallel_drop_fraction
    if f is None or f * gradient <= d_seg0:
        r_term, d_pair, d_seg = 0.0, d_seg0, d_seg0          # bare calibration: the pair drop is the vein's own drop
    else:
        d_pair = f * gradient
        r_term = brentq(lambda rt: rt * q_ref + seg_drop(used, rt) - d_pair, 0.0, d_pair / q_ref, xtol=1e-13)
        d_seg = d_pair - r_term * q_ref
    if gradient <= d_pair + 1e-9:
        raise ScenarioError(
            f"Reference gradient ({gradient:.2f} mmHg) does not exceed the IJV branch drop ({d_pair:.2f} mmHg) at the "
            f"reference flow, so no positive upstream resistance can reproduce q_ref = {q_ref} mL/s.")
    share = patient.ijv_share_ref
    q_total = q_ref / (share if share is not None else 1.0)
    q_collat = q_total - q_ref
    g_collat = (q_collat / max(d_pair, 1e-9)) if q_collat > 0 else 0.0
    return NetworkCalibration(r_up=(gradient - d_pair) / q_total, g_collat=g_collat, r_term=r_term, q_total_ref=q_total,
                              q_ijv_ref=q_ref, drop_pair_ref=d_pair, drop_segment_ref=d_seg, parallel_drop_fraction=f,
                              p_in_ref=patient.p_in_ref_mmHg, p_out_ref=p_out, method=used)


# ----------------------------------------------------------------------------------------------------------
# 7. Results and regime classification
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeThresholds:
    """Alert thresholds. These are MODELING ASSUMPTIONS, not clinical cut-offs."""
    v_stasis_cm_s: float = 2.0
    shear_warn_1_s: float = 15.0
    shear_crit_1_s: float = 5.0
    severe_collapse_fraction: float = 0.25      # min area below this fraction of A0
    diversion_share: float = 0.5                # IJV carries less than this share of total outflow (forward flow)
    speed_index_warn: float = 0.5


@dataclass(frozen=True, eq=False)
class SteadyStateResult:
    scenario: Optional[str]
    p_out_ijv: float
    p_out_collat: float
    g_eff: float
    p_ext: float
    method_used: str                  # '1d' or '0d'
    fallback_reason: Optional[str]
    # prescribed input and solved pressures
    q_total_factor: float
    q_total: float                    # prescribed total cranial outflow, mL/s
    p_in: float                       # SOLVED inlet (cranial venous) pressure, mmHg
    p_node_mmHg: float                # pressure at node N (IJV inlet)
    p_vein_outlet_mmHg: float         # pressure at the 1D segment's outlet, upstream of R_term
    reversal_threshold_mmHg: float    # outlet difference at which IJV flow is zero (Q_total * R_collat)
    # flow split (mL/s); forward = towards the heart. Q_IJV < 0 means reversed jugular flow.
    q_ijv: float
    q_collateral: float
    ijv_share: float                  # Q_IJV / Q_total (negative when reversed)
    # headline outputs
    velocity_mean_cm_s: float         # signed, Q_IJV / (length-averaged area)
    shear_mean_1_s: float
    shear_min_1_s: float
    viscosity_mean_mPa_s: float
    wss_mean_Pa: float                # wall shear STRESS (Pa): station-wise mu_eff(gamma) x mean wall shear rate, length-averaged
    wss_min_Pa: float                 # lowest station value (Pa); 0 when flow is zero
    wss_carreau_over_newtonian: float # wss_mean / (4 mPa.s x mean shear rate): how much shear-thinning blood changes the stress
    residence_time_s: float           # V / |Q_IJV|; infinity when Q_IJV is zero (valve closed: no exchange)
    volume_mL: float
    # geometry / pressure diagnostics
    area_mean_cm2: float
    area_min_cm2: float
    ptm_min_mmHg: float
    ptm_max_mmHg: float
    frac_collapsed: float
    speed_index_max: float
    closure_error: float              # |Q_IJV + Q_collat - Q_total| (mL/s), ~0
    shape_penalty_max: float          # largest slit-lumen resistance penalty along the vein (1 = circular)
    aspect_ratio_max: float           # largest lumen aspect ratio a/b along the vein (1 = circular)
    valve_closed: bool                # a competent valve blocked reversed flow (Q_IJV forced to 0)
    q_ijv_unconstrained: float        # jugular flow without the valve (NaN if not computed); equals q_ijv when open
    stasis: bool                      # valve-closed zero flow, or |mean velocity| below the stagnation cutoff
    stasis_basis: Optional[str]       # why stasis was declared
    phase_x_mmHg: float               # phase-map coordinate: tissue pressure P_ext
    phase_y_mmHg: float               # phase-map coordinate: effective jugular outlet pressure (after respiration)
    respiration_phase: Optional[str]
    respiration_shift_mmHg: float
    regime_flags: List[str]
    warnings: List[str]
    x_cm: np.ndarray = field(repr=False, default=None)
    p_mmHg: np.ndarray = field(repr=False, default=None)
    area_cm2: np.ndarray = field(repr=False, default=None)
    velocity_cm_s: np.ndarray = field(repr=False, default=None)
    shear_1_s: np.ndarray = field(repr=False, default=None)

    @property
    def p_out_difference_mmHg(self) -> float:
        return self.p_out_ijv - self.p_out_collat


def classify_regime(v_mean_cm_s: float, shear_min: float, area_min: float, A0: float, frac_collapsed: float,
                    speed_index: float, thr: RegimeThresholds = RegimeThresholds(),
                    ijv_share: float = float("nan")) -> List[str]:
    """Human-readable flags from the thresholds above. Direction is decided by the mean velocity."""
    flags = []
    if v_mean_cm_s >= thr.v_stasis_cm_s:
        flags.append("forward flow")
    elif v_mean_cm_s <= -thr.v_stasis_cm_s:
        flags.append("reversed flow")
    else:
        flags.append("stagnant (|v| below cutoff)")
        if v_mean_cm_s < 0:
            flags.append("net direction reversed")
    if math.isfinite(ijv_share) and 0.0 <= ijv_share < thr.diversion_share:
        flags.append(f"flow diverted to collateral (IJV {ijv_share:.0%})")
    if frac_collapsed > 0:
        flags.append("partial collapse (P_tm < 0 over part of the vein)")
    if area_min < thr.severe_collapse_fraction * A0:
        flags.append("severe collapse")
    if shear_min < thr.shear_crit_1_s and abs(v_mean_cm_s) > 0:
        flags.append("critical low shear")
    elif shear_min < thr.shear_warn_1_s and abs(v_mean_cm_s) > 0:
        flags.append("low shear")
    if speed_index > thr.speed_index_warn:
        flags.append("flow-limitation risk (inertia-free model unreliable)")
    return flags


# ----------------------------------------------------------------------------------------------------------
# 8. The steady-state solve (prescribed total outflow, two outlets; P_in is an output)
# ----------------------------------------------------------------------------------------------------------
def _ijv_branch(method, q, p_out_ijv, p_ext, g_eff, law, L_cm, rheology, n_pts, r_term):
    """IJV branch = 1D segment + terminal resistance. Returns (segment state, branch drop D_ijv signed like q)."""
    st = _vein(method, q, p_out_ijv + r_term * q, p_ext, g_eff, law, L_cm, rheology, n_pts)
    return st, st.drop + r_term * q


def _outlets(conditions: Mapping[str, object]):
    """(p_out_ijv, p_out_collat); a single legacy 'p_out' is applied to both outlets."""
    if "p_out_ijv" in conditions or "p_out_collat" in conditions:
        return float(conditions["p_out_ijv"]), float(conditions["p_out_collat"])
    return float(conditions["p_out"]), float(conditions["p_out"])


def _static_column(p_node: float, p_ext: float, g_eff: float, law: TubeLaw, L_cm: float, n_pts: int) -> _VeinState:
    """Isolated, flow-free vein (valve closed): pressure is hydrostatic along the vein, so no ODE is needed."""
    x = np.linspace(0.0, L_cm, n_pts)
    p = p_node + _hyd_mmHg_per_m(g_eff) * (x / 100.0)
    return _VeinState(x_cm=x, p=p, area=law.area(p - p_ext), drop=0.0, method="static")


# ---- Respiration (thoracic pump), head-down tilt, NASA flowchart: closed-form helpers ----------------------------
RESPIRATION_PHASES = ("Deep Inspiration", "Resting", "Valsalva")
# Working values (mmHg added to the jugular outlet pressure). Inspiration: right atrial pressure falls by 'several mmHg'
# (CV Physiology). Valsalva: a standard 40 mmHg strain raised CVP by ~40 mmHg (J Appl Physiol 2000); +20 is a moderate strain.
RESPIRATION_SHIFT_MMHG = {"Deep Inspiration": -4.0, "Resting": 0.0, "Valsalva": 20.0}


def effective_gravity_for_tilt(tilt_deg: float) -> float:
    """Axial gravity along inlet -> outlet for a recumbent body: 9.81 * sin(tilt). 0 = supine, -6 = head-down tilt (negative)."""
    return G_EARTH * math.sin(math.radians(float(tilt_deg)))


def respiration_shifts(phase: str, collateral_coupling: float = 0.0, shifts: Optional[Mapping[str, float]] = None):
    """(shift at the jugular outlet, shift at the collateral outlet) in mmHg for a respiration phase."""
    table = dict(RESPIRATION_SHIFT_MMHG)
    table.update(shifts or {})
    if phase not in table:
        raise ScenarioError(f"Unknown respiration phase '{phase}'. Available: {', '.join(table)}.")
    if not 0.0 <= collateral_coupling <= 1.0:
        raise ValueError("collateral_coupling must be between 0 and 1.")
    d = float(table[phase])
    return d, float(collateral_coupling) * d


def apply_respiration(conditions: Mapping[str, object], phase: str, collateral_coupling: float = 0.0,
                      shifts: Optional[Mapping[str, float]] = None) -> Dict[str, object]:
    """Copy of `conditions` with the respiration shift added to the outlet pressures (jugular fully, collateral by coupling)."""
    d_ijv, d_col = respiration_shifts(phase, collateral_coupling, shifts)
    p_oi, p_oc = _outlets(conditions)
    out = dict(conditions)
    out["p_out_ijv"], out["p_out_collat"] = p_oi + d_ijv, p_oc + d_col
    out["respiration_phase"], out["respiration_shift_mmHg"] = phase, d_ijv
    return out


NASA_PROPHYLAXIS_TEXT = ("NASA PROTOCOL: Prophylaxis indicated. Administer Apixaban 2.5 mg twice a day. "
                         "Discontinue 24h prior to Earth landing.")
NASA_MONITORING_TEXT = "NASA PROTOCOL: Stasis detected. Increased monitoring and hydration indicated."


def nasa_protocol(stasis: bool, family_history: bool = False, thrombophilia: bool = False,
                  high_risk_hormones: bool = False):
    """
    Branch of the NASA OCHMO-MTB-007 in-flight flowchart for a crew member with no thrombosis found: stasis plus any risk
    factor -> prophylaxis; stasis alone -> increased monitoring and hydration; no stasis -> nothing. Returns None or
    (level, text) with level 'error' (prophylaxis) or 'warning' (monitoring). A reference mapping, not medical advice.
    """
    if not stasis:
        return None
    if family_history or thrombophilia or high_risk_hormones:
        return "error", NASA_PROPHYLAXIS_TEXT
    return "warning", NASA_MONITORING_TEXT


def solve_steady_state(conditions: Mapping[str, object], patient: PatientProfile,
                       network: Optional[NetworkCalibration] = None, *, p_ext: Optional[float] = None,
                       q_total: Optional[float] = None, method: str = "auto", n_pts: int = 121,
                       thresholds: RegimeThresholds = RegimeThresholds(),
                       report_unconstrained: bool = True) -> SteadyStateResult:
    """
    Solve the flow split (IJV flow may reverse) and the inlet pressure for a prescribed total outflow.

    conditions  dict with q_total_factor, p_out_ijv, p_out_collat (mmHg) and g_eff (m/s^2); get_scenario_conditions()
                output works. A single 'p_out' is accepted and applied to both outlets.
    network     NetworkCalibration from calibrate_network(patient); computed here if omitted (slower).
    p_ext       tissue pressure for THIS run (defaults to the patient's value for the scenario). Not re-calibrated.
    q_total     absolute total outflow (mL/s); overrides q_total_factor x network.q_total_ref when given.
    method      'auto' (1D, falling back to 0D on failure or a non-physical result), '1d', or '0d'.
    report_unconstrained  when a competent valve closes, also solve the valve-free flow to report what it blocked
                (costs one normal solve; set False for map sweeps, where the closed state is then fully algebraic).
    """
    if method not in ("auto", "1d", "0d"):
        raise ValueError("method must be 'auto', '1d' or '0d'.")
    if network is None:
        network = calibrate_network(patient)
    p_oi, p_oc = _outlets(conditions)
    g_eff = float(conditions["g_eff"])
    factor = float(conditions.get("q_total_factor", 1.0))
    q_tot = float(q_total) if q_total is not None else factor * network.q_total_ref
    if q_total is not None and network.q_total_ref > 0:
        factor = q_tot / network.q_total_ref
    scen = conditions.get("key")
    law = build_tube_law(patient.A0_cm2, patient.K_mmHg, patient.n, patient.m, lumen=patient.lumen_model)
    p_ext_val = float(p_ext) if p_ext is not None else patient.p_ext_for(scen)
    L, g_c, r_t = patient.L_cm, network.g_collat, network.r_term
    dp_out = p_oi - p_oc

    def solve_with(m_try: str):
        branch = lambda q_, npts=n_pts: _ijv_branch(m_try, q_, p_oi, p_ext_val, g_eff, law, L, patient.rheology, npts, r_t)
        if g_c == 0.0:
            q = q_tot                                          # no collateral: the IJV carries everything
        else:
            s = q_tot - g_c * dp_out                           # root lies between 0 and s (see module docstring)
            if abs(s) < 1e-14:
                q = 0.0
            else:
                # the drop depends only on the segment's end pressures, so the root-finder uses a 2-point output grid
                residual = lambda q_: q_ + g_c * (dp_out + branch(q_, 2)[1]) - q_tot
                q = brentq(residual, min(0.0, s), max(0.0, s), xtol=ROOT_TOL, rtol=ROOT_TOL)
        st, d_branch = branch(q)
        return q, st, d_branch

    # Competent valve: the unconstrained flow is reversed exactly when s < 0 (the closure root lies on the side of zero
    # where s lies), so the valve decision is an algebraic test made before any solving.
    hyd_L = _hyd_mmHg_per_m(g_eff) * (L / 100.0)
    valve_closed = bool(patient.valve_competent and g_c > 0.0 and (q_tot - g_c * dp_out) < 0.0)
    q, state, d_branch, used, reason = None, None, None, None, None
    attempts = (("1d", "0d") if method == "auto" else (method,)) if (not valve_closed or report_unconstrained) else ()
    for m_try in attempts:
        try:
            q_try, st_try, d_try = solve_with(m_try)
            if not (math.isfinite(q_try) and math.isfinite(d_try) and np.all(np.isfinite(st_try.area))
                    and np.all(st_try.area > 0) and np.all(np.isfinite(st_try.p))):
                raise RuntimeError("non-finite or non-positive result")
            q, state, d_branch, used = q_try, st_try, d_try, m_try
            break
        except Exception as exc:
            if m_try == "1d" and method == "auto":
                reason = f"1D solve failed ({type(exc).__name__}: {exc}); lumped 0D fallback used."
            elif method != "auto":
                raise
    q_unc = float(q) if q is not None else float("nan")
    if valve_closed:
        # Closed valve: the jugular vein is an isolated column. All outflow goes through the collateral, which fixes the
        # node pressure in closed form: Q_total = g_c (P_N + rho g L - P_out,collat).
        q, d_branch = 0.0, 0.0
        state = _static_column(p_oc + q_tot / g_c - hyd_L, p_ext_val, g_eff, law, L, n_pts)
        used, reason = "static", None
    if state is None:
        raise RuntimeError("Both the 1D solver and the 0D fallback failed.")

    # ---- outputs ----
    area, x = state.area, state.x_cm
    vel = q / area
    phi_v, sfac_v = law.shape_factors(area)
    shear = 4.0 * np.abs(vel) / np.sqrt(area / math.pi) * sfac_v          # slit-corrected mean wall shear rate
    phi_max = float(np.max(phi_v))
    mu_pa = effective_viscosity(shear, patient.rheology)                      # Pa*s, station-wise
    mu = mu_pa * 1e3
    tau = mu_pa * shear                                                       # wall shear stress (Pa) per station
    span = x[-1] - x[0]
    trap = lambda y: float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)) / span)
    volume = float(np.sum(0.5 * (area[1:] + area[:-1]) * np.diff(x)))
    area_mean = volume / span
    ptm = state.p - p_ext_val
    q_c = q_tot if valve_closed else (g_c * (dp_out + d_branch) if g_c > 0 else 0.0)
    p_node = float(state.p[0])
    p_in = p_node + network.r_up * q_tot
    sidx = float(np.max(law.speed_index(area / patient.A0_cm2, vel)))
    closure = abs(q + q_c - q_tot)
    v_mean = q / area_mean
    share = (q / q_tot) if abs(q_tot) > Q_FLOOR else float("nan")
    thr_dp = reversal_threshold_mmHg(network, q_tot)
    notes = []
    if reason:
        notes.append(reason)
    if sidx > thresholds.speed_index_warn:
        notes.append(f"Speed index {sidx:.2f}: inertia/flow limitation may matter; steady Poiseuille result is unreliable.")
    if state.method == "0d":
        notes.append("0D result: uniform vein, cannot represent partial collapse.")
    if q < 0:
        notes.append("Reversed jugular flow assumes the venous valve is absent or incompetent. A valve is present in about 90 % "
                     "of people but competence varies by study; switch the valve on to see it block the backflow.")
    # ODE pressure noise (~1e-6 mmHg) is amplified by the collateral conductance; warn only above 0.01 % of the largest flow.
    if closure > 1e-4 * max(1.0, abs(q_tot), abs(q), abs(q_c)):
        notes.append(f"Flow closure error {closure:.1e} mL/s.")
    flags = classify_regime(v_mean, float(shear.min()), float(area.min()), patient.A0_cm2, float(np.mean(ptm < 0)),
                            sidx, thresholds, share)
    if valve_closed:
        flags.append("venous valve closed: blood trapped (Q_IJV = 0)")
    if valve_closed:
        stasis, basis = True, "valve closed: Q_IJV forced to 0 (blood trapped)"
    elif abs(v_mean) < thresholds.v_stasis_cm_s:
        stasis, basis = True, f"|mean velocity| {abs(v_mean):.2f} cm/s is below the {thresholds.v_stasis_cm_s:g} cm/s cutoff"
    else:
        stasis, basis = False, None
    return SteadyStateResult(
        scenario=scen, p_out_ijv=p_oi, p_out_collat=p_oc, g_eff=g_eff, p_ext=p_ext_val, method_used=used,
        fallback_reason=reason, q_total_factor=float(factor), q_total=float(q_tot), p_in=float(p_in), p_node_mmHg=p_node,
        p_vein_outlet_mmHg=float(state.p[-1] if valve_closed else p_oi + r_t * q), reversal_threshold_mmHg=float(thr_dp), q_ijv=float(q),
        q_collateral=float(q_c), ijv_share=float(share), velocity_mean_cm_s=float(v_mean), shear_mean_1_s=trap(shear),
        shear_min_1_s=float(shear.min()), viscosity_mean_mPa_s=trap(mu),
        wss_mean_Pa=trap(tau), wss_min_Pa=float(tau.min()),
        wss_carreau_over_newtonian=(trap(tau) / (NEWTONIAN_MU * trap(shear)) if trap(shear) > 1e-9 else float("nan")),
        residence_time_s=(math.inf if abs(q) <= Q_FLOOR else float(volume / abs(q))), volume_mL=volume,
        area_mean_cm2=float(area_mean),
        area_min_cm2=float(area.min()), ptm_min_mmHg=float(ptm.min()), ptm_max_mmHg=float(ptm.max()),
        frac_collapsed=float(np.mean(ptm < 0)), speed_index_max=sidx, closure_error=float(closure),
        shape_penalty_max=phi_max, aspect_ratio_max=float(phi_max + math.sqrt(max(phi_max ** 2 - 1.0, 0.0))),
        valve_closed=valve_closed, q_ijv_unconstrained=q_unc, stasis=stasis, stasis_basis=basis,
        phase_x_mmHg=float(p_ext_val), phase_y_mmHg=float(p_oi),
        respiration_phase=conditions.get("respiration_phase"),
        respiration_shift_mmHg=float(conditions.get("respiration_shift_mmHg", 0.0)),
        regime_flags=flags, warnings=notes, x_cm=x, p_mmHg=state.p, area_cm2=area, velocity_cm_s=vel, shear_1_s=shear)


def run_scenario(scenario_name: str, patient: PatientProfile, network: Optional[NetworkCalibration] = None, *,
                 overrides: Optional[Mapping[str, float]] = None, p_ext: Optional[float] = None,
                 q_total: Optional[float] = None, method: str = "auto", respiration: Optional[str] = None,
                 collateral_coupling: float = 0.0, respiration_table: Optional[Mapping[str, float]] = None,
                 tilt_deg: Optional[float] = None, **kwargs) -> SteadyStateResult:
    """
    Look up a scenario, apply optional overrides (q_total_factor, p_out_ijv, p_out_collat, g_eff), then the optional
    respiration phase (shifts the outlets) and head-down tilt (replaces g_eff by 9.81 sin(tilt)), then solve.
    `respiration_table` overrides the default phase shifts; extra keyword arguments go to solve_steady_state.
    """
    cond = get_scenario_conditions(scenario_name)
    if overrides:
        for k, v in overrides.items():
            if k not in ("q_total_factor", "p_out_ijv", "p_out_collat", "g_eff"):
                raise ValueError(f"Cannot override '{k}'; allowed: q_total_factor, p_out_ijv, p_out_collat, g_eff.")
            cond[k] = float(v)
    if respiration is not None:
        cond = apply_respiration(cond, respiration, collateral_coupling, respiration_table)
    if tilt_deg is not None:
        cond["g_eff"] = effective_gravity_for_tilt(tilt_deg)
    return solve_steady_state(cond, patient, network, p_ext=p_ext, q_total=q_total, method=method, **kwargs)


# ----------------------------------------------------------------------------------------------------------
# Demo: python ijv_math_engine.py
# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pat = PatientProfile.population_placeholder()
    net = calibrate_network(pat)
    print(f"Calibration ({net.method}, parallel_drop_fraction {net.parallel_drop_fraction}): R_up {net.r_up:.3f}, "
          f"R_term {net.r_term:.3f}, R_collat {net.r_collat:.3f} mmHg*s/mL; reference total outflow {net.q_total_ref:.2f} "
          f"mL/s; reversal threshold {reversal_threshold_mmHg(net, net.q_total_ref):.2f} mmHg\n")
    print(f"{'scenario':30s}{'Q_tot':>7s}{'Pout_I':>8s}{'Pout_C':>8s}{'Q_IJV':>8s}{'Q_col':>8s}{'v_mean':>8s}{'A_mean':>8s}{'P_in':>8s}  flags")
    for key in list_scenarios():
        r = run_scenario(key, pat, net)
        print(f"{key:30s}{r.q_total:>7.2f}{r.p_out_ijv:>8.1f}{r.p_out_collat:>8.1f}{r.q_ijv:>8.2f}{r.q_collateral:>8.2f}"
              f"{r.velocity_mean_cm_s:>8.2f}{r.area_mean_cm2:>8.2f}{r.p_in:>8.2f}  {'; '.join(r.regime_flags)}")
