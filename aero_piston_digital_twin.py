
"""
AERO-PROPULSION DIGITAL TWIN — SIH 2026 SINGLE-FILE PROTOTYPE
==============================================================

Hybrid, mission-aware Digital Twin for a generic aero-piston engine.

IMPORTANT SCIENTIFIC STATUS
----------------------------
* The engine model is a simplified engineering model, NOT a certified engine model.
* NASA C-MAPSS is a TURBOFAN reference dataset. It is NOT an aero-piston dataset.
* NASA/C-MAPSS concepts are used for degradation/prognostics methodology only.
* Aero-piston telemetry generated here is SYNTHETIC DATA.
* Health Index, RUL, fault confidence and mission reliability are MODEL ESTIMATES.
* Nothing in this prototype is flight-certified or an operational maintenance limit.

ONE-FILE DESIGN
---------------
This file intentionally contains:
  1. Physics-based aero-piston model
  2. Synthetic correlated telemetry generator
  3. Sensor noise/bias/drift/outlier model
  4. Custom Unscented Kalman Filter (UKF)
  5. Physics/UKF residual anomaly detection
  6. Random Forest fault classifier
  7. Prototype Health Index
  8. Degradation model
  9. Random Forest RUL regression
 10. Mission simulator
 11. Monte-Carlo mission reliability
 12. What-if analysis
 13. Grid-search mission optimization
 14. Explainability / feature importance
 15. CSV upload and column mapping
 16. Real-time streaming simulation
 17. 3D engine visualization
 18. Cryptographic telemetry integrity: SHA-256 hash chain + HMAC
 19. CSV/JSON/report export
 20. Validation metrics and self-tests

Recommended install:
    pip install streamlit numpy pandas scipy scikit-learn plotly

Run:
    streamlit run aero_piston_digital_twin.py

Optional:
    Set DT_HMAC_SECRET to a private secret before running.
"""

from __future__ import annotations

import os
import io
import json
import time
import hmac
import hashlib
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# 0. CONFIGURATION
# ============================================================

@dataclass
class EngineConfig:
    # Generic demonstration engine: order-of-magnitude only.
    displacement_L: float = 6.0
    cylinders: int = 4
    compression_ratio: float = 8.5
    max_rpm: float = 5200.0
    idle_rpm: float = 900.0
    reference_rpm: float = 3600.0
    reference_power_kW: float = 74.6       # ~100 hp
    reference_torque_Nm: float = 198.0
    stoich_afr: float = 14.7
    lhv_MJ_kg: float = 43.0
    air_R: float = 287.05
    gamma: float = 1.4
    mechanical_efficiency: float = 0.88
    max_cylinder_temp_C: float = 230.0
    max_oil_temp_C: float = 130.0
    min_oil_pressure_bar: float = 1.2
    max_vibration_g: float = 2.0

    # Degradation coefficients — synthetic/empirical.
    bearing_deg_coeff: float = 0.000012
    cooling_deg_coeff: float = 0.000010
    lubrication_deg_coeff: float = 0.000009
    fuel_deg_coeff: float = 0.000008
    generic_deg_coeff: float = 0.000004

    # Sensor noise standard deviations.
    rpm_noise: float = 10.0
    map_noise: float = 0.9
    cht_noise: float = 1.2
    oil_temp_noise: float = 0.8
    oil_pressure_noise: float = 0.035
    torque_noise: float = 1.8
    fuel_noise: float = 0.0008
    vibration_noise: float = 0.018

    seed: int = 42


CFG = EngineConfig()


# ============================================================
# 1. DATA LABELS / SCIENTIFIC HONESTY
# ============================================================

NASA_REFERENCE_NOTE = (
    "NASA C-MAPSS is a simulated commercial turbofan reference dataset. "
    "It contains multivariate time series, operational settings, sensor noise "
    "and run-to-failure degradation. It is used here only as a methodological "
    "reference for degradation/prognostics. The aero-piston telemetry is synthetic."
)

STATUS_LEVELS = ["NORMAL", "WARNING", "ABNORMAL", "CRITICAL"]

FAULT_NAMES = [
    "Normal",
    "Bearing degradation",
    "Cooling degradation",
    "Fuel-system degradation",
    "Lubrication degradation",
    "Air-intake restriction",
    "Ignition degradation",
    "Sensor fault",
]


# ============================================================
# 2. PHYSICS MODEL
# ============================================================

class AeroPistonPhysics:
    """
    Simplified nonlinear aero-piston model.

    Physically derived:
      rho = P/(R*T)
      omega = 2*pi*N/60
      P_mech = T*omega

    Empirical/assumed:
      volumetric efficiency, torque/load scaling, thermal dynamics,
      friction and vibration relationships.

    These equations are deliberately transparent rather than proprietary.
    """

    def __init__(self, cfg: EngineConfig = CFG):
        self.c = cfg

    @staticmethod
    def isa_atmosphere(altitude_m: float, delta_T_C: float = 0.0):
        h = max(0.0, float(altitude_m))
        T0 = 288.15
        P0 = 101325.0
        lapse = 0.0065
        R = 287.05
        g = 9.80665
        if h <= 11000:
            T = T0 - lapse * h + delta_T_C
            P = P0 * (max(T0 - lapse*h, 1.0)/T0) ** (g/(R*lapse))
        else:
            T11 = T0 - lapse*11000
            P11 = P0 * (T11/T0) ** (g/(R*lapse))
            T = T11 + delta_T_C
            P = P11 * np.exp(-g*(h-11000)/(R*T11))
        T = max(180.0, T)
        P = max(15000.0, P)
        rho = P/(R*T)
        return T, P, rho

    def volumetric_efficiency(self, rpm: float, map_kpa: float, health: float):
        n = max(500.0, rpm)
        speed_factor = np.exp(-((n - 3400.0)/2300.0)**2) * 0.13
        pressure_factor = np.clip((map_kpa - 35.0)/70.0, 0, 1) * 0.22
        base = 0.58 + speed_factor + pressure_factor
        degradation_penalty = 0.10*(1.0-health/100.0)
        return float(np.clip(base - degradation_penalty, 0.45, 0.93))

    def step(self, controls: Dict[str, float], hidden: Dict[str, float], dt: float = 1.0):
        throttle = float(np.clip(controls.get("throttle", 0.6), 0.0, 1.0))
        altitude = max(0.0, float(controls.get("altitude_m", 1500.0)))
        ambient_delta = float(controls.get("ambient_delta_C", 0.0))
        load = float(np.clip(controls.get("load", throttle), 0.0, 1.0))

        health = float(np.clip(hidden.get("health", 100.0), 1.0, 100.0))
        bearing = float(np.clip(hidden.get("bearing", 0.0), 0, 1))
        cooling = float(np.clip(hidden.get("cooling", 0.0), 0, 1))
        lubrication = float(np.clip(hidden.get("lubrication", 0.0), 0, 1))
        fuel_fault = float(np.clip(hidden.get("fuel_system", 0.0), 0, 1))
        air_fault = float(np.clip(hidden.get("air_intake", 0.0), 0, 1))
        ignition = float(np.clip(hidden.get("ignition", 0.0), 0, 1))

        T_amb, P_amb, rho = self.isa_atmosphere(altitude, ambient_delta)
        amb_C = T_amb - 273.15

        # Manifold pressure: throttle and altitude, with degradation losses.
        max_map = P_amb / 1000.0 * (0.98 + 0.03*throttle)
        map_kpa = max(25.0, max_map*(0.28 + 0.72*throttle))
        map_kpa *= (1.0 - 0.22*air_fault)
        map_kpa *= (0.96 + 0.04*health/100.0)

        # RPM target and first-order rotational dynamics.
        target_rpm = self.c.idle_rpm + throttle*0.72*(self.c.max_rpm-self.c.idle_rpm)
        target_rpm *= (0.88 + 0.12*health/100.0)
        target_rpm *= (1.0 - 0.08*load*air_fault)
        target_rpm *= (1.0 - 0.06*ignition)
        rpm_prev = hidden.get("rpm", target_rpm)
        rpm = rpm_prev + (target_rpm-rpm_prev)*(1.0-np.exp(-dt/3.5))

        omega = 2*np.pi*rpm/60.0
        eta_v = self.volumetric_efficiency(rpm, map_kpa, health)

        # Air mass flow: displacement * RPM / 2 for 4-stroke.
        swept_m3 = self.c.displacement_L/1000.0
        cycles_per_s = rpm/120.0
        air_mass_flow = rho*swept_m3*cycles_per_s*eta_v

        # Mixture/fuel relationship.
        effective_afr = self.c.stoich_afr*(1.0 + 0.06*fuel_fault + 0.04*ignition)
        fuel_flow = max(0.00005, air_mass_flow/effective_afr*(0.75 + 0.25*throttle))
        fuel_flow *= (1.0 + 0.10*fuel_fault)

        # Indicated/shaft power approximation.
        p_ref = self.c.reference_power_kW*1000.0
        speed_ratio = np.clip(rpm/self.c.reference_rpm, 0.15, 1.35)
        pressure_ratio = np.clip(map_kpa/95.0, 0.15, 1.35)
        health_factor = health/100.0
        fault_factor = (1-0.35*bearing)*(1-0.18*lubrication)*(1-0.22*fuel_fault)
        load_factor = 0.28 + 0.72*load
        power = p_ref * speed_ratio**0.65 * pressure_ratio**0.75
        power *= eta_v/0.78 * health_factor * fault_factor * load_factor
        power = float(np.clip(power, 1000.0, p_ref*1.08))

        torque = power/max(omega, 1e-6)

        # Friction / thermal relationships.
        friction = 0.05 + 0.000000012*rpm**2
        friction *= (1 + 1.8*lubrication + 1.4*bearing)

        heat_load = (
            55.0
            + 95.0*load
            + 22.0*throttle
            + 0.018*max(0.0, amb_C-15.0)
            + 28.0*cooling
            + 14.0*ignition
            + 10.0*fuel_fault
            + 18.0*bearing
        )
        cht = amb_C + heat_load/(1.0 + 0.9*throttle) + 0.0022*rpm
        cht += 15.0*(1.0-health_factor)

        oil_temp = amb_C + 38.0 + 0.012*rpm + 24.0*load + 22.0*lubrication + 15.0*bearing
        oil_temp += 8.0*(1.0-health_factor)

        oil_pressure = 4.2 + 0.00018*rpm - 0.95*lubrication - 0.65*bearing
        oil_pressure = max(0.2, oil_pressure)

        vibration = (
            0.12 + 0.000000035*(rpm-1000.0)**2
            + 0.72*bearing
            + 0.24*lubrication
            + 0.18*ignition
            + 0.12*air_fault
        )
        vibration *= (0.75 + 0.35*load)
        vibration = float(max(0.02, vibration))

        return {
            "rpm": float(rpm),
            "map_kpa": float(map_kpa),
            "cht_C": float(cht),
            "oil_temp_C": float(oil_temp),
            "oil_pressure_bar": float(oil_pressure),
            "torque_Nm": float(torque),
            "power_kW": float(power/1000.0),
            "fuel_flow_kg_s": float(fuel_flow),
            "air_mass_flow_kg_s": float(air_mass_flow),
            "vibration_g": float(vibration),
            "ambient_temp_C": float(amb_C),
            "ambient_pressure_kPa": float(P_amb/1000.0),
            "air_density_kg_m3": float(rho),
            "eta_v": float(eta_v),
            "health": float(health),
            "bearing_deg": float(bearing),
            "cooling_deg": float(cooling),
            "lubrication_deg": float(lubrication),
            "fuel_deg": float(fuel_fault),
            "air_deg": float(air_fault),
            "ignition_deg": float(ignition),
            "friction_index": float(friction),
        }


# ============================================================
# 3. DEGRADATION MODEL
# ============================================================

def advance_degradation(hidden: Dict[str, float], controls: Dict[str, float],
                        dt_hours: float, rng: np.random.Generator):
    """
    D(t+1)=D(t)+rate*operating_stress*dt+process_noise.

    This is a synthetic degradation model inspired by the general
    run-to-failure methodology used in prognostics datasets.
    """
    h = float(hidden["health"])
    throttle = np.clip(controls.get("throttle", 0.6), 0, 1)
    load = np.clip(controls.get("load", 0.6), 0, 1)
    altitude = max(0.0, controls.get("altitude_m", 1500.0))
    ambient = controls.get("ambient_delta_C", 0.0)

    stress = (
        0.35
        + 0.55*throttle**1.6
        + 0.45*load**1.5
        + 0.20*max(0, ambient)/25.0
        + 0.10*min(altitude/12000.0, 1.0)
    )

    base = 0.00035*stress*dt_hours
    noise = rng.normal(0, 0.000025)

    hidden["bearing"] = np.clip(hidden["bearing"] + base*1.35 + noise, 0, 1.0)
    hidden["cooling"] = np.clip(hidden["cooling"] + base*1.05 + noise, 0, 1.0)
    hidden["lubrication"] = np.clip(hidden["lubrication"] + base*0.95 + noise, 0, 1.0)
    hidden["fuel_system"] = np.clip(hidden["fuel_system"] + base*0.70 + noise*0.7, 0, 1.0)

    total = (
        0.32*hidden["bearing"] +
        0.24*hidden["cooling"] +
        0.22*hidden["lubrication"] +
        0.14*hidden["fuel_system"] +
        0.08*hidden.get("air_intake", 0.0)
    )

    hidden["health"] = float(np.clip(100.0*(1.0-total), 1.0, 100.0))
    return hidden


# ============================================================
# 4. SENSOR MODEL
# ============================================================

SENSOR_STD = {
    "rpm": CFG.rpm_noise,
    "map_kpa": CFG.map_noise,
    "cht_C": CFG.cht_noise,
    "oil_temp_C": CFG.oil_temp_noise,
    "oil_pressure_bar": CFG.oil_pressure_noise,
    "torque_Nm": CFG.torque_noise,
    "fuel_flow_kg_s": CFG.fuel_noise,
    "vibration_g": CFG.vibration_noise,
}

def sensor_measure(true_state: Dict[str, float], rng: np.random.Generator,
                   bias_state: Optional[Dict[str, float]] = None,
                   fault: Optional[Dict[str, Any]] = None):
    bias_state = bias_state or {}
    fault = fault or {}
    measured = {}
    for k, sigma in SENSOR_STD.items():
        value = float(true_state[k])
        bias = float(bias_state.get(k, 0.0))
        value += rng.normal(0, sigma) + bias

        # Optional fault injection.
        if fault.get("sensor_fault") and fault.get("sensor_channel") == k:
            mode = fault.get("sensor_mode", "offset")
            sev = float(fault.get("severity", 1.0))
            if mode == "offset":
                value += sev * 4.0 * max(sigma, 0.1)
            elif mode == "stuck":
                value = float(fault.get("stuck_value", value))
            elif mode == "noise":
                value += rng.normal(0, 5*sigma*sev)

        # Small random outlier chance.
        if rng.random() < 0.003:
            value += rng.normal(0, 5*sigma)

        measured[k] = value
    return measured


# ============================================================
# 5. CUSTOM UNSCENTED KALMAN FILTER
# ============================================================

class UKF:
    """
    Minimal self-contained scaled UKF.

    State:
      [rpm, map_kpa, cht_C, oil_temp_C, oil_pressure_bar, torque_Nm, health]

    Measurement:
      [rpm, map_kpa, cht_C, oil_temp_C, oil_pressure_bar, torque_Nm]

    The process function is the nonlinear physics model.
    """

    def __init__(self, physics: AeroPistonPhysics, x0: np.ndarray):
        self.physics = physics
        self.x = np.array(x0, dtype=float)
        self.n = len(self.x)
        self.P = np.diag([25.0, 2.0, 4.0, 2.0, 0.10, 8.0, 9.0])
        self.Q = np.diag([5.0, 0.30, 0.40, 0.25, 0.02, 0.8, 0.35])
        self.R = np.diag([
            CFG.rpm_noise**2,
            CFG.map_noise**2,
            CFG.cht_noise**2,
            CFG.oil_temp_noise**2,
            CFG.oil_pressure_noise**2,
            CFG.torque_noise**2,
        ])
        self.alpha = 0.35
        self.beta = 2.0
        self.kappa = 0.0
        self.lam = self.alpha**2*(self.n+self.kappa)-self.n
        self.Wm = np.full(2*self.n+1, 1/(2*(self.n+self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam/(self.n+self.lam)
        self.Wc[0] = self.Wm[0] + (1-self.alpha**2+self.beta)

    def sigma_points(self):
        S = np.linalg.cholesky((self.n+self.lam)*(self.P + 1e-9*np.eye(self.n)))
        pts = [self.x.copy()]
        for i in range(self.n):
            pts.append(self.x + S[:, i])
        for i in range(self.n):
            pts.append(self.x - S[:, i])
        return np.array(pts)

    def process(self, x: np.ndarray, controls: Dict[str, float], dt: float):
        hidden = {
            "rpm": x[0],
            "health": np.clip(x[6], 1, 100),
            "bearing": max(0.0, (100-x[6])/100.0*0.60),
            "cooling": max(0.0, (100-x[6])/100.0*0.35),
            "lubrication": max(0.0, (100-x[6])/100.0*0.30),
            "fuel_system": max(0.0, (100-x[6])/100.0*0.20),
        }
        s = self.physics.step(controls, hidden, dt)
        return np.array([
            s["rpm"], s["map_kpa"], s["cht_C"], s["oil_temp_C"],
            s["oil_pressure_bar"], s["torque_Nm"], x[6]
        ])

    def h(self, x: np.ndarray):
        return x[:6].copy()

    def predict(self, controls: Dict[str, float], dt: float = 1.0):
        sig = self.sigma_points()
        sig_f = np.array([self.process(p, controls, dt) for p in sig])
        x_pred = np.sum(self.Wm[:, None]*sig_f, axis=0)
        P_pred = self.Q.copy()
        for i in range(len(sig_f)):
            d = sig_f[i]-x_pred
            P_pred += self.Wc[i]*np.outer(d, d)
        self.x = x_pred
        self.P = P_pred
        return self.x.copy()

    def update(self, measurement: np.ndarray):
        sig = self.sigma_points()
        Z = np.array([self.h(p) for p in sig])
        z_pred = np.sum(self.Wm[:, None]*Z, axis=0)

        S = self.R.copy()
        Pxz = np.zeros((self.n, 6))
        for i in range(len(Z)):
            dz = Z[i]-z_pred
            dx = sig[i]-self.x
            S += self.Wc[i]*np.outer(dz, dz)
            Pxz += self.Wc[i]*np.outer(dx, dz)

        innovation = measurement-z_pred
        K = Pxz @ np.linalg.pinv(S)
        self.x = self.x + K@innovation
        self.P = self.P - K@S@K.T
        self.P = 0.5*(self.P+self.P.T)
        self.x[6] = np.clip(self.x[6], 1, 100)
        return self.x.copy(), innovation, S


# ============================================================
# 6. SYNTHETIC DATASET
# ============================================================

SCENARIOS = {
    "Healthy engine": {},
    "Bearing degradation": {"bearing": 0.45},
    "Cooling degradation": {"cooling": 0.45},
    "Fuel-system degradation": {"fuel_system": 0.45},
    "Lubrication degradation": {"lubrication": 0.45},
    "Air-intake restriction": {"air_intake": 0.40},
    "Ignition degradation": {"ignition": 0.40},
    "Sensor fault": {"sensor_fault": True},
    "High-altitude mission": {"altitude_m": 5500},
    "High-temperature mission": {"ambient_delta_C": 18},
    "Long-endurance mission": {"duration_h": 8.0},
    "Combined degradation + high-load": {
        "bearing": 0.30, "cooling": 0.25, "lubrication": 0.20,
        "throttle": 0.85, "load": 0.85
    },
}


def apply_fault_seed(hidden: Dict[str, float], scenario: str):
    spec = SCENARIOS.get(scenario, {})
    for key in ["bearing", "cooling", "lubrication", "fuel_system", "air_intake", "ignition"]:
        if key in spec:
            hidden[key] = float(spec[key])
    return hidden


def classify_fault_from_state(row: Dict[str, float]) -> str:
    """
    Synthetic labels generated from interpretable signatures.
    These are labels for the prototype's synthetic training environment.
    """
    b = row.get("bearing_deg", 0)
    c = row.get("cooling_deg", 0)
    l = row.get("lubrication_deg", 0)
    f = row.get("fuel_deg", 0)
    a = row.get("air_deg", 0)
    ig = row.get("ignition_deg", 0)

    vals = {
        "Bearing degradation": b,
        "Cooling degradation": c,
        "Lubrication degradation": l,
        "Fuel-system degradation": f,
        "Air-intake restriction": a,
        "Ignition degradation": ig,
    }
    k, v = max(vals.items(), key=lambda kv: kv[1])
    if v < 0.12:
        return "Normal"
    return k


@st.cache_data(show_spinner=False)
def generate_synthetic_dataset(n_engines: int = 12, cycles: int = 500,
                               seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    physics = AeroPistonPhysics(CFG)
    rows = []

    for unit in range(1, n_engines+1):
        hidden = {
            "rpm": 1100.0,
            "health": rng.uniform(92, 100),
            "bearing": rng.uniform(0, 0.03),
            "cooling": rng.uniform(0, 0.03),
            "lubrication": rng.uniform(0, 0.03),
            "fuel_system": rng.uniform(0, 0.02),
            "air_intake": 0.0,
            "ignition": 0.0,
        }

        failure_cycle = rng.integers(int(cycles*0.70), cycles)
        mission_offset = rng.uniform(-2, 2)

        for t in range(cycles):
            phase = t/cycles
            throttle = np.clip(
                0.48 + 0.20*np.sin(t/45.0) + 0.10*np.sin(t/13.0)
                + rng.normal(0, 0.035), 0.20, 0.92
            )
            load = np.clip(throttle + rng.normal(0, 0.04), 0.15, 0.95)
            altitude = np.clip(
                1500 + 3500*np.sin(t/130.0) + rng.normal(0, 80), 0, 8000
            )
            ambient = mission_offset + 6*np.sin(t/170.0)

            controls = {
                "throttle": throttle,
                "load": load,
                "altitude_m": altitude,
                "ambient_delta_C": ambient,
            }

            # Progressive stress and accelerated end-of-life degradation.
            dt_h = 1/60.0
            hidden = advance_degradation(hidden, controls, dt_h, rng)

            # Extra acceleration close to synthetic failure.
            if t > failure_cycle:
                extra = min(1.0, (t-failure_cycle)/max(1, cycles-failure_cycle))
                hidden["bearing"] = np.clip(hidden["bearing"] + 0.0008*extra, 0, 1)
                hidden["health"] = np.clip(hidden["health"] - 0.06*extra, 1, 100)

            true = physics.step(controls, hidden, dt=1.0)
            meas = sensor_measure(true, rng)

            row = {
                "unit": unit,
                "cycle": t,
                "timestamp_s": float(t),
                "throttle": throttle,
                "load": load,
                "altitude_m": altitude,
                "ambient_delta_C": ambient,
                "rpm": true["rpm"],
                "rpm_measured": meas["rpm"],
                "map_kpa": true["map_kpa"],
                "map_measured": meas["map_kpa"],
                "cht_C": true["cht_C"],
                "cht_measured": meas["cht_C"],
                "oil_temp_C": true["oil_temp_C"],
                "oil_temp_measured": meas["oil_temp_C"],
                "oil_pressure_bar": true["oil_pressure_bar"],
                "oil_pressure_measured": meas["oil_pressure_bar"],
                "torque_Nm": true["torque_Nm"],
                "torque_measured": meas["torque_Nm"],
                "fuel_flow_kg_s": true["fuel_flow_kg_s"],
                "fuel_flow_measured": meas["fuel_flow_kg_s"],
                "vibration_g": true["vibration_g"],
                "vibration_measured": meas["vibration_g"],
                "power_kW": true["power_kW"],
                "health_true": hidden["health"],
                "bearing_deg": hidden["bearing"],
                "cooling_deg": hidden["cooling"],
                "lubrication_deg": hidden["lubrication"],
                "fuel_deg": hidden["fuel_system"],
                "air_deg": hidden["air_intake"],
                "ignition_deg": hidden["ignition"],
                "degradation": 1-hidden["health"]/100,
                "fault_type": classify_fault_from_state({
                    "bearing_deg": hidden["bearing"],
                    "cooling_deg": hidden["cooling"],
                    "lubrication_deg": hidden["lubrication"],
                    "fuel_deg": hidden["fuel_system"],
                    "air_deg": hidden["air_intake"],
                    "ignition_deg": hidden["ignition"],
                }),
            }
            row["RUL_cycles_true"] = max(0, failure_cycle-t)
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

FEATURES = [
    "rpm_measured", "map_measured", "cht_measured",
    "oil_temp_measured", "oil_pressure_measured",
    "## Contributors
- [Your Name](https://github.com/your-username) — lead dev
- [Friend's Name](https://github.com/friends-username) — contributed the UKF/ML modulestorque_measured", "fuel_flow_measured", "vibration_measured",
    "throttle", "load", "altitude_m",
    "rpm_residual", "map_residual", "cht_residual",
    "oil_temp_residual", "oil_pressure_residual",
    "torque_residual", "vibration_rms",
    "temp_rate", "vibration_trend", "pressure_deviation",
    "thermal_stress",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    pairs = [
        ("rpm_measured", "rpm"),
        ("map_measured", "map_kpa"),
        ("cht_measured", "cht_C"),
        ("oil_temp_measured", "oil_temp_C"),
        ("oil_pressure_measured", "oil_pressure_bar"),
        ("torque_measured", "torque_Nm"),
    ]
    for m, t in pairs:
        x[m.replace("_measured", "_residual")] = x[m]-x[t]

    x["vibration_rms"] = (
        x["vibration_measured"].rolling(15, min_periods=1).apply(
            lambda z: float(np.sqrt(np.mean(np.square(z)))), raw=True
        )
    )
    x["temp_rate"] = x["cht_measured"].diff().fillna(0)
    x["vibration_trend"] = x["vibration_measured"].rolling(20, min_periods=2).mean().diff().fillna(0)
    x["pressure_deviation"] = x["map_measured"] - x["map_measured"].rolling(20, min_periods=1).mean()
    x["thermal_stress"] = (
        np.clip(x["cht_measured"]-160, 0, None)/50
        + np.clip(x["oil_temp_measured"]-100, 0, None)/30
    )
    x = x.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    return x


# ============================================================
# 8. AI MODELS
# ============================================================

@dataclass
class ModelBundle:
    scaler: Any = None
    fault_model: Any = None
    rul_model: Any = None
    anomaly_model: Any = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, seed: int = 42) -> ModelBundle:
    x = engineer_features(df)
    available = [c for c in FEATURES if c in x.columns]
    X = x[available].astype(float).values

    # Chronological split by engine units to avoid temporal leakage.
    units = sorted(x["unit"].unique())
    split = max(1, int(len(units)*0.75))
    train_units = set(units[:split])
    tr = x[x["unit"].isin(train_units)]
    te = x[~x["unit"].isin(train_units)]
    if te.empty:
        te = tr.tail(max(100, len(tr)//5))

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[available])
    Xte = scaler.transform(te[available])

    fault = RandomForestClassifier(
        n_estimators=180, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=seed, n_jobs=2
    )
    fault.fit(Xtr, tr["fault_type"])

    pred_fault = fault.predict(Xte)
    acc = accuracy_score(te["fault_type"], pred_fault)
    p, r, f1, _ = precision_recall_fscore_support(
        te["fault_type"], pred_fault, average="weighted", zero_division=0
    )

    rul = RandomForestRegressor(
        n_estimators=220, max_depth=12, min_samples_leaf=3,
        random_state=seed, n_jobs=2
    )
    rul.fit(Xtr, tr["RUL_cycles_true"])
    pred_rul = np.maximum(0, rul.predict(Xte))
    mae = mean_absolute_error(te["RUL_cycles_true"], pred_rul)
    rmse = np.sqrt(mean_squared_error(te["RUL_cycles_true"], pred_rul))
    r2 = r2_score(te["RUL_cycles_true"], pred_rul)

    anomaly = IsolationForest(
        n_estimators=180, contamination=0.08, random_state=seed
    )
    anomaly.fit(Xtr)

    imp = dict(zip(available, fault.feature_importances_))

    return ModelBundle(
        scaler=scaler,
        fault_model=fault,
        rul_model=rul,
        anomaly_model=anomaly,
        feature_importance=imp,
        validation={
            "fault_accuracy": float(acc),
            "fault_precision": float(p),
            "fault_recall": float(r),
            "fault_f1": float(f1),
            "rul_mae_cycles": float(mae),
            "rul_rmse_cycles": float(rmse),
            "rul_r2": float(r2),
            "train_units": sorted(train_units),
            "test_units": sorted(set(te["unit"])),
            "features": available,
        }
    )


# ============================================================
# 9. HEALTH / ANOMALY / DIAGNOSIS
# ============================================================

def prototype_health_index(
    ukf_health: float,
    vibration: float,
    cht: float,
    oil_temp: float,
    oil_pressure: float,
    anomaly_score: float,
) -> float:
    """
    Configurable prototype index, not a certification metric.
    """
    thermal_penalty = (
        0.55*max(0, cht-175)/55
        + 0.30*max(0, oil_temp-105)/30
    )
    vib_penalty = max(0, vibration-0.45)/1.0
    pressure_penalty = max(0, 1.8-oil_pressure)/1.8
    ai_penalty = np.clip(anomaly_score, 0, 1)

    hi = (
        0.55*ukf_health
        + 45*(1-0.55*thermal_penalty)
        + 35*(1-0.50*vib_penalty)
        + 30*(1-0.60*pressure_penalty)
        - 25*ai_penalty
    ) / 1.65
    return float(np.clip(hi, 0, 100))


def status_from_health(health: float, anomaly: float) -> str:
    if health < 30 or anomaly > 0.90:
        return "CRITICAL"
    if health < 55 or anomaly > 0.65:
        return "ABNORMAL"
    if health < 75 or anomaly > 0.40:
        return "WARNING"
    return "NORMAL"


def diagnose(row: pd.Series) -> Tuple[str, float, List[str]]:
    scores = {
        "Bearing degradation": max(0, row.get("vibration_measured", 0)-0.45)
                           + 0.5*max(0, row.get("oil_temp_measured", 0)-100)/30,
        "Cooling degradation": max(0, row.get("cht_measured", 0)-175)/40
                              + 0.3*max(0, row.get("oil_temp_measured", 0)-105)/30,
        "Lubrication degradation": max(0, 1.8-row.get("oil_pressure_measured", 2))/1.8
                                   + 0.35*max(0, row.get("oil_temp_measured", 0)-105)/30,
        "Fuel-system degradation": abs(row.get("fuel_flow_residual", 0))/0.01
                                   + 0.25*abs(row.get("rpm_residual", 0))/100,
        "Air-intake restriction": max(0, -row.get("map_residual", 0))/8
                                   + max(0, 60-row.get("map_measured", 80))/60,
        "Ignition degradation": abs(row.get("rpm_residual", 0))/120
                                + max(0, row.get("vibration_measured", 0)-0.7)/1.0,
    }
    name, raw = max(scores.items(), key=lambda kv: kv[1])
    confidence = float(np.clip(0.52 + 0.18*raw, 0.52, 0.97))
    evidence = []

    if "Bearing" in name:
        if row.get("vibration_measured", 0) > 0.45:
            evidence.append("vibration trend elevated")
        if row.get("oil_temp_measured", 0) > 100:
            evidence.append("oil temperature elevated")
    elif "Cooling" in name:
        if row.get("cht_measured", 0) > 175:
            evidence.append("cylinder/head temperature elevated")
        if row.get("oil_temp_measured", 0) > 105:
            evidence.append("oil temperature elevated")
    elif "Lubrication" in name:
        if row.get("oil_pressure_measured", 2) < 1.8:
            evidence.append("oil pressure reduced")
        if row.get("oil_temp_measured", 0) > 105:
            evidence.append("oil temperature elevated")
    elif "Fuel" in name:
        evidence.append("fuel-flow/RPM relationship abnormal")
    elif "Air" in name:
        evidence.append("manifold pressure deviation")
    else:
        evidence.append("RPM/vibration residual pattern abnormal")

    return name, confidence, evidence


# ============================================================
# 10. RUL ESTIMATION
# ============================================================

def estimate_rul(model: ModelBundle, current_row: pd.DataFrame) -> Tuple[float, float]:
    x = engineer_features(current_row)
    feats = model.validation["features"]
    X = model.scaler.transform(x[feats].tail(1))
    point = float(max(0, model.rul_model.predict(X)[0]))

    # Uncertainty from RF tree ensemble spread.
    tree_preds = np.array([
        est.predict(X)[0] for est in model.rul_model.estimators_
    ])
    sigma = float(max(1.0, np.std(tree_preds)))
    return point, sigma


# ============================================================
# 11. MISSION SIMULATION
# ============================================================

DEFAULT_MISSION = [
    ("Takeoff", 10, 0.82, 0.85),
    ("Climb", 30, 0.78, 0.80),
    ("Cruise", 120, 0.62, 0.60),
    ("Loiter", 240, 0.55, 0.52),
    ("High-load", 30, 0.85, 0.88),
    ("Descent", 20, 0.45, 0.40),
    ("Landing", 10, 0.65, 0.68),
]


@st.cache_data(show_spinner=False)
def mission_simulate(
    initial_health: float,
    altitude_m: float,
    ambient_delta_C: float,
    mission_minutes: float,
    throttle_scale: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    physics = AeroPistonPhysics(CFG)

    # Scale the default profile to requested mission duration.
    base_total = sum(p[1] for p in DEFAULT_MISSION)
    scale = max(0.05, mission_minutes/base_total)

    points = []
    hidden = {
        "rpm": 1100.0,
        "health": initial_health,
        "bearing": (100-initial_health)/100*0.45,
        "cooling": (100-initial_health)/100*0.30,
        "lubrication": (100-initial_health)/100*0.25,
        "fuel_system": (100-initial_health)/100*0.12,
        "air_intake": 0.0,
        "ignition": 0.0,
    }

    t_min = 0.0
    failed = False
    failure_phase = None

    for phase, base_min, throttle, load in DEFAULT_MISSION:
        dur = base_min*scale
        n = max(2, int(round(dur)))
        for j in range(n):
            controls = {
                "throttle": np.clip(throttle*throttle_scale, 0.1, 0.98),
                "load": np.clip(load*throttle_scale, 0.1, 0.98),
                "altitude_m": altitude_m,
                "ambient_delta_C": ambient_delta_C,
            }
            hidden = advance_degradation(hidden, controls, 1/60.0, rng)
            s = physics.step(controls, hidden, 1.0)

            risk = (
                0.25*max(0, hidden["health"]-100)/100
                + 0.32*max(0, s["cht_C"]-195)/45
                + 0.20*max(0, s["oil_temp_C"]-120)/25
                + 0.23*max(0, s["vibration_g"]-1.25)/1.0
            )
            if hidden["health"] < 20 or risk > 1.0 or s["oil_pressure_bar"] < 0.8:
                failed = True
                failure_phase = phase

            points.append({
                "time_min": t_min,
                "phase": phase,
                **s,
                "risk": float(np.clip(risk, 0, 1.5)),
            })
            t_min += 1.0

            if failed:
                break
        if failed:
            break

    out = pd.DataFrame(points)
    success = not failed and not out.empty and float(out["health"].iloc[-1]) > 20
    if out.empty:
        success = False

    return {
        "data": out,
        "success": bool(success),
        "failure_phase": failure_phase,
        "final_health": float(out["health"].iloc[-1]) if not out.empty else 0,
        "max_cht": float(out["cht_C"].max()) if not out.empty else 0,
        "max_oil_temp": float(out["oil_temp_C"].max()) if not out.empty else 0,
        "max_vibration": float(out["vibration_g"].max()) if not out.empty else 0,
        "fuel_kg": float(out["fuel_flow_kg_s"].sum()*60) if not out.empty else 0,
    }


@st.cache_data(show_spinner=False)
def monte_carlo_reliability(
    initial_health: float,
    altitude_m: float,
    ambient_delta_C: float,
    mission_minutes: float,
    throttle_scale: float,
    runs: int = 300,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    successes = 0
    phase_failures = {}
    outcomes = []

    for i in range(runs):
        h = np.clip(initial_health + rng.normal(0, 2.5), 5, 100)
        alt = max(0, altitude_m + rng.normal(0, 250))
        amb = ambient_delta_C + rng.normal(0, 3)
        throttle = max(0.5, throttle_scale + rng.normal(0, 0.025))

        res = mission_simulate(
            h, alt, amb, mission_minutes, throttle, seed=int(rng.integers(0, 2**31-1))
        )
        successes += int(res["success"])
        if not res["success"]:
            phase_failures[res["failure_phase"] or "Unknown"] = (
                phase_failures.get(res["failure_phase"] or "Unknown", 0)+1
            )
        outcomes.append(res["final_health"])

    reliability = successes/runs
    se = np.sqrt(max(reliability*(1-reliability)/runs, 1e-12))
    ci_low = max(0, reliability-1.96*se)
    ci_high = min(1, reliability+1.96*se)

    phase_risk = {
        k: v/runs for k, v in phase_failures.items()
    }

    return {
        "reliability": float(reliability),
        "failure_probability": float(1-reliability),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "phase_risk": phase_risk,
        "final_health_mean": float(np.mean(outcomes)),
        "final_health_std": float(np.std(outcomes)),
        "runs": runs,
    }


# ============================================================
# 12. WHAT-IF + OPTIMIZATION
# ============================================================

def what_if_compare(base: Dict[str, float], scenario: Dict[str, float], runs=180):
    a = monte_carlo_reliability(
        base["health"], base["altitude"], base["ambient"],
        base["duration"], base["throttle"], runs, 42
    )
    b = monte_carlo_reliability(
        scenario["health"], scenario["altitude"], scenario["ambient"],
        scenario["duration"], scenario["throttle"], runs, 43
    )
    return a, b


def optimize_throttle(base: Dict[str, float], runs=140):
    values = np.linspace(0.55, 0.90, 8)
    results = []
    for v in values:
        r = monte_carlo_reliability(
            base["health"], base["altitude"], base["ambient"],
            base["duration"], float(v), runs, 100+int(v*1000)
        )
        results.append((v, r["reliability"]))
    best = max(results, key=lambda x: x[1])
    return results, best


# ============================================================
# 13. CRYPTOGRAPHIC TELEMETRY INTEGRITY
# ============================================================

class TelemetryIntegrity:
    """
    Cryptographic provenance layer.

    SHA-256 creates an append-only hash chain.
    HMAC-SHA256 authenticates the chain when a secret is available.

    This does NOT make the telemetry confidential; it provides integrity
    and tamper-evidence for a prototype telemetry/logging pipeline.
    """

    def __init__(self, secret: Optional[str] = None):
        self.secret = (secret or os.environ.get(
            "DT_HMAC_SECRET", "SIH2026-DEMO-CHANGE-ME"
        )).encode("utf-8")
        self.previous_hash = "0"*64

    def sign(self, record: Dict[str, Any]) -> Dict[str, str]:
        payload = json.dumps(record, sort_keys=True, default=str).encode()
        chain_input = self.previous_hash.encode()+payload
        sha = hashlib.sha256(chain_input).hexdigest()
        mac = hmac.new(self.secret, sha.encode(), hashlib.sha256).hexdigest()
        self.previous_hash = sha
        return {"sha256": sha, "hmac_sha256": mac}

    def verify_chain(self, records: List[Dict[str, Any]]) -> bool:
        previous = "0"*64
        for item in records:
            record = item["record"]
            payload = json.dumps(record, sort_keys=True, default=str).encode()
            sha = hashlib.sha256(previous.encode()+payload).hexdigest()
            mac = hmac.new(self.secret, sha.encode(), hashlib.sha256).hexdigest()
            if sha != item["sha256"] or not hmac.compare_digest(mac, item["hmac_sha256"]):
                return False
            previous = sha
        return True


# ============================================================
# 14. CSV MAPPING
# ============================================================

ALIASES = {
    "rpm_measured": ["rpm", "engine_rpm", "n", "speed"],
    "map_measured": ["map", "manifold_pressure", "map_kpa", "manifold_pressure_kpa"],
    "cht_measured": ["cht", "cht_c", "cylinder_head_temperature", "head_temp"],
    "oil_temp_measured": ["oil_temp", "oil_temperature", "oil_temp_c"],
    "oil_pressure_measured": ["oil_pressure", "oil_pressure_bar"],
    "torque_measured": ["torque", "torque_nm"],
    "fuel_flow_measured": ["fuel_flow", "fuel_flow_kg_s", "fuel_rate"],
    "vibration_measured": ["vibration", "vibration_g", "vib"],
    "throttle": ["throttle", "throttle_pct", "throttle_percent"],
    "load": ["load", "engine_load", "load_pct"],
    "altitude_m": ["altitude", "altitude_m", "height"],
}


def auto_map_columns(columns):
    lower = {str(c).lower().strip(): c for c in columns}
    mapping = {}
    for target, candidates in ALIASES.items():
        for cand in candidates:
            if cand in lower:
                mapping[target] = lower[cand]
                break
    return mapping


def normalize_uploaded_csv(raw: pd.DataFrame):
    mapping = auto_map_columns(raw.columns)
    out = pd.DataFrame()
    for target, source in mapping.items():
        out[target] = pd.to_numeric(raw[source], errors="coerce")

    # Required defaults are explicitly marked as unavailable/assumed.
    if "throttle" not in out:
        out["throttle"] = 0.60
    else:
        if out["throttle"].median() > 1.5:
            out["throttle"] /= 100.0

    if "load" not in out:
        out["load"] = out["throttle"]

    if "altitude_m" not in out:
        out["altitude_m"] = 1500.0

    # Map physical values into a synthetic-compatible structure only where
    # a true measured column is present.
    for col in [
        "rpm_measured", "map_measured", "cht_measured",
        "oil_temp_measured", "oil_pressure_measured",
        "torque_measured", "fuel_flow_measured", "vibration_measured"
    ]:
        if col not in out:
            out[col] = np.nan

    out = out.interpolate(limit_direction="both").ffill().bfill()
    return out, mapping


# ============================================================
# 15. PLOTS
# ============================================================

def line_plot(df, x, ys, title, ytitle):
    fig = go.Figure()
    for y, name in ys:
        if y in df:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[y], mode="lines", name=name
            ))
    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title=ytitle,
        height=340, margin=dict(l=30, r=20, t=50, b=30),
        template="plotly_dark"
    )
    return fig


def engine_3d(rpm: float, status: str):
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(-1.0, 1.0, 30)
    T, Z = np.meshgrid(theta, z)
    r = 0.62
    X = r*np.cos(T)
    Y = r*np.sin(T)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, opacity=0.55, showscale=False,
        name="Cylinder"
    ))

    angle = (rpm/60.0)*2*np.pi
    crank_x = np.array([0, 0.7*np.cos(angle)])
    crank_y = np.array([0, 0.7*np.sin(angle)])
    crank_z = np.array([0, 0])

    fig.add_trace(go.Scatter3d(
        x=crank_x, y=crank_y, z=crank_z,
        mode="lines+markers", line=dict(width=8),
        name="Crank/rod"
    ))

    fig.update_layout(
        title=f"ENGINE SCHEMATIC • {rpm:.0f} RPM • {status}",
        height=430, template="plotly_dark",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="cube"
        ),
        margin=dict(l=0, r=0, t=45, b=0)
    )
    return fig


# ============================================================
# 16. FULL DIGITAL-TWIN RUN
# ============================================================

@st.cache_data(show_spinner=False)
def run_twin(df: pd.DataFrame, controls: Dict[str, float]):
    physics = AeroPistonPhysics(CFG)
    rng = np.random.default_rng(int(controls.get("seed", 42)))

    first = df.iloc[0] if len(df) else None
    x0 = np.array([
        float(first["rpm_measured"]) if first is not None else 1800,
        float(first["map_measured"]) if first is not None else 70,
        float(first["cht_measured"]) if first is not None else 155,
        float(first["oil_temp_measured"]) if first is not None else 90,
        float(first["oil_pressure_measured"]) if first is not None else 3,
        float(first["torque_measured"]) if first is not None else 80,
        90.0
    ])
    ukf = UKF(physics, x0)

    outputs = []
    innovations = []

    for _, row in df.iterrows():
        c = {
            "throttle": float(np.clip(row.get("throttle", controls["throttle"]), 0, 1)),
            "load": float(np.clip(row.get("load", controls["load"]), 0, 1)),
            "altitude_m": float(row.get("altitude_m", controls["altitude"])),
            "ambient_delta_C": float(controls.get("ambient", 0)),
        }

        ukf.predict(c, 1.0)

        z = np.array([
            row["rpm_measured"], row["map_measured"],
            row["cht_measured"], row["oil_temp_measured"],
            row["oil_pressure_measured"], row["torque_measured"]
        ])
        est, innovation, S = ukf.update(z)
        innovations.append(np.linalg.norm(innovation))

        out = row.to_dict()
        out["ukf_rpm"] = est[0]
        out["ukf_map"] = est[1]
        out["ukf_cht"] = est[2]
        out["ukf_oil_temp"] = est[3]
        out["ukf_oil_pressure"] = est[4]
        out["ukf_torque"] = est[5]
        out["ukf_health"] = est[6]
        out["ukf_innovation_norm"] = innovations[-1]
        out["ukf_health_std"] = float(np.sqrt(max(0, ukf.P[6,6])))
        outputs.append(out)

    result = pd.DataFrame(outputs)
    result["anomaly_raw"] = np.clip(
        (result["ukf_innovation_norm"] -
         result["ukf_innovation_norm"].rolling(25, min_periods=1).median()) /
        (result["ukf_innovation_norm"].rolling(25, min_periods=1).std().fillna(1)+1e-6),
        -3, 6
    )
    result["anomaly_score"] = 1/(1+np.exp(-result["anomaly_raw"]))
    result["health_index"] = [
        prototype_health_index(
            h, v, cht, ot, op, a
        )
        for h, v, cht, ot, op, a in zip(
            result["ukf_health"],
            result["vibration_measured"],
            result["cht_measured"],
            result["oil_temp_measured"],
            result["oil_pressure_measured"],
            result["anomaly_score"],
        )
    ]
    result["status"] = [
        status_from_health(h, a)
        for h, a in zip(result["health_index"], result["anomaly_score"])
    ]
    return result


# ============================================================
# 17. SELF TESTS
# ============================================================

def self_tests():
    rng = np.random.default_rng(1)
    physics = AeroPistonPhysics(CFG)

    s = physics.step(
        {"throttle": 0.65, "load": 0.65, "altitude_m": 1500, "ambient_delta_C": 0},
        {"rpm": 1800, "health": 95, "bearing": 0.02, "cooling": 0.02,
         "lubrication": 0.02, "fuel_system": 0.01, "air_intake": 0, "ignition": 0}
    )

    assert s["rpm"] > 0
    assert s["torque_Nm"] > 0
    assert s["power_kW"] > 0
    assert s["ambient_pressure_kPa"] > 0

    z = sensor_measure(s, rng)
    ukf = UKF(physics, np.array([
        s["rpm"], s["map_kpa"], s["cht_C"], s["oil_temp_C"],
        s["oil_pressure_bar"], s["torque_Nm"], 95.0
    ]))
    ukf.predict({"throttle": .65, "load": .65, "altitude_m": 1500, "ambient_delta_C": 0})
    est, innovation, cov = ukf.update(np.array([
        z["rpm"], z["map_kpa"], z["cht_C"], z["oil_temp_C"],
        z["oil_pressure_bar"], z["torque_Nm"]
    ]))
    assert est.shape == (7,)
    assert innovation.shape == (6,)
    assert cov.shape == (6, 6)

    integrity = TelemetryIntegrity("test-secret")
    signed = []
    for i in range(3):
        rec = {"i": i, "rpm": 2000+i}
        sig = integrity.sign(rec)
        signed.append({"record": rec, **sig})
    assert integrity.verify_chain(signed)

    return True


# ============================================================
# 18. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Aero-Piston Digital Twin | SIH 2026",
    page_icon="✈",
    layout="wide",
)

st.markdown("""
<style>
body { background:#06111f; }
.block-container { padding-top: 1rem; }
.metric-card {
    background: linear-gradient(135deg,#0b1c31,#0a1424);
    border:1px solid #173b61; border-radius:12px;
    padding:14px; text-align:center;
}
.small {font-size:0.82rem;color:#9db1c8;}
.warningbox {
    padding:12px;border-radius:10px;background:#261d0a;
    border:1px solid #715a18;
}
</style>
""", unsafe_allow_html=True)

st.title("✈ AERO-PISTON ENGINE DIGITAL TWIN")
st.caption(
    "SIH 2026 • Physics + Synthetic Telemetry + UKF + AI + RUL + Mission Reliability + Integrity"
)

with st.expander("SCIENTIFIC STATUS / IMPORTANT LIMITATIONS", expanded=False):
    st.warning(
        "SYNTHETIC DATA / SIMULATION RESULT / MODEL ESTIMATE. "
        "This prototype is NOT flight-certified, NOT an operational maintenance system, "
        "and NOT experimentally validated on a physical UAV engine."
    )
    st.write(NASA_REFERENCE_NOTE)
    st.write(
        "NASA C-MAPSS data are used conceptually for run-to-failure/prognostics methodology; "
        "the present aero-piston variables are generated from a transparent generic physics model."
    )

# Sidebar
st.sidebar.header("SYSTEM CONTROLS")
seed = st.sidebar.number_input("Random seed", 1, 999999, CFG.seed)
mode = st.sidebar.selectbox(
    "Mode", ["Live Synthetic Twin", "Offline Synthetic Dataset", "Mission / What-If", "CSV Upload"]
)
scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()))

st.sidebar.markdown("### Mission inputs")
initial_health = st.sidebar.slider("Initial Health (%)", 20, 100, 82)
altitude = st.sidebar.slider("Altitude (m)", 0, 10000, 5500)
ambient = st.sidebar.slider("Ambient temperature offset (°C)", -15, 35, 10)
duration = st.sidebar.slider("Mission duration (min)", 30, 720, 360)
throttle = st.sidebar.slider("Throttle", 0.30, 0.95, 0.70, 0.01)
load = st.sidebar.slider("Engine load", 0.25, 0.95, 0.70, 0.01)

# Session state
if "data" not in st.session_state:
    st.session_state.data = None
if "models" not in st.session_state:
    st.session_state.models = None
if "twin" not in st.session_state:
    st.session_state.twin = None
if "integrity_records" not in st.session_state:
    st.session_state.integrity_records = []

# ------------------------------------------------------------
# Data generation / upload
# ------------------------------------------------------------

if mode in ["Live Synthetic Twin", "Offline Synthetic Dataset"]:
    if st.button("GENERATE / REFRESH SYNTHETIC DATA", type="primary"):
        with st.spinner("Generating correlated physics-based synthetic telemetry..."):
            df = generate_synthetic_dataset(
                n_engines=12, cycles=500, seed=int(seed)
            )
            st.session_state.data = df
            st.session_state.models = train_models(df, int(seed))
            st.session_state.twin = run_twin(
                df[df["unit"] == 1].tail(500),
                {"throttle": throttle, "load": load, "altitude": altitude,
                 "ambient": ambient, "seed": seed}
            )
            st.success("Synthetic dataset + UKF + AI models generated.")

elif mode == "CSV Upload":
    upload = st.file_uploader("Upload engine telemetry CSV", type=["csv"])
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            mapped, mapping = normalize_uploaded_csv(raw)
            st.session_state.upload_mapping = mapping
            st.session_state.uploaded = mapped
            st.success(f"Loaded {len(raw):,} rows.")
            st.write("Detected column mapping:")
            st.json(mapping)
            st.caption(
                "Missing physical measurements are shown as unavailable; "
                "the system does not silently turn them into real measurements."
            )
        except Exception as e:
            st.error(f"CSV processing error: {e}")

# Default generation on first page load
if st.session_state.data is None and mode != "CSV Upload":
    with st.spinner("First-time setup: generating synthetic telemetry and training AI models (one-time, cached after this)..."):
        df = generate_synthetic_dataset(12, 500, int(seed))
        st.session_state.data = df
        st.session_state.models = train_models(df, int(seed))
        st.session_state.twin = run_twin(
            df[df["unit"] == 1],
            {"throttle": throttle, "load": load, "altitude": altitude,
             "ambient": ambient, "seed": seed}
        )

df = st.session_state.data
models = st.session_state.models
twin = st.session_state.twin

# ------------------------------------------------------------
# TOP STATUS
# ------------------------------------------------------------

if twin is not None and len(twin):
    latest = twin.iloc[-1]
    current_health = float(latest["health_index"])
    anomaly = float(latest["anomaly_score"])
    status = str(latest["status"])
    diagnosis, confidence, evidence = diagnose(latest)
else:
    latest = None
    current_health = float(initial_health)
    anomaly = 0.0
    status = "NORMAL"
    diagnosis, confidence, evidence = "Normal", 0.60, []

base_for_rul = twin.tail(120) if twin is not None else df[df.unit == 1].tail(120)
try:
    rul_cycles, rul_sigma = estimate_rul(models, base_for_rul)
except Exception:
    rul_cycles, rul_sigma = 0.0, 0.0

rul_hours = rul_cycles/60.0
rul_sigma_h = rul_sigma/60.0

mission_base = {
    "health": initial_health,
    "altitude": altitude,
    "ambient": ambient,
    "duration": duration,
    "throttle": throttle,
}

reliability = monte_carlo_reliability(
    initial_health, altitude, ambient, duration, throttle, 250, int(seed)
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ENGINE STATUS", status)
c2.metric("HEALTH INDEX", f"{current_health:.1f}%")
c3.metric("RUL", f"{rul_hours:.1f} ± {rul_sigma_h:.1f} h")
c4.metric("MISSION RELIABILITY", f"{100*reliability['reliability']:.1f}%")
c5.metric("FAILURE PROBABILITY", f"{100*reliability['failure_probability']:.1f}%")

st.caption(
    "All headline values are prototype model estimates. "
    "They are not certified limits or experimentally validated engine measurements."
)

# ------------------------------------------------------------
# DASHBOARD TABS
# ------------------------------------------------------------

tabs = st.tabs([
    "LIVE TWIN", "PHYSICS / UKF", "AI / FAULTS", "RUL",
    "MISSION", "WHAT-IF", "3D ENGINE", "SECURITY", "VALIDATION", "EXPORT"
])

# ============================================================
# LIVE TWIN
# ============================================================

with tabs[0]:
    st.subheader("Real-time / streaming digital twin")

    if st.button("▶ START 10-SECOND STREAM"):
        progress = st.progress(0)
        status_box = st.empty()
        chart_box = st.empty()

        stream = []
        physics = AeroPistonPhysics(CFG)
        rng = np.random.default_rng(int(seed))
        hidden = {
            "rpm": 1200.0, "health": initial_health,
            "bearing": (100-initial_health)/100*0.5,
            "cooling": (100-initial_health)/100*0.3,
            "lubrication": (100-initial_health)/100*0.25,
            "fuel_system": 0.03,
            "air_intake": 0.0, "ignition": 0.0,
        }

        for k in range(20):
            controls = {
                "throttle": throttle,
                "load": load,
                "altitude_m": altitude,
                "ambient_delta_C": ambient,
            }
            if scenario in SCENARIOS:
                spec = SCENARIOS[scenario]
                for key in ["bearing", "cooling", "lubrication", "fuel_system",
                            "air_intake", "ignition"]:
                    if key in spec:
                        hidden[key] = max(hidden[key], spec[key])

            hidden = advance_degradation(hidden, controls, 1/3600, rng)
            true = physics.step(controls, hidden, 0.5)
            meas = sensor_measure(true, rng)
            stream.append({
                "t_s": k*0.5, "rpm": meas["rpm"],
                "cht_C": meas["cht_C"], "oil_temp_C": meas["oil_temp_C"],
                "oil_pressure_bar": meas["oil_pressure_bar"],
                "vibration_g": meas["vibration_g"],
                "health": hidden["health"],
            })
            status_box.info(
                f"STREAM {k+1:02d}/20 • RPM {meas['rpm']:.0f} • "
                f"CHT {meas['cht_C']:.1f} °C • VIB {meas['vibration_g']:.2f} g"
            )
            progress.progress((k+1)/20)
            chart_box.plotly_chart(
                line_plot(pd.DataFrame(stream), "t_s",
                          [("rpm", "RPM"), ("health", "Health")],
                          "Streaming telemetry", "Value"),
                use_container_width=True
            )
            time.sleep(0.15)

    if twin is not None:
        st.plotly_chart(
            line_plot(
                twin, "cycle",
                [("rpm_measured", "Noisy RPM"),
                 ("ukf_rpm", "UKF RPM"),
                 ("rpm", "True RPM")],
                "Measured vs true vs UKF RPM", "RPM"
            ), use_container_width=True
        )
        st.plotly_chart(
            line_plot(
                twin, "cycle",
                [("cht_measured", "Measured CHT"),
                 ("ukf_cht", "UKF CHT"),
                 ("cht_C", "True CHT")],
                "Thermal state estimation", "°C"
            ), use_container_width=True
        )

        st.dataframe(
            twin.tail(12)[[
                "cycle", "rpm_measured", "ukf_rpm", "cht_measured",
                "ukf_cht", "vibration_measured", "health_index",
                "anomaly_score", "status"
            ]],
            use_container_width=True
        )

# ============================================================
# PHYSICS / UKF
# ============================================================

with tabs[1]:
    st.subheader("Physics model + UKF state estimation")

    st.markdown("""
**Core equations**

- Air density: `ρ = P / (R T)`
- Angular velocity: `ω = 2πN / 60`
- Mechanical power: `P = Tω`
- Four-stroke air throughput is approximated from displacement × RPM / 120 × volumetric efficiency.
- Thermal and degradation terms are parameterized empirical approximations.
- UKF estimates nonlinear hidden states without requiring an analytic Jacobian.
""")

    if twin is not None:
        st.plotly_chart(
            line_plot(
                twin, "cycle",
                [("rpm_measured", "Sensor"),
                 ("ukf_rpm", "UKF"),
                 ("rpm", "True")],
                "RPM estimation", "RPM"
            ), use_container_width=True
        )

        st.plotly_chart(
            line_plot(
                twin, "cycle",
                [("map_measured", "Sensor"),
                 ("ukf_map", "UKF"),
                 ("map_kpa", "True")],
                "Manifold pressure estimation", "kPa"
            ), use_container_width=True
        )

        st.plotly_chart(
            line_plot(
                twin, "cycle",
                [("ukf_health", "UKF health"),
                 ("health_true", "Synthetic ground truth"),
                 ("health_index", "Prototype HI")],
                "Hidden degradation / health estimation", "%"
            ), use_container_width=True
        )

        rmse_rpm = float(np.sqrt(np.mean((twin["ukf_rpm"]-twin["rpm"])**2)))
        rmse_temp = float(np.sqrt(np.mean((twin["ukf_cht"]-twin["cht_C"])**2)))
        st.write({
            "UKF RPM RMSE": round(rmse_rpm, 3),
            "UKF CHT RMSE": round(rmse_temp, 3),
            "Mean UKF health uncertainty": round(float(twin["ukf_health_std"].mean()), 3)
        })

# ============================================================
# AI / FAULTS
# ============================================================

with tabs[2]:
    st.subheader("AI anomaly detection + fault diagnosis")

    st.write(
        "Hybrid anomaly evidence = nonlinear physics/UKF innovation + statistical "
        "residual behaviour + Isolation Forest. The classifier is trained only on "
        "synthetic labels generated from interpretable fault signatures."
    )

    if latest is not None:
        st.metric("Predicted fault", diagnosis)
        st.metric("Prototype confidence", f"{confidence*100:.1f}%")
        st.write("Evidence:", ", ".join(evidence) if evidence else "No strong fault evidence.")

    if models is not None:
        imp = pd.Series(models.feature_importance).sort_values(ascending=False).head(12)
        st.bar_chart(imp)

        st.write("Validation:")
        st.json(models.validation)

    if twin is not None:
        st.plotly_chart(
            line_plot(
                twin, "cycle",
                [("anomaly_score", "Hybrid anomaly score"),
                 ("health_index", "Health Index")],
                "Anomaly / health trajectory", "Score / %"
            ), use_container_width=True
        )

# ============================================================
# RUL
# ============================================================

with tabs[3]:
    st.subheader("Remaining Useful Life")

    st.metric("Estimated RUL", f"{rul_hours:.2f} ± {rul_sigma_h:.2f} hours")
    st.metric("Estimated RUL", f"{rul_cycles:.1f} ± {rul_sigma:.1f} cycles")

    st.caption(
        "RUL is learned from synthetic run-to-failure trajectories. "
        "The uncertainty is estimated from the spread of Random Forest trees."
    )

    if twin is not None:
        temp = twin.copy()
        temp["RUL_proxy_h"] = np.maximum(
            0, (temp["cycle"].max()-temp["cycle"])/60
        )
        st.plotly_chart(
            line_plot(
                temp, "cycle",
                [("health_index", "Health Index"),
                 ("RUL_proxy_h", "Synthetic RUL trajectory")],
                "Health / RUL trend", "Health / hours"
            ), use_container_width=True
        )

# ============================================================
# MISSION
# ============================================================

with tabs[4]:
    st.subheader("Mission simulation + Monte Carlo reliability")

    m1, m2, m3 = st.columns(3)
    m1.metric("Reliability", f"{reliability['reliability']*100:.2f}%")
    m2.metric("95% CI", f"{reliability['ci_low']*100:.1f}–{reliability['ci_high']*100:.1f}%")
    m3.metric("Failure probability", f"{reliability['failure_probability']*100:.2f}%")

    if st.button("RUN DETERMINISTIC MISSION"):
        res = mission_simulate(
            initial_health, altitude, ambient, duration, throttle, int(seed)
        )
        st.session_state.mission = res

    if "mission" in st.session_state:
        res = st.session_state.mission
        md = res["data"]
        st.write({
            "Mission success": res["success"],
            "Failure phase": res["failure_phase"],
            "Final health": round(res["final_health"], 2),
            "Maximum CHT": round(res["max_cht"], 2),
            "Maximum oil temperature": round(res["max_oil_temp"], 2),
            "Maximum vibration": round(res["max_vibration"], 3),
            "Estimated fuel used (kg)": round(res["fuel_kg"], 2),
        })
        st.plotly_chart(
            line_plot(md, "time_min",
                      [("rpm", "RPM"), ("health", "Health")],
                      "Mission engine evolution", "Value"),
            use_container_width=True
        )
        st.plotly_chart(
            line_plot(md, "time_min",
                      [("cht_C", "CHT"), ("oil_temp_C", "Oil Temp"),
                       ("vibration_g", "Vibration")],
                      "Mission thermal/vibration load", "Value"),
            use_container_width=True
        )

    st.write("Phase-specific simulated risk:")
    if reliability["phase_risk"]:
        st.dataframe(
            pd.DataFrame([
                {"Phase": k, "Failure probability contribution": v*100}
                for k, v in reliability["phase_risk"].items()
            ]),
            use_container_width=True
        )
    else:
        st.info("No failures observed in this Monte-Carlo sample.")

# ============================================================
# WHAT-IF
# ============================================================

with tabs[5]:
    st.subheader('What happens if...?')

    w_health = st.slider("Scenario B health (%)", 20, 100, max(20, initial_health-15))
    w_throttle = st.slider("Scenario B throttle", 0.30, 0.95, min(0.95, throttle+0.12), 0.01)
    w_alt = st.slider("Scenario B altitude (m)", 0, 10000, min(10000, altitude+1500))
    w_amb = st.slider("Scenario B ambient offset (°C)", -15, 35, min(35, ambient+10))
    w_duration = st.slider("Scenario B mission duration (min)", 30, 720, min(720, duration+60))

    if st.button("COMPARE SCENARIOS", type="primary"):
        base = {
            "health": initial_health, "altitude": altitude,
            "ambient": ambient, "duration": duration, "throttle": throttle
        }
        alt = {
            "health": w_health, "altitude": w_alt,
            "ambient": w_amb, "duration": w_duration, "throttle": w_throttle
        }
        a, b = what_if_compare(base, alt, runs=180)

        comparison = pd.DataFrame({
            "Metric": ["Reliability", "Failure probability", "CI low", "CI high", "Mean final health"],
            "Scenario A": [
                a["reliability"]*100, a["failure_probability"]*100,
                a["ci_low"]*100, a["ci_high"]*100, a["final_health_mean"]
            ],
            "Scenario B": [
                b["reliability"]*100, b["failure_probability"]*100,
                b["ci_low"]*100, b["ci_high"]*100, b["final_health_mean"]
            ]
        })
        st.dataframe(comparison, use_container_width=True)

        delta = (b["reliability"]-a["reliability"])*100
        if delta >= 0:
            st.success(f"Scenario B improves predicted reliability by {delta:.2f} percentage points.")
        else:
            st.warning(f"Scenario B reduces predicted reliability by {abs(delta):.2f} percentage points.")

    st.divider()
    st.subheader("Mission-aware throttle optimization")

    if st.button("OPTIMIZE THROTTLE"):
        with st.spinner("Running grid-search reliability optimization..."):
            results, best = optimize_throttle(mission_base, runs=100)
        odf = pd.DataFrame(results, columns=["Throttle", "Reliability"])
        odf["Reliability_percent"] = odf["Reliability"]*100
        st.line_chart(odf.set_index("Throttle")["Reliability_percent"])
        st.success(
            f"Best tested throttle = {best[0]*100:.0f}% "
            f"with predicted reliability = {best[1]*100:.2f}%"
        )
        st.caption(
            "This is a model-based grid search, not a certified flight-control recommendation."
        )

# ============================================================
# 3D ENGINE
# ============================================================

with tabs[6]:
    st.subheader("Lightweight 3D engine schematic")

    rpm_display = float(latest["ukf_rpm"]) if latest is not None else 2200
    st.plotly_chart(engine_3d(rpm_display, status), use_container_width=True)
    st.caption(
        "Visualization is a schematic, not CAD-accurate geometry. "
        "The crank/rod animation is driven by estimated RPM."
    )

# ============================================================
# SECURITY
# ============================================================

with tabs[7]:
    st.subheader("Cryptographic telemetry integrity")

    st.write(
        "Implemented: SHA-256 append-only hash chaining + HMAC-SHA256 authentication. "
        "This provides tamper-evidence/integrity; it does not provide data confidentiality."
    )

    if st.button("CREATE SIGNED TELEMETRY CHAIN"):
        integ = TelemetryIntegrity()
        records = []
        sample = twin.tail(20) if twin is not None else pd.DataFrame()
        for _, row in sample.iterrows():
            rec = {
                "cycle": int(row.get("cycle", 0)),
                "rpm": float(row.get("rpm_measured", 0)),
                "health_index": float(row.get("health_index", 0)),
                "status": str(row.get("status", "")),
            }
            sig = integ.sign(rec)
            records.append({"record": rec, **sig})
        st.session_state.integrity_records = records
        st.success("Telemetry chain signed.")

    records = st.session_state.integrity_records
    if records:
        verified = TelemetryIntegrity().verify_chain(records)
        st.metric("Chain integrity", "VALID" if verified else "FAILED")
        st.dataframe(pd.DataFrame([
            {"cycle": r["record"]["cycle"], "SHA-256": r["sha256"],
             "HMAC-SHA256": r["hmac_sha256"]}
            for r in records[-8:]
        ]), use_container_width=True)
        st.download_button(
            "DOWNLOAD SIGNED TELEMETRY JSON",
            data=json.dumps(records, indent=2),
            file_name="signed_telemetry_chain.json",
            mime="application/json"
        )

# ============================================================
# VALIDATION
# ============================================================

with tabs[8]:
    st.subheader("Validation and test evidence")

    if st.button("RUN SELF-TESTS"):
        try:
            self_tests()
            st.success("All core mathematical/integrity self-tests passed.")
        except Exception as e:
            st.error(f"Self-test failed: {e}")

    st.markdown("### Physics validation")
    st.write(
        "The model checks that increasing throttle/load generally increases RPM, "
        "power and thermal load; increasing altitude reduces ambient pressure/density; "
        "degradation reduces effective health/power and increases fault signatures."
    )

    st.markdown("### UKF validation")
    if twin is not None:
        ukf_rmse = {
            "RPM RMSE": float(np.sqrt(np.mean((twin["ukf_rpm"]-twin["rpm"])**2))),
            "MAP RMSE": float(np.sqrt(np.mean((twin["ukf_map"]-twin["map_kpa"])**2))),
            "CHT RMSE": float(np.sqrt(np.mean((twin["ukf_cht"]-twin["cht_C"])**2))),
            "Oil temp RMSE": float(np.sqrt(np.mean((twin["ukf_oil_temp"]-twin["oil_temp_C"])**2))),
        }
        st.json({k: round(v, 4) for k, v in ukf_rmse.items()})

    st.markdown("### AI validation")
    if models is not None:
        st.json(models.validation)

    st.markdown("### Limitations")
    st.write([
        "Synthetic aero-piston data are not a substitute for test-cell or flight telemetry.",
        "NASA C-MAPSS is turbofan reference data and is not directly mapped as piston-engine truth.",
        "RUL model performance on synthetic data does not establish operational accuracy.",
        "Mission reliability is Monte-Carlo model probability, not a certification probability.",
        "Thresholds are configurable prototype thresholds, not regulatory/engine-manufacturer limits.",
    ])

# ============================================================
# EXPORT
# ============================================================

with tabs[9]:
    st.subheader("Export results")

    if twin is not None:
        csv_bytes = twin.to_csv(index=False).encode("utf-8")
        st.download_button(
            "DOWNLOAD DIGITAL-TWIN CSV",
            data=csv_bytes,
            file_name="aero_piston_digital_twin_results.csv",
            mime="text/csv"
        )

    report = {
        "title": "Aero-Piston Engine Digital Twin — SIH 2026 Prototype",
        "data_status": "SYNTHETIC DATA",
        "validation_status": "NOT EXPERIMENTALLY VALIDATED",
        "current_health_index": current_health,
        "rul_hours": rul_hours,
        "rul_uncertainty_hours": rul_sigma_h,
        "diagnosis": diagnosis,
        "diagnosis_confidence": confidence,
        "mission_reliability": reliability["reliability"],
        "failure_probability": reliability["failure_probability"],
        "reliability_ci_95": [reliability["ci_low"], reliability["ci_high"]],
        "mission": mission_base,
        "recommendation": (
            "Prototype recommendation: investigate the variables supporting the "
            f"detected condition ({diagnosis}) before extended high-load operation."
        ),
        "scientific_note": NASA_REFERENCE_NOTE,
    }
    st.download_button(
        "DOWNLOAD JSON REPORT",
        data=json.dumps(report, indent=2),
        file_name="engine_health_mission_report.json",
        mime="application/json"
    )

    report_text = f"""
ENGINE HEALTH REPORT — SIH 2026 PROTOTYPE

DATA STATUS:
SYNTHETIC DATA / SIMULATION RESULT

Current Health Index: {current_health:.1f}%
Estimated RUL: {rul_hours:.1f} ± {rul_sigma_h:.1f} hours
Detected Condition: {diagnosis}
Fault Confidence: {confidence*100:.1f}%
Mission Reliability: {reliability['reliability']*100:.2f}%
Failure Probability: {reliability['failure_probability']*100:.2f}%
95% Reliability CI: {reliability['ci_low']*100:.2f}%–{reliability['ci_high']*100:.2f}%

Prototype recommendation:
Investigate {diagnosis} supporting variables before extended high-load operation.

IMPORTANT:
This report is model-based and not experimentally validated.
It is not flight-certified and must not be used as an operational maintenance limit.
"""
    st.download_button(
        "DOWNLOAD TEXT REPORT",
        data=report_text,
        file_name="engine_health_report.txt",
        mime="text/plain"
    )

# Footer
st.divider()
st.caption(
    "AERO-PISTON DIGITAL TWIN • SIH 2026 • SYNTHETIC / MODEL-BASED PROTOTYPE • NOT FLIGHT CERTIFIED"
)
