import json, joblib, optuna
import pandas as pd
from copy import deepcopy
import numpy as np
from pathlib import Path
from django.conf import settings


# ============================================
#  PATH SETTINGS
# ============================================
BASE_ML_PATH = Path(settings.BASE_DIR) / "quali_defect_app" / "ml_model"

# ============================================
#  LOAD MODEL + SCALER + FEATURE ORDER
# ============================================
feature_order_path = BASE_ML_PATH / "model1_feature_order.json"
bounds_path        = BASE_ML_PATH / "recommendation_bounds_model_1.json"

model_path         = BASE_ML_PATH / "model1.pkl"
scaler_path        = BASE_ML_PATH / "model1_scaler.pkl"

rf_model = joblib.load(model_path)
scaler_mh = joblib.load(scaler_path)

with open(feature_order_path, "r") as f:
    feature_order_mh = json.load(f)

with open(bounds_path, "r") as f:
    bounds_summary = json.load(f)

machine_bounds = bounds_summary.get("machine_bounds", {})


# ============================================
#  CONVERT INPUT → ORDERED DATAFRAME
# ============================================
def dict_to_df_ordered(input_dict):
    row = []
    for col in feature_order_mh:
        if col in input_dict:
            row.append(input_dict[col])
        else:
            # fallback to midpoint of bounds
            b = machine_bounds.get(col, {"min": 0, "max": 1})
            row.append((float(b["min"]) + float(b["max"])) / 2)
    return pd.DataFrame([row], columns=feature_order_mh)


# ============================================
#  FAIL PROBABILITY PREDICTOR
# ============================================
def predict_mh_prob(input_dict):

    df = dict_to_df_ordered(input_dict)

    # Scale
    df_scaled = scaler_mh.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_order_mh)

    # Predict
    prob = rf_model.predict_proba(df_scaled)[0][1]   # FAIL probability
    cls = int(rf_model.predict(df_scaled)[0])        # 0=PASS,1=FAIL

    return cls, float(prob)


# ============================================
#  OPTUNA OBJECTIVE
# ============================================
def make_mh_objective(current_input):

    def objective(trial):
        cand = {}

        for f in feature_order_mh:
            low = float(machine_bounds[f]["min"])
            high = float(machine_bounds[f]["max"])

            # categorical → integer
            if f in ["Machine_Level", "Maintenance_Indicator"]:
                cand[f] = trial.suggest_int(f, int(low), int(high))

            else:
                cand[f] = trial.suggest_float(f, low, high)

        _, prob = predict_mh_prob(cand)
        return prob

    return objective


# ============================================
#  GENERATE RECOMMENDATION SENTENCES
# ============================================
def numeric_recommendations(current_input, optimal_input):

    recs = {}

    for k in feature_order_mh:

        curr = float(current_input.get(k, 0.0))
        opt  = float(optimal_input.get(k, curr))
        diff = curr - opt

        if abs(diff) < 1e-6:
            continue

        if diff > 0:
            recs[k] = f"Reduce {k} by {round(diff, 2)}"
        else:
            recs[k] = f"Increase {k} by {round(abs(diff), 2)}"

    return recs


# ============================================
#  MAIN API — CALL THIS FROM views.py
# ============================================
# -------------------------------------------------------------------
# Only these features should be shown in UI recommendations
USER_EDITABLE_FEATURES = [
    "Machine_Level",
    "Maintenance_Indicator",
    "Air_Temperature(Kelvin)",
    "Process_Temperature(Kelvin)",
    "Rotational_Speed(rpm)",
    "Torque(Nm)",
    "Tool_Wear(min)",
    "Wear_Rate",
    "Thermal_Stress_Index",
    "Normalized_Wear_Rate",
]
# -------------------------------------------------------------------


def numeric_recommendations(current, best):
    """
    Generate text recommendations ONLY for user-editable features.
    Derived features are skipped automatically.
    """
    recs = {}

    for key in USER_EDITABLE_FEATURES:
        if key not in current or key not in best:
            continue

        old = float(current[key])
        new = float(best[key])
        diff = new - old

        if abs(diff) < 0.01:
            continue  # Skip tiny changes

        if diff > 0:
            recs[key] = f"Increase {key} by {round(diff, 2)}"
        else:
            recs[key] = f"Reduce {key} by {round(abs(diff), 2)}"

    return recs



def recommend_machine_health(current_input, n_trials=40):

    cls, fail_prob = predict_mh_prob(current_input)

    output = {
        "current_fail_probability": round(fail_prob * 100, 2),
        "current_class": "PASS" if cls == 0 else "FAIL",
    }

    # PASS → no optimization
    if cls == 0:
        output["recommendations"] = ["Machine is healthy. No changes required."]
        output["optimized_fail_probability"] = fail_prob * 100
        output["improvement"] = 0
        return output

    # Run Optuna optimization
    objective = make_mh_objective(current_input)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_prob = study.best_value

    # Build cleaned recommendations
    recs = numeric_recommendations(current_input, best_params)

    output["optimized_fail_probability"] = round(best_prob * 100, 2)
    output["improvement"] = round((fail_prob - best_prob) * 100, 2)
    output["recommendations"] = recs

    return output

