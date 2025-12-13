from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.contrib.auth.decorators import login_required
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import firebase_admin
import razorpay
from firebase_admin import firestore,auth
from django.views.decorators.csrf import csrf_exempt

from quali_defect_app.ml_model.model1_recommender import recommend_machine_health
from quali_defect_app.ml_model.model2_recommendations import get_model2_recommendations


# Firestore
db = firestore.client()


# ============================================================
#  ML MODELS PATH
# ============================================================
BASE_ML_PATH = Path(settings.BASE_DIR) / "quali_defect_app" / "ml_model"


# ============================================================
#  FIREBASE AUTH HELPERS
# ============================================================
def get_uid(request):
    """Returns Firebase UID stored in session, or None."""
    return request.session.get("firebase_uid")


def firebase_login_required(view_func):
    """Decorator to enforce Firebase login."""

    def wrapper(request, *args, **kwargs):
        if not get_uid(request):
            return redirect("login")
        return view_func(request, *args, **kwargs)

    return wrapper


# ============================================================
#  LOGIN / SIGNUP HTML PAGES
# ============================================================
def login_page(request):
    return render(request, "quali_defect_app/login.html")


def signup_page(request):
    return render(request, "quali_defect_app/signup.html")


def session_login(request):
    if request.method == "POST":
        data = json.loads(request.body)
        uid = data.get("uid")
        token = data.get("token")

        if not uid or not token:
            return JsonResponse({"error": "Missing UID or token"}, status=400)

        decoded = auth.verify_id_token(token)
        email = decoded.get("email")

        request.session["firebase_uid"] = uid

        # Fetch existing doc
        user_doc = db.collection("users").document(uid).get()

        # IF USER LOGGING IN FOR THE FIRST TIME → CREATE ENTRY WITH FREE CREDITS
        if not user_doc.exists:
            db.collection("users").document(uid).set({
                "email": email,
                "role": "user",
                "status": "active",
                "created_at": firestore.SERVER_TIMESTAMP,

                # ⭐ CREDIT SYSTEM
                "free_credits": 5,
                "paid_credits": 0,
                "total_predictions_used": 0,
                "plan": "free",
            })
        else:
            # If doc exists but missing credit fields → ensure they are added
            data = user_doc.to_dict()

            updates = {}
            if "free_credits" not in data:
                updates["free_credits"] = 5
            if "paid_credits" not in data:
                updates["paid_credits"] = 0
            if "total_predictions_used" not in data:
                updates["total_predictions_used"] = 0
            if "plan" not in data:
                updates["plan"] = "free"

            if updates:
                db.collection("users").document(uid).update(updates)

        return JsonResponse({"status": "ok", "uid": uid})

    return JsonResponse({"error": "Invalid request"}, status=400)


def pricing_page(request):
    return render(request, "quali_defect_app/pricing.html")


def logout_view(request):
    """Clears Firebase session."""
    request.session.flush()
    return redirect("index")


# ============================================================
#  STATIC PAGES
# ============================================================
def index(request):
    return render(request, "quali_defect_app/index.html")


def about(request):
    return render(request, "quali_defect_app/about.html")



def get_user_credit_info(uid):
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return {"free_credits": 0, "paid_credits": 0, "total_predictions_used": 0, "plan": "free"}

    data = doc.to_dict()
    return {
        "free_credits": data.get("free_credits", 0),
        "paid_credits": data.get("paid_credits", 0),
        "total_predictions_used": data.get("total_predictions_used", 0),
        "plan": data.get("plan", "free"),
    }


def deduct_prediction_credit(uid):
    info = get_user_credit_info(uid)

    free = info["free_credits"]
    paid = info["paid_credits"]

    if free > 0:
        free -= 1
    elif paid > 0:
        paid -= 1
    else:
        return False  # No credits left

    db.collection("users").document(uid).update({
        "free_credits": free,
        "paid_credits": paid,
        "total_predictions_used": info["total_predictions_used"] + 1
    })

    return True




# ============================================================
#  MODEL-1 LOADING
# ============================================================
# ----------------------------- MODEL-1 (updated: minimal inputs + derived features) -----------------------------
# load model + scaler + feature order as before
model1 = joblib.load(BASE_ML_PATH / "model1.pkl")
scaler1 = joblib.load(BASE_ML_PATH / "model1_scaler.pkl")

with open(BASE_ML_PATH / "model1_feature_order.json") as f:
    feature_order_model1 = json.load(f)

# Full feature list the model expects (unchanged)
MODEL1_REQUIRED = [
    "Machine_Level", "Maintenance_Indicator",
    "Air_Temperature(Kelvin)", "Process_Temperature(Kelvin)",
    "Rotational_Speed(rpm)", "Torque(Nm)", "Tool_Wear(min)",
    "Temp_Difference", "Speed_Torque_Ratio", "Wear_Rate",
    "Energy_Index", "Thermal_Stress_Index", "Torque_Wear_Product",
    "Speed_Temp_Interaction", "Normalized_Wear_Rate"
]

# The subset we ask from the user (10 fields)
MODEL1_USER_FIELDS = [
    "Air_Temperature(Kelvin)", "Process_Temperature(Kelvin)",
    "Rotational_Speed(rpm)", "Torque(Nm)", "Tool_Wear(min)",
    "Wear_Rate", "Thermal_Stress_Index", "Normalized_Wear_Rate"
]


# mapping for Firestore storage field names (keep human-readable original strings when saving)
MODEL1_FIELD_MAP = {
    "Machine_Level": "machine_level",
    "Maintenance_Indicator": "maintenance_indicator",
    "Air_Temperature(Kelvin)": "air_temperature_kelvin",
    "Process_Temperature(Kelvin)": "process_temperature_kelvin",
    "Rotational_Speed(rpm)": "rotational_speed_rpm",
    "Torque(Nm)": "torque_nm",
    "Tool_Wear(min)": "tool_wear_min",
    "Temp_Difference": "temp_difference",
    "Speed_Torque_Ratio": "speed_torque_ratio",
    "Wear_Rate": "wear_rate",
    "Energy_Index": "energy_index",
    "Thermal_Stress_Index": "thermal_stress_index",
    "Torque_Wear_Product": "torque_wear_product",
    "Speed_Temp_Interaction": "speed_temp_interaction",
    "Normalized_Wear_Rate": "normalized_wear_rate",
}

# sensible defaults (used by template autofill)
DEFAULT_MODEL1 = {
    "Machine_Level": "High",
    "Maintenance_Indicator": "Yes",
    "Air_Temperature(Kelvin)": 306,
    "Process_Temperature(Kelvin)": 317,
    "Rotational_Speed(rpm)": 1195,
    "Torque(Nm)": 61,
    "Tool_Wear(min)": 249,
    "Wear_Rate": 208,
    "Thermal_Stress_Index": 642,
    "Normalized_Wear_Rate": 2.83,
}


def _compute_model1_derived(user_inputs):
    """
    Accepts a dict with keys in MODEL1_USER_FIELDS (strings from the form).
    Returns a dict of all MODEL1_REQUIRED features (ready for prediction).
    All numeric conversions happen here; missing/invalid numbers raise ValueError.
    """
    # copy input to avoid mutation
    u = dict(user_inputs)

    # Convert numeric fields (string -> float)
    numeric_fields = [
        "Air_Temperature(Kelvin)", "Process_Temperature(Kelvin)",
        "Rotational_Speed(rpm)", "Torque(Nm)", "Tool_Wear(min)",
        "Wear_Rate", "Thermal_Stress_Index", "Normalized_Wear_Rate"
    ]
    for f in numeric_fields:
        val = u.get(f)
        try:
            u[f] = float(val)
        except Exception as e:
            raise ValueError(f"Invalid numeric value for {f}: {val}") from e

    # Derived features (standard, sensible formulas)
    temp_diff = u["Process_Temperature(Kelvin)"] - u["Air_Temperature(Kelvin)"]
    # avoid division by zero for torque in ratio
    torque_safe = u["Torque(Nm)"] if abs(u["Torque(Nm)"]) > 1e-9 else 1e-9
    speed_torque_ratio = u["Rotational_Speed(rpm)"] / torque_safe
    energy_index = u["Rotational_Speed(rpm)"] * u["Torque(Nm)"]               # rotational speed * torque
    torque_wear_product = u["Torque(Nm)"] * u["Tool_Wear(min)"]
    speed_temp_interaction = u["Rotational_Speed(rpm)"] * temp_diff

    # Build full feature dict in the MODEL1_REQUIRED order (strings kept for categorical)
    full = {
        "Machine_Level": u.get("Machine_Level"),
        "Maintenance_Indicator": u.get("Maintenance_Indicator"),
        "Air_Temperature(Kelvin)": u["Air_Temperature(Kelvin)"],
        "Process_Temperature(Kelvin)": u["Process_Temperature(Kelvin)"],
        "Rotational_Speed(rpm)": u["Rotational_Speed(rpm)"],
        "Torque(Nm)": u["Torque(Nm)"],
        "Tool_Wear(min)": u["Tool_Wear(min)"],
        "Temp_Difference": temp_diff,
        "Speed_Torque_Ratio": speed_torque_ratio,
        "Wear_Rate": u["Wear_Rate"],
        "Energy_Index": energy_index,
        "Thermal_Stress_Index": u["Thermal_Stress_Index"],
        "Torque_Wear_Product": torque_wear_product,
        "Speed_Temp_Interaction": speed_temp_interaction,
        "Normalized_Wear_Rate": u["Normalized_Wear_Rate"],
    }

    return full


def model1_predict(manual_full):
    """
    manual_full: dict with ALL MODEL1_REQUIRED keys (strings/numbers).
    Returns the same result format as before.
    """
    # make a DataFrame with exact required column order
    df = pd.DataFrame([manual_full], columns=MODEL1_REQUIRED)

    # map categorical to numeric for model
    df["Machine_Level"] = df["Machine_Level"].map({"Low": 1, "Medium": 2, "High": 3})
    df["Maintenance_Indicator"] = df["Maintenance_Indicator"].map({"No": 0, "Yes": 1})

    # Ensure the features are ordered the same as the saved feature_order
    df_ordered = df[feature_order_model1].astype(float)

    # scale and predict
    scaled = scaler1.transform(df_ordered)
    pred = model1.predict(scaled)[0]
    proba = model1.predict_proba(scaled)[0]

    return {
        "Predicted_Class": "PASS" if int(pred) == 0 else "FAIL",
        "Confidence": round(float(max(proba)) * 100, 2),
        "Prob_PASS": round(float(proba[0]) * 100, 2),
        "Prob_FAIL": round(float(proba[1]) * 100, 2),
    }

# --- Model2 loading (keep one copy only) ---
model2 = joblib.load(BASE_ML_PATH / "model2.pkl")
model2_scaler = joblib.load(BASE_ML_PATH / "model2_scaler.pkl")
model2_label_encoder = joblib.load(BASE_ML_PATH / "model2_label_encoder.pkl")

with open(BASE_ML_PATH / "model2_feature_order.json") as f:
    feature_order_model2 = json.load(f)   # full order expected by the model

# --- Define which fields are provided by user (8) and which are derived (3) ---
MODEL2_USER_INPUTS = [
    "Melt_Temperature",
    "Mold_Temperature",
    "Casting_Pressure",
    "Cooling_Time",
    "Flow_Rate",
    "Ambient_Humidity",
    "Operator_Experience",
    "Tool_Condition"   # keep as string (e.g. "Worn"/"Good")
]

MODEL2_DERIVED = [
    "Temp_Diff",              # Melt_Temperature - Mold_Temperature
    "Cooling_Pressure_Ratio", # Cooling_Time / Casting_Pressure
    "Flow_Temp_Product"       # Flow_Rate * Melt_Temperature
]

# Default values for the 8 user inputs (used in forms)
DEFAULT_MODEL2_USER = {
    "Melt_Temperature": 742.95,
    "Mold_Temperature": 259.06,
    "Casting_Pressure": 81.30,
    "Cooling_Time": 24.85,
    "Flow_Rate": 31.07,
    "Ambient_Humidity": 78.90,
    "Operator_Experience": 3.42,
    "Tool_Condition": "Worn",
}

# Field name mapping when saving to Firestore (optional)
MODEL2_FIELD_MAP = {
    "Melt_Temperature": "melt_temperature",
    "Mold_Temperature": "mold_temperature",
    "Casting_Pressure": "casting_pressure",
    "Cooling_Time": "cooling_time",
    "Flow_Rate": "flow_rate",
    "Ambient_Humidity": "ambient_humidity",
    "Operator_Experience": "operator_experience",
    "Tool_Condition": "tool_condition",
    "Temp_Diff": "temp_diff",
    "Cooling_Pressure_Ratio": "cooling_pressure_ratio",
    "Flow_Temp_Product": "flow_temp_product",
}

# --- Model2 predictor that accepts only user inputs (8), computes derived (3), and predicts ---
def model2_predict_from_user_inputs(user_inputs):
    """
    user_inputs: dict with keys in MODEL2_USER_INPUTS (strings from form)
    returns: dict with predicted info and full_features dict (all 11)
    """

    # -----------------------------
    # 1) Parse numeric + categorical
    # -----------------------------
    parsed = {}

    numeric_fields = [f for f in MODEL2_USER_INPUTS if f != "Tool_Condition"]
    for f in numeric_fields:
        try:
            parsed[f] = float(user_inputs.get(f, 0))
        except Exception:
            parsed[f] = 0.0

    parsed["Tool_Condition"] = user_inputs.get("Tool_Condition", "Worn")

    # -----------------------------
    # 2) Derived features
    # -----------------------------
    parsed["Temp_Diff"] = parsed["Melt_Temperature"] - parsed["Mold_Temperature"]

    try:
        parsed["Cooling_Pressure_Ratio"] = parsed["Cooling_Time"] / parsed["Casting_Pressure"]
    except Exception:
        parsed["Cooling_Pressure_Ratio"] = 0.0

    parsed["Flow_Temp_Product"] = parsed["Flow_Rate"] * parsed["Melt_Temperature"]

    # -----------------------------
    # 3) Build full feature dict in correct model order
    # -----------------------------
    full_features = {}
    for feat in feature_order_model2:

        # Convert tool condition
        if feat == "Tool_Condition":
            full_features[feat] = 0 if parsed["Tool_Condition"].lower().startswith("w") else 1

        else:
            full_features[feat] = parsed.get(feat, 0.0)

    # -----------------------------
    # 4) SCALE ONLY THE TRAINED COLUMNS
    # -----------------------------
    scale_cols = [
        "Melt_Temperature", "Mold_Temperature", "Casting_Pressure",
        "Cooling_Time", "Flow_Rate", "Ambient_Humidity",
        "Temp_Diff", "Flow_Temp_Product"
    ]

    df = pd.DataFrame([full_features], columns=feature_order_model2)

    # → Extract numeric columns used for scaling
    df_scale = df[scale_cols].astype(float)

    # → Scale ONLY the required columns
    df_scaled = model2_scaler.transform(df_scale)

    # -----------------------------
    # 5) Reconstruct final feature set
    # -----------------------------
    # Start with original df
    final_df = df.copy()

    # Replace only the scaled columns
    final_df[scale_cols] = df_scaled

    # Now everything matches training exactly
    # -----------------------------
    # 6) Predict
    # -----------------------------
    pred_idx = model2.predict(final_df)[0]
    proba = model2.predict_proba(final_df)[0]

    labels = model2_label_encoder.inverse_transform(np.arange(len(proba)))
    pred_label = model2_label_encoder.inverse_transform([pred_idx])[0]

    prob_list = [
        {"label": lbl, "prob": round(float(p)*100, 2)}
        for lbl, p in zip(labels, proba)
    ]

    # -----------------------------
    # 7) Return result
    # -----------------------------
    return {
        "Predicted_Defect": pred_label,
        "Confidence": round(float(max(proba))*100, 2),
        "Probabilities": prob_list,
        "user_inputs": {k: user_inputs.get(k) for k in MODEL2_USER_INPUTS},
        "derived": {k: parsed.get(k) for k in MODEL2_DERIVED},
        "full_features": full_features
    }


# =====================================================================
#  TABLE HELPER (FOR DISPLAY AS SUMMARY TABLE IN HTML)
# =====================================================================
def make_columns_values_from_dict(data_dict, ordered_keys):
    """Return readable column names + values for HTML table."""
    cols, vals = [], []
    for k in ordered_keys:
        cols.append(k.replace("_", " "))
        vals.append(data_dict.get(k, ""))
    return cols, vals

# ============================================================
#  DATA INPUT (MODEL1 + MODEL2) — FIREBASE VERSION (FINAL)
# ============================================================
def data_input(request):
    uid = get_uid(request)

    # ==================================================
    # CREDIT INFO (for frontend & prediction logic)
    # ==================================================
    if uid:
        credit_info = get_user_credit_info(uid)
        total_credits_left = credit_info["free_credits"] + credit_info["paid_credits"]
    else:
        total_credits_left = 0

    ctx = {
        "model1_user_fields": MODEL1_USER_FIELDS,
        "model2_user_fields": MODEL2_USER_INPUTS,
        "default_model1": DEFAULT_MODEL1,
        "default_model2": DEFAULT_MODEL2_USER,

        "model1_result": request.session.get("model1_result"),
        "model2_result": request.session.get("model2_result"),

        "model1_recommendations": request.session.get("model1_recommendations"),
        "model2_recommendations": request.session.get("model2_recommendations"),

        "model1_user_inputs": request.session.get("model1_user_inputs"),
        "model2_user_inputs": request.session.get("model2_user_inputs"),

        "model1_columns_readable": request.session.get("model1_columns_readable"),
        "model1_values": request.session.get("model1_values"),

        "model2_columns_readable": request.session.get("model2_columns_readable"),
        "model2_values": request.session.get("model2_values"),

        "show_model1_form": True,
        "show_model2_form": False,
        "error": None,

        # For UI popup: if True → show Buy Plan popup
        "payment_required": (total_credits_left == 0),
        "credits_left": total_credits_left,
    }

    # =====================================================================
    # POST HANDLING
    # =====================================================================
    if request.method == "POST":
        action = request.POST.get("action")

        # ---------------------------------------------------------
        # BLOCK ALL ACTIONS IF NO CREDITS LEFT
        # ---------------------------------------------------------
        if action in ["predict_model1", "predict_model2"]:
            if total_credits_left <= 0:
                ctx["error"] = "You have no credits left to perform predictions. Please purchase a plan."
                ctx["payment_required"] = True
                return render(request, "quali_defect_app/data_input.html", ctx)

        # ---------------------------------------------------------
        # MODEL-1 PREDICT
        # ---------------------------------------------------------
        if action == "predict_model1":
            user_inputs = {
                "Machine_Level": request.POST.get("Machine_Level"),
                "Maintenance_Indicator": request.POST.get("Maintenance_Indicator"),
            }

            for f in MODEL1_USER_FIELDS:
                user_inputs[f] = request.POST.get(f)

            if any(v in ["", None] for v in user_inputs.values()):
                ctx["error"] = "Please fill all fields."
                ctx["default_model1"] = user_inputs
                return render(request, "quali_defect_app/data_input.html", ctx)

            try:
                full_inputs = _compute_model1_derived(user_inputs)
            except Exception as e:
                ctx["error"] = str(e)
                return render(request, "quali_defect_app/data_input.html", ctx)

            # Predict
            full_inputs_numeric = full_inputs.copy()
            full_inputs_numeric["Machine_Level"] = {"Low": 1, "Medium": 2, "High": 3}.get(full_inputs["Machine_Level"], 2)
            full_inputs_numeric["Maintenance_Indicator"] = {"No": 0, "Yes": 1}.get(full_inputs["Maintenance_Indicator"], 1)

            result1 = model1_predict(full_inputs)

            # OPTUNA RECOMMENDATION
            rec_out = recommend_machine_health(full_inputs_numeric)
            recs = rec_out.get("recommendations", {})
            before = rec_out.get("machine_fail_probability", 0)
            after = rec_out.get("optimized_fail_probability", before)

            reduction = round((before - after) * 100, 2) if before > 0 else 0

            ctx["model1_recommendations"] = recs
            ctx["model1_failure_reduction"] = reduction
            request.session["model1_recommendations"] = recs
            request.session["model1_failure_reduction"] = reduction

            cols, vals = make_columns_values_from_dict(full_inputs, MODEL1_REQUIRED)

            request.session["model1_user_inputs"] = full_inputs
            request.session["model1_result"] = result1
            request.session["model1_columns_readable"] = cols
            request.session["model1_values"] = vals

            ctx.update({
                "model1_result": result1,
                "model1_user_inputs": full_inputs,
                "model1_columns_readable": cols,
                "model1_values": vals,
                "show_model1_form": False,
            })

            # Save to Firestore
            if uid:
                # Save to Firestore (Model-1 WITH recommendations)
                firestore_data = {
                    "user_uid": uid,
                    "predicted_class": result1["Predicted_Class"],
                    "confidence": result1["Confidence"],

                    #  STORE RECOMMENDATIONS
                    "recommendations": recs,
                    "failure_reduction": reduction,

                    # Optional but useful for analytics
                    "machine_fail_probability": before,
                    "optimized_fail_probability": after,

                    "timestamp": firestore.SERVER_TIMESTAMP,
                }

                # Store all input + derived features
                for key, val in full_inputs.items():
                    firestore_data[MODEL1_FIELD_MAP[key]] = val

                db.collection("model1_records").add(firestore_data)


                #  Deduct credit
                if not deduct_prediction_credit(uid):
                    ctx["error"] = "No credits left."
                    ctx["payment_required"] = True
                    return render(request, "quali_defect_app/data_input.html", ctx)

            return render(request, "quali_defect_app/data_input.html", ctx)

        # ---------------------------------------------------------
        # MODEL-1 → MODEL-2
        # ---------------------------------------------------------
        if action == "go_model2":
            ctx["show_model1_form"] = False
            ctx["show_model2_form"] = True
            return render(request, "quali_defect_app/data_input.html", ctx)

        # ---------------------------------------------------------
        # MODEL-2 PREDICTION
        # ---------------------------------------------------------
        if action == "predict_model2":
            user_inputs = {f: request.POST.get(f) for f in MODEL2_USER_INPUTS}

            if any(v in ["", None] for v in user_inputs.values()):
                ctx["error"] = "Please fill all fields."
                ctx["show_model2_form"] = True
                return render(request, "quali_defect_app/data_input.html", ctx)

            result2 = model2_predict_from_user_inputs(user_inputs)

            pred_class = result2["Predicted_Defect"]
            model2_recs = get_model2_recommendations(pred_class)

            ctx["model2_recommendations"] = model2_recs
            request.session["model2_recommendations"] = model2_recs

            full_inputs = result2["full_features"]
            derived = result2["derived"]
            original_inputs = result2["user_inputs"]

            ordered = {k: full_inputs[k] for k in feature_order_model2}
            cols2, vals2 = make_columns_values_from_dict(ordered, feature_order_model2)

            request.session["model2_user_inputs"] = ordered
            request.session["model2_result"] = result2
            request.session["model2_columns_readable"] = cols2
            request.session["model2_values"] = vals2

            ctx.update({
                "model2_result": result2,
                "model2_user_inputs": ordered,
                "model2_columns_readable": cols2,
                "model2_values": vals2,
                "show_model1_form": False,
                "show_model2_form": False,
            })

            # Save to Firestore
            if uid:
                firestore_data = {
                    "user_uid": uid,
                    "predicted_defect": result2["Predicted_Defect"],
                    "confidence": result2["Confidence"],
                    "timestamp": firestore.SERVER_TIMESTAMP,
                }

                combined = {**original_inputs, **derived}

                for key, val in combined.items():
                    firestore_data[MODEL2_FIELD_MAP[key]] = val

                firestore_data["recommendations"] = model2_recs
                db.collection("model2_records").add(firestore_data)

                # ⭐ Deduct credit
                if not deduct_prediction_credit(uid):
                    ctx["error"] = "No credits left."
                    ctx["payment_required"] = True
                    return render(request, "quali_defect_app/data_input.html", ctx)

            return render(request, "quali_defect_app/data_input.html", ctx)

        # ---------------------------------------------------------
        # RESET ALL DATA
        # ---------------------------------------------------------
        if action == "reset_all":
            for key in [
                "model1_result", "model2_result",
                "model1_user_inputs", "model2_user_inputs",
                "model1_columns_readable", "model1_values",
                "model2_columns_readable", "model2_values"
            ]:
                request.session.pop(key, None)

            return redirect("data_input")

    return render(request, "quali_defect_app/data_input.html", ctx)



# ============================================================
#  CONTACT FORM (FIREBASE)
# ============================================================
def contact_view(request):

    uid = get_uid(request)

    if request.method == "POST":

        db.collection("contact_messages").add({
            "user_uid": uid,
            "name": request.POST.get("name"),
            "email": request.POST.get("email"),
            "mobile": request.POST.get("mobile"),
            "message": request.POST.get("message"),
        })

        return render(request, "quali_defect_app/contact.html", {"success": True})

    return render(request, "quali_defect_app/contact.html")



# ============================================================
#  HISTORY PAGE (Firebase)
# ============================================================
@firebase_login_required
def history_view(request):

    uid = get_uid(request)

    model1_history = [
    {**doc.to_dict(), "id": doc.id}
    for doc in db.collection("model1_records")
        .where("user_uid", "==", uid)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .stream()
]

    model2_history = [
    {**doc.to_dict(), "id": doc.id}
    for doc in db.collection("model2_records")
        .where("user_uid", "==", uid)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .stream()
]

    return render(request, "quali_defect_app/history.html", {
        "model1_history": model1_history,
        "model2_history": model2_history,
    })


# ============================================================
#  EXPORT EXCEL — FIRESTORE
# ============================================================
def export_model1_excel(request):

    uid = get_uid(request)

    docs = db.collection("model1_records").where("user_uid", "==", uid).stream()
    df = pd.DataFrame([doc.to_dict() for doc in docs])

    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = 'attachment; filename="model1_history.xlsx"'

    df.to_excel(response, index=False)
    return response


def export_model2_excel(request):

    uid = get_uid(request)

    docs = db.collection("model2_records").where("user_uid", "==", uid).stream()
    df = pd.DataFrame([doc.to_dict() for doc in docs])

    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = 'attachment; filename="model2_history.xlsx"'

    df.to_excel(response, index=False)
    return response


# ============================================================
#  TEST FIREBASE CONNECTION
# ============================================================
def test_firebase(request):
    if firebase_admin._apps:
        return HttpResponse("Firebase connected!")
    return HttpResponse("Firebase NOT connected.")

#######################################################################################
##########################################################
# Admin dashboard
##########################################################
#######################################################################################


def admin_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.session.get("admin_uid"):
            return redirect("admin_login")
        return view_func(request, *args, **kwargs)
    return wrapper


def admin_login_page(request):
    return render(request, "quali_defect_app/admin_panel/admin_login.html")


@csrf_exempt
def verify_admin_token(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except:
            return JsonResponse({"allowed": False})

        token = data.get("token")
        if not token:
            return JsonResponse({"allowed": False})

        try:
            decoded = auth.verify_id_token(token)
            uid = decoded.get("uid")

            user_doc = db.collection("users").document(uid).get()

            if user_doc.exists:
                role = user_doc.to_dict().get("role")

                if role == "admin":
                    request.session["admin_uid"] = uid
                    return JsonResponse({"allowed": True})

            return JsonResponse({"allowed": False})

        except Exception as e:
            print("Admin verify error:", e)
            return JsonResponse({"allowed": False})

    return JsonResponse({"allowed": False})



# ============================================================
# ADMIN ACTIVITY LOGGER
# ============================================================
# def log_admin_action(admin_uid, action_type, details):
#     try:
#         db.collection("admin_logs").add({
#             "admin_uid": admin_uid,
#             "action_type": action_type,
#             "details": details,
#             "timestamp": firestore.SERVER_TIMESTAMP
#         })
#     except Exception as e:
#         print("🔥 LOG ERROR:", e)



@admin_required
def admin_users(request):

    # -----------------------------------------
    # FETCH USERS
    # -----------------------------------------
    users_ref = db.collection("users").stream()
    users = []

    for doc in users_ref:
        data = doc.to_dict()
        users.append({
            "uid": doc.id,
            "email": data.get("email"),
            "role": data.get("role", "user"),
            "status": data.get("status", "active"),
            "created_at": data.get("created_at"),
        })

    # -----------------------------------------
    # GET ADMIN EMAIL FOR LOGGING
    # -----------------------------------------
    admin_uid = request.session.get("admin_uid")
    admin_email = None

    if admin_uid:
        admin_doc = db.collection("users").document(admin_uid).get()
        if admin_doc.exists:
            admin_email = admin_doc.to_dict().get("email")

    # -----------------------------------------
    # HANDLE POST ACTIONS
    # -----------------------------------------
    if request.method == "POST":
        action = request.POST.get("action")

        # ==========================================================
        # 1️⃣ CREATE USER
        # ==========================================================
        if action == "create_user":
            email = request.POST.get("email")
            password = request.POST.get("password")
            role = request.POST.get("role")
            status = request.POST.get("status")

            try:
                user_record = auth.create_user(email=email, password=password)

                db.collection("users").document(user_record.uid).set({
                    "email": email,
                    "role": role,
                    "status": status,
                    "created_at": firestore.SERVER_TIMESTAMP
                })

                # LOG ENTRY
                db.collection("logs").add({
                    "action": "CREATE_USER",
                    "admin_email": admin_email,
                    "user_email": email,
                    "description": f"Created new user → email={email}, role={role}, status={status}",
                    "timestamp": firestore.SERVER_TIMESTAMP
                })

            except Exception as e:
                print("🔥 CREATE USER ERROR:", e)

        # ==========================================================
        # 2️⃣ EDIT USER (role + status)
        # ==========================================================
        if action == "edit_user":
            uid = request.POST.get("uid")
            new_role = request.POST.get("role")
            new_status = request.POST.get("status")

            user_doc = db.collection("users").document(uid).get()
            user_email = user_doc.to_dict().get("email") if user_doc.exists else ""

            db.collection("users").document(uid).update({
                "role": new_role,
                "status": new_status
            })

            # LOG ENTRY
            db.collection("logs").add({
                "action": "EDIT_USER",
                "admin_email": admin_email,
                "user_email": user_email,
                "description": f"Updated user {user_email} → role={new_role}, status={new_status}",
                "timestamp": firestore.SERVER_TIMESTAMP
            })

        # ==========================================================
        # 3️⃣ DELETE USER
        # ==========================================================
        if action == "delete_user":
            uid = request.POST.get("uid")

            # Get email BEFORE deleting
            user_doc = db.collection("users").document(uid).get()
            user_email = user_doc.to_dict().get("email") if user_doc.exists else ""

            try:
                auth.delete_user(uid)
            except Exception as e:
                print("🔥 AUTH DELETE ERROR:", e)

            db.collection("users").document(uid).delete()

            # LOG ENTRY
            db.collection("logs").add({
                "action": "DELETE_USER",
                "admin_email": admin_email,
                "user_email": user_email,
                "description": f"Deleted user {user_email}",
                "timestamp": firestore.SERVER_TIMESTAMP
            })

        return redirect("admin_users")

    return render(request, "quali_defect_app/admin_panel/users.html", {
        "users": users,
        "active_page": "users",
    })



@admin_required
def admin_logs(request):
    logs = [
        {**doc.to_dict(), "id": doc.id}
        for doc in db.collection("logs")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .stream()
    ]

    return render(request, "quali_defect_app/admin_panel/logs.html", {
        "logs": logs,
        "active_page": "logs"
    })


@admin_required
def admin_settings(request):
    admin_uid = request.session.get("admin_uid")

    # Fetch admin email
    admin_doc = db.collection("users").document(admin_uid).get()
    admin_email = admin_doc.to_dict().get("email")

    context = {
        "admin_email": admin_email,
        "active_page": "settings",
        "success": None,
        "error": None,
    }

    # -----------------------------
    # CHANGE PASSWORD
    # -----------------------------
    if request.method == "POST":
        action = request.POST.get("action")

        if action == "change_password":
            old_password = request.POST.get("old_password")
            new_password = request.POST.get("new_password")
            confirm_password = request.POST.get("confirm_password")

            # Validate confirm password
            if new_password != confirm_password:
                context["error"] = "New password & confirm password do not match."
                return render(request, "quali_defect_app/admin_panel/settings.html", context)

            try:
                # 1️⃣ Admin must reauthenticate → sign-in using old password
                from firebase_admin import auth as firebase_auth

                # Fetch email for re-auth
                email = admin_email

                # Use Firebase REST API for reauthentication
                import requests

                FIREBASE_KEY = "AIzaSyBPjTIFC8czCOm_q6wxSCRH3RkbXQ543gg"
                url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_KEY}"

                payload = {
                    "email": email,
                    "password": old_password,
                    "returnSecureToken": True
                }

                r = requests.post(url, json=payload)
                data = r.json()

                if "error" in data:
                    context["error"] = "Old password is incorrect."
                    return render(request, "quali_defect_app/admin_panel/settings.html", context)

                # 2️⃣ Update password in Firebase Auth
                firebase_auth.update_user(
                    admin_uid,
                    password=new_password
                )

                context["success"] = "Password updated successfully!"
                return render(request, "quali_defect_app/admin_panel/settings.html", context)

            except Exception as e:
                print("Password update error:", e)
                context["error"] = "Something went wrong. Try again."

    return render(request, "quali_defect_app/admin_panel/settings.html", context)


@admin_required
def admin_dashboard(request):

    # -------------------------------------------
    # 1️⃣ USER ANALYTICS
    # -------------------------------------------
    users_ref = db.collection("users").stream()

    total_users = 0
    active_users = 0
    blocked_users = 0
    user_email_map = {}   # uid → email (for analytics)

    for doc in users_ref:
        d = doc.to_dict()
        if d is None:
            continue

        uid = doc.id
        email = d.get("email", "Unknown")

        user_email_map[uid] = email

        total_users += 1
        status = d.get("status", "active")

        if status == "active":
            active_users += 1
        else:
            blocked_users += 1

    # -------------------------------------------
    # 2️⃣ PREDICTION COUNTS
    # -------------------------------------------
    model1_docs = list(db.collection("model1_records").stream())
    model2_docs = list(db.collection("model2_records").stream())

    total_predictions = len(model1_docs) + len(model2_docs)

    # PASS/FAIL (Model1 binary classification)
    pass_count = sum(1 for d in model1_docs if d.to_dict().get("predicted_class") == "PASS")
    fail_count = sum(1 for d in model1_docs if d.to_dict().get("predicted_class") == "FAIL")

    # -------------------------------------------
    # 3️⃣ TREND (LAST 7 DAYS)
    # -------------------------------------------
    from datetime import datetime, timedelta

    today = datetime.utcnow().date()
    last_7_days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]

    daily_counts = {str(day): 0 for day in last_7_days}

    all_records = model1_docs + model2_docs

    for doc in all_records:
        d = doc.to_dict()
        if d is None:
            continue

        ts = d.get("timestamp")
        if not ts:
            continue

        try:
            day = ts.date()
            day_str = str(day)
            if day_str in daily_counts:
                daily_counts[day_str] += 1
        except:
            pass

    avg_predictions = []
    trend_labels = []
    trend_values = []

    for day in last_7_days:
        day_str = str(day)
        total = daily_counts.get(day_str, 0)
        avg = total / total_users if total_users else 0

        trend_labels.append(day_str)
        trend_values.append(round(avg, 2))

    # -------------------------------------------
    # 4️⃣ TOP USERS BY PREDICTIONS
    # -------------------------------------------
    user_prediction_count = {}

    for doc in all_records:
        d = doc.to_dict()
        if not d:
            continue

        uid = d.get("user_uid")
        if not uid:
            continue

        user_prediction_count[uid] = user_prediction_count.get(uid, 0) + 1

    # Sort top 5 users
    sorted_users = sorted(
        user_prediction_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    top_user_labels = [user_email_map.get(uid, "Unknown") for uid, _ in sorted_users]
    top_user_counts = [count for _, count in sorted_users]
    
    # ---------------------------------------------------------
    # 5️⃣ MODEL-2 CLASS DISTRIBUTION
    # ---------------------------------------------------------
    model2_class_counts = {}

    for doc in model2_docs:
        data = doc.to_dict()
        defect = data.get("predicted_defect", "Unknown")

        if defect not in model2_class_counts:
            model2_class_counts[defect] = 0
        model2_class_counts[defect] += 1

    # Convert keys + values for frontend charts
    model2_class_labels = list(model2_class_counts.keys())
    model2_class_values = list(model2_class_counts.values())

    # -------------------------------------------
    # 5️⃣ CONTEXT TO FRONTEND
    # -------------------------------------------
    context = {
        "active_page": "analytics",

        "total_users": total_users,
        "active_users": active_users,
        "blocked_users": blocked_users,
        "total_predictions": total_predictions,

        "pass_count": pass_count,
        "fail_count": fail_count,

        "trend_labels": trend_labels,   # list → JS safe
        "trend_values": trend_values,   # list → JS safe

        "top_user_labels": top_user_labels,
        "top_user_counts": top_user_counts,

        "model1_count": len(model1_docs),
        "model2_count": len(model2_docs),
        
        "model2_class_labels": model2_class_labels,
        "model2_class_values": model2_class_values,

    }

    return render(request, "quali_defect_app/admin_panel/dashboard.html", context)



import razorpay
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def create_order(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    data = json.loads(request.body)
    plan = data.get("plan")

    pricing = {
        "starter": 49,
        "standard": 199,
        "professional": 499,
    }

    if plan not in pricing:
        return JsonResponse({"error": "Invalid plan"}, status=400)

    amount = pricing[plan] * 100

    client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))
    order = client.order.create(dict(amount=amount, currency="INR", payment_capture=1))

    return JsonResponse({
        "key": settings.RAZORPAY_KEY_ID,
        "order_id": order["id"],
        "amount": amount,
    })


@csrf_exempt
def payment_success(request):
    data = json.loads(request.body)
    uid = get_uid(request)

    if not uid:
        return JsonResponse({"error": "Not logged in"}, status=403)

    payment = data.get("payment")
    plan = data.get("plan")

    # Razorpay verify
    client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))

    try:
        client.utility.verify_payment_signature({
            "razorpay_order_id": payment.get("razorpay_order_id"),
            "razorpay_payment_id": payment.get("razorpay_payment_id"),
            "razorpay_signature": payment.get("razorpay_signature"),
        })
    except razorpay.errors.SignatureVerificationError as e:
        return JsonResponse({"error": "Verification failed", "details": str(e)}, status=400)

    # Map plan → credits & price
    credit_map = {
        "starter": {"credits": 10, "price": 49},
        "standard": {"credits": 60, "price": 199},
        "professional": {"credits": 250, "price": 499},
    }

    if plan not in credit_map:
        return JsonResponse({"error": "Invalid plan name"}, status=400)

    credits_added = credit_map[plan]["credits"]
    amount_value = credit_map[plan]["price"]  # price in INR

    # Update user credits
    user_info = get_user_credit_info(uid)
    new_paid = user_info["paid_credits"] + credits_added

    db.collection("users").document(uid).update({
        "paid_credits": new_paid,
        "plan": plan,
    })

    # Store payment record in Firestore
    db.collection("payments").add({
        "user_uid": uid,
        "plan": plan,
        "credits_added": credits_added,
        "amount_in_inr": amount_value,
        "razorpay_order_id": payment.get("razorpay_order_id"),
        "razorpay_payment_id": payment.get("razorpay_payment_id"),
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    return JsonResponse({"status": "success"})


def user_credits_api(request):
    uid = get_uid(request)
    if not uid:
        return JsonResponse({"credits": 0})

    info = get_user_credit_info(uid)
    return JsonResponse({
        "credits": info["free_credits"] + info["paid_credits"]
    })