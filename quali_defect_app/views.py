from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.utils.timezone import make_naive
from django.contrib.auth import logout
from django.conf import settings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

from .models import ContactMessage
from django.contrib import messages
from .models import Model1Record, Model2Record

# ============================================================
#  MODEL DIRECTORY
# ============================================================
BASE_ML_PATH = Path(settings.BASE_DIR) / "quali_defect_app" / "ml_model"


# ============================================================
#  SIGNUP & LOGOUT
# ============================================================
def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, "quali_defect_app/signup.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("index")


# ============================================================
#  UTIL HELPERS
# ============================================================
def make_columns_values_from_dict(d, keys_order):
    cols, vals = [], []
    for k in keys_order:
        cols.append(k.replace("_", " "))
        vals.append(d.get(k, ""))
    return cols, vals


# ============================================================
#  MODEL-1 LOADING
# ============================================================
model1 = joblib.load(BASE_ML_PATH / "model1.pkl")
scaler1 = joblib.load(BASE_ML_PATH / "model1_scaler.pkl")

with open(BASE_ML_PATH / "model1_feature_order.json") as f:
    feature_order_model1 = json.load(f)

MODEL1_REQUIRED = [
    "Machine_Level", "Maintenance_Indicator",
    "Air_Temperature(Kelvin)", "Process_Temperature(Kelvin)",
    "Rotational_Speed(rpm)", "Torque(Nm)", "Tool_Wear(min)",
    "Temp_Difference", "Speed_Torque_Ratio", "Wear_Rate",
    "Energy_Index", "Thermal_Stress_Index", "Torque_Wear_Product",
    "Speed_Temp_Interaction", "Normalized_Wear_Rate"
]

MODEL1_FEATURES = MODEL1_REQUIRED[2:]  # numeric only

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

DEFAULT_MODEL1 = {
    "Machine_Level": "High",
    "Maintenance_Indicator": "Yes",
    "Air_Temperature(Kelvin)": 306,
    "Process_Temperature(Kelvin)": 317,
    "Rotational_Speed(rpm)": 1195,
    "Torque(Nm)": 61,
    "Tool_Wear(min)": 249,
    "Temp_Difference": 11,
    "Speed_Torque_Ratio": 19.5,
    "Wear_Rate": 208,
    "Energy_Index": 73063,
    "Thermal_Stress_Index": 642,
    "Torque_Wear_Product": 15211,
    "Speed_Temp_Interaction": 378.2,
    "Normalized_Wear_Rate": 2.83,
}


# ============================================================
#  MODEL-1 PREDICTOR
# ============================================================
def model1_predict(manual):
    df = pd.DataFrame([manual], columns=MODEL1_REQUIRED)

    df["Machine_Level"] = df["Machine_Level"].map({"Low": 1, "Medium": 2, "High": 3})
    df["Maintenance_Indicator"] = df["Maintenance_Indicator"].map({"No": 0, "Yes": 1})

    df_ordered = df[feature_order_model1].astype(float)
    scaled = scaler1.transform(df_ordered)

    pred = model1.predict(scaled)[0]
    proba = model1.predict_proba(scaled)[0]

    return {
        "Predicted_Class": "PASS" if pred == 0 else "FAIL",
        "Confidence": round(float(max(proba)) * 100, 2),
        "Prob_PASS": round(float(proba[0]) * 100, 2),
        "Prob_FAIL": round(float(proba[1]) * 100, 2),
    }


# ============================================================
#  MODEL-2 LOADING
# ============================================================
model2 = joblib.load(BASE_ML_PATH / "model2.pkl")
model2_scaler = joblib.load(BASE_ML_PATH / "model2_scaler.pkl")
model2_label_encoder = joblib.load(BASE_ML_PATH / "model2_label_encoder.pkl")

with open(BASE_ML_PATH / "model2_feature_order.json") as f:
    feature_order_model2 = json.load(f)

MODEL2_FEATURES = feature_order_model2[:]

MODEL2_SCALED_COLS = [
    "Melt_Temperature",
    "Mold_Temperature",
    "Casting_Pressure",
    "Cooling_Time",
    "Flow_Rate",
    "Ambient_Humidity",
    "Temp_Diff",
    "Flow_Temp_Product"
]

MODEL2_FIELD_MAP = {
    "Tool_Condition": "tool_condition",
    "Melt_Temperature": "melt_temperature",
    "Mold_Temperature": "mold_temperature",
    "Casting_Pressure": "casting_pressure",
    "Cooling_Time": "cooling_time",
    "Flow_Rate": "flow_rate",
    "Ambient_Humidity": "ambient_humidity",
    "Operator_Experience": "operator_experience",
    "Temp_Diff": "temp_diff",
    "Cooling_Pressure_Ratio": "cooling_pressure_ratio",
    "Flow_Temp_Product": "flow_temp_product",
}

DEFAULT_MODEL2 = {
    "Melt_Temperature": 742.95,
    "Mold_Temperature": 259.06,
    "Casting_Pressure": 81.30,
    "Cooling_Time": 24.85,
    "Flow_Rate": 31.07,
    "Ambient_Humidity": 78.90,
    "Operator_Experience": 3.42,
    "Tool_Condition": "Worn",
    "Temp_Diff": 483.89,
    "Cooling_Pressure_Ratio": 0.31,
    "Flow_Temp_Product": 23081.22,
}


# ============================================================
#  MODEL-2 PREDICTOR
# ============================================================
def model2_predict(manual):

    df = pd.DataFrame([manual], columns=MODEL2_FEATURES)

    df["Tool_Condition"] = df["Tool_Condition"].map({"Worn": 0, "Good": 1})

    for col in MODEL2_SCALED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Cooling_Pressure_Ratio"] = pd.to_numeric(df["Cooling_Pressure_Ratio"], errors="coerce")

    df_scaled = df.copy()
    df_scaled[MODEL2_SCALED_COLS] = model2_scaler.transform(df_scaled[MODEL2_SCALED_COLS])

    df_ordered = df_scaled[feature_order_model2]

    pred_idx = model2.predict(df_ordered)[0]
    proba = model2.predict_proba(df_ordered)[0]

    labels = model2_label_encoder.inverse_transform(np.arange(len(proba)))

    prob_list = [{"label": lbl, "prob": round(float(p) * 100, 2)} for lbl, p in zip(labels, proba)]
    confidence = round(float(max(proba)) * 100, 2)
    pred_label = model2_label_encoder.inverse_transform([pred_idx])[0]

    return {
        "Predicted_Defect": pred_label,
        "Confidence": confidence,
        "Probabilities": prob_list
    }


# ============================================================
#  STATIC PAGES
# ============================================================
def index(request):
    return render(request, "quali_defect_app/index.html")


def about(request):
    return render(request, "quali_defect_app/about.html")


def contact(request):
    return render(request, "quali_defect_app/contact.html")


# ============================================================
#  MAIN — DATA INPUT VIEW
# ============================================================
def data_input(request):

    ctx = {
        "features": MODEL1_FEATURES,
        "model2_features": MODEL2_FEATURES,
        "default_model1": DEFAULT_MODEL1,
        "default_model2": DEFAULT_MODEL2,
        "model1_result": request.session.get("model1_result"),
        "model2_result": request.session.get("model2_result"),
        "model1_user_inputs": request.session.get("model1_user_inputs"),
        "model2_user_inputs": request.session.get("model2_user_inputs"),
        "show_model1_form": True,
        "show_model2_form": False,
        "error": None,
    }

    # ============================================================
    #  POST — MODEL-1 PREDICT
    # ============================================================
    if request.method == "POST":
        action = request.POST.get("action")

        # ----------------------------------------------------------
        # MODEL-1 PREDICT
        # ----------------------------------------------------------
        if action == "predict_model1":

            manual1 = {f: request.POST.get(f) for f in MODEL1_REQUIRED}

            if any(v in ["", None] for v in manual1.values()):
                ctx["error"] = "Please fill all fields."
                ctx["default_model1"] = manual1
                return render(request, "quali_defect_app/data_input.html", ctx)

            result1 = model1_predict(manual1)

            request.session["model1_user_inputs"] = manual1
            request.session["model1_result"] = result1

            ctx["model1_result"] = result1
            ctx["model1_user_inputs"] = manual1

            cols, vals = make_columns_values_from_dict(manual1, MODEL1_REQUIRED)
            ctx["model1_columns_readable"] = cols
            ctx["model1_values"] = vals

            mapped = {model_key: float(manual1[form_key])
                      if form_key not in ["Machine_Level", "Maintenance_Indicator"]
                      else manual1[form_key]
                      for form_key, model_key in MODEL1_FIELD_MAP.items()}

            Model1Record.objects.create(
                user=request.user if request.user.is_authenticated else None,
                predicted_class=result1["Predicted_Class"],
                **mapped
            )

            ctx["show_model1_form"] = False

            return render(request, "quali_defect_app/data_input.html", ctx)

        # ----------------------------------------------------------
        #  GO TO MODEL-2
        # ----------------------------------------------------------
        if action == "go_model2":
            ctx["model1_user_inputs"] = request.session.get("model1_user_inputs")
            ctx["model1_result"] = request.session.get("model1_result")
            ctx["show_model1_form"] = False
            ctx["show_model2_form"] = True
            return render(request, "quali_defect_app/data_input.html", ctx)

        # ----------------------------------------------------------
        # MODEL-2 PREDICT
        # ----------------------------------------------------------
        if action == "predict_model2":

            manual2 = {f: request.POST.get(f) for f in MODEL2_FEATURES}

            if any(v in ["", None] for v in manual2.values()):
                ctx["error"] = "Please fill all fields."
                ctx["show_model2_form"] = True
                return render(request, "quali_defect_app/data_input.html", ctx)

            result2 = model2_predict(manual2)

            request.session["model2_result"] = result2
            request.session["model2_user_inputs"] = manual2

            ctx["model2_result"] = result2
            ctx["model2_user_inputs"] = manual2

            cols2, vals2 = make_columns_values_from_dict(manual2, MODEL2_FEATURES)
            ctx["model2_columns_readable"] = cols2
            ctx["model2_values"] = vals2

            mapped2 = {}
            for form_key, model_key in MODEL2_FIELD_MAP.items():
                value = manual2.get(form_key)
                if form_key == "Tool_Condition":
                    mapped2[model_key] = value
                else:
                    try:
                        mapped2[model_key] = float(value)
                    except:
                        mapped2[model_key] = None

            Model2Record.objects.create(
                user=request.user if request.user.is_authenticated else None,
                predicted_defect=result2["Predicted_Defect"],
                **mapped2
            )

            ctx["show_model1_form"] = False
            ctx["show_model2_form"] = False

            ctx["model1_user_inputs"] = request.session.get("model1_user_inputs")
            ctx["model1_result"] = request.session.get("model1_result")
            
            if ctx["model1_user_inputs"]:
                cols, vals = make_columns_values_from_dict(ctx["model1_user_inputs"], MODEL1_REQUIRED)
                ctx["model1_columns_readable"] = cols
                ctx["model1_values"] = vals

            return render(request, "quali_defect_app/data_input.html", ctx)

        # ----------------------------------------------------------
        # RESET ALL
        # ----------------------------------------------------------
        if action == "reset_all":
            request.session.pop("model1_user_inputs", None)
            request.session.pop("model1_result", None)
            request.session.pop("model2_user_inputs", None)
            request.session.pop("model2_result", None)
            return redirect("data_input")

    # ============================================================
    #  GET — RESTORE TABLES + RESULTS AFTER REFRESH
    # ============================================================

    # MODEL-1 TABLE RESTORE
    if ctx["model1_user_inputs"]:
        cols, vals = make_columns_values_from_dict(ctx["model1_user_inputs"], MODEL1_REQUIRED)
        ctx["model1_columns_readable"] = cols
        ctx["model1_values"] = vals
        ctx["show_model1_form"] = False

    # MODEL-2 TABLE RESTORE
    if ctx["model2_user_inputs"]:
        cols2, vals2 = make_columns_values_from_dict(ctx["model2_user_inputs"], MODEL2_FEATURES)
        ctx["model2_columns_readable"] = cols2
        ctx["model2_values"] = vals2
        ctx["show_model1_form"] = False
        ctx["show_model2_form"] = False

    return render(request, "quali_defect_app/data_input.html", ctx)


def contact_view(request):
    success = False

    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        mobile = request.POST.get("mobile")
        message_text = request.POST.get("message")

        if request.user.is_authenticated:
            ContactMessage.objects.create(
                user=request.user,
                name=name,
                email=email,
                mobile=mobile,
                message=message_text
            )
            success = True

        return render(request, 'quali_defect_app/contact.html', {"success": success})

    return render(request, 'quali_defect_app/contact.html')


@login_required
def history_view(request):
    model1_history = Model1Record.objects.filter(
        user=request.user
    ).order_by('-timestamp')

    model2_history = Model2Record.objects.filter(
        user=request.user
    ).order_by('-timestamp')

    return render(request, "quali_defect_app/history.html", {
        "model1_history": model1_history,
        "model2_history": model2_history,
    })
    
    

def export_model1_excel(request):
    records = Model1Record.objects.filter(user=request.user).order_by('-timestamp')

    # Convert QuerySet → List of dicts
    data = []
    for r in records:
        row = vars(r).copy()
        row.pop('_state', None)

        # Convert timezone-aware timestamps to naive
        if r.timestamp:
            row['timestamp'] = make_naive(r.timestamp)

        data.append(row)

    df = pd.DataFrame(data)

    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response['Content-Disposition'] = 'attachment; filename=\"model1_history.xlsx\"'

    df.to_excel(response, index=False)
    return response



def export_model2_excel(request):
    records = Model2Record.objects.filter(user=request.user).order_by('-timestamp')

    data = []
    for r in records:
        row = vars(r).copy()
        row.pop('_state', None)

        if r.timestamp:
            row['timestamp'] = make_naive(r.timestamp)

        data.append(row)

    df = pd.DataFrame(data)

    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response['Content-Disposition'] = 'attachment; filename=\"model2_history.xlsx\"'

    df.to_excel(response, index=False)
    return response

