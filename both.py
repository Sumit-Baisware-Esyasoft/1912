# app_joint.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="DISCOM: Fault + ETR Joint Predictor", layout="wide")

# -------------------------
# Local file paths (update if needed)
# -------------------------
FAULT_MODEL_PATH = "best_model.pkl"
ETR_MODEL_PATH   = "ETR_MODEL.pkl"
ETR_ENCODERS     = "ETR_ENCODERS.pkl"
MASTER_CSV_PATH  = "TRAINING_DATA.csv"   # used only for populating ETR dropdowns

st.title("⚡ DISCOM - Joint Fault & ETR Predictor")
st.markdown("**Master file used for dropdowns:**")
st.code(MASTER_CSV_PATH)

# -------------------------
# Load models (cached for speed)
# -------------------------
@st.cache_resource
def load_fault_model(path=FAULT_MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        pipeline = bundle.get("pipeline", bundle)
        label_enc = bundle.get("label_encoder", None)
    else:
        pipeline = bundle
        label_enc = None
    return pipeline, label_enc

@st.cache_resource
def load_etr(path=ETR_MODEL_PATH, enc_path=ETR_ENCODERS):
    etr_m = joblib.load(path)
    etr_enc = joblib.load(enc_path)
    return etr_m, etr_enc

fault_pipeline, fault_label_encoder = load_fault_model()
etr_model, etr_encoders = load_etr()

# -------------------------
# Load master file for dropdowns
# -------------------------
@st.cache_data
def load_master(path=MASTER_CSV_PATH):
    try:
        dfm = pd.read_csv(path)
        return dfm
    except Exception:
        return pd.DataFrame()

df_master = load_master()

# Convenience lists for dropdowns (safe fallback if master missing)
def unique_sorted(col):
    if col in df_master.columns:
        return sorted(df_master[col].dropna().astype(str).unique().tolist())
    return []

region_options = ["--select--"] + unique_sorted("region_name")
circle_options = ["--select--"] + unique_sorted("circle_name")
division_options = ["--select--"] + unique_sorted("division_name")

# -------------------------
# Fault features list (used to prepare input)
# -------------------------
FEATURES = [
 'Feeder_ProcessStatus','DTR_ProcessStatus','Consumer_ProcessStatus','Consumer_Phase_Id',
 'f_vr','f_vy','f_vb','f_ir','f_iy','f_ib',
 'd_vr','d_vy','d_vb','d_ir','d_iy','d_ib',
 'C_tp_vr','C_tp_vy','C_tp_vb','C_tp_ir','C_tp_iy','C_tp_ib',
 'C_sp_i','C_sp_v'
]

# -------------------------
# Helper functions
# -------------------------
def prepare_fault_input(values_dict):
    """Return 1-row DataFrame with FEATURES columns in correct order."""
    data = {f: values_dict.get(f, np.nan) for f in FEATURES}
    return pd.DataFrame([data], columns=FEATURES)

def predict_fault(df_one):
    pred = fault_pipeline.predict(df_one)[0]
    try:
        if fault_label_encoder is not None:
            label = fault_label_encoder.inverse_transform([pred])[0]
        else:
            label = pred
    except Exception:
        label = pred
    probs = None
    if hasattr(fault_pipeline, "predict_proba"):
        try:
            p = fault_pipeline.predict_proba(df_one)
            classes = getattr(fault_pipeline, "classes_", None)
            if classes is None:
                classes = [str(i) for i in range(p.shape[1])]
            probs = pd.Series(p[0], index=[str(c) for c in classes])
        except Exception:
            probs = None
    return label, probs

def get_season(month_num):
    try:
        m = int(month_num)
    except:
        m = 1
    if m in [4,5,6]:
        return "Summer"
    if m in [7,8,9]:
        return "Rainy"
    return "Winter"

def predict_etr(month_input, complaint_time, region_name, circle_name, division_name):
    # month_input may be '2025-05' or '05' or integer
    try:
        if isinstance(month_input, str) and "-" in month_input:
            month_num = int(month_input.split("-")[1])
        else:
            month_num = int(month_input)
    except:
        month_num = 1
    season = get_season(month_num)
    etr_df = pd.DataFrame({
        "month":[month_num],
        "region_name":[region_name],
        "circle_name":[circle_name],
        "division_name":[division_name],
        "season":[season]
    })
    # encode with saved encoders (append unseen classes once)
    for col in ["region_name","circle_name","division_name","season"]:
        enc = etr_encoders.get(col)
        if enc is None:
            continue
        val = etr_df.at[0,col]
        if val not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, val)
        # try both transform APIs
        try:
            etr_df[col] = enc.transform(etr_df[[col]])
        except Exception:
            etr_df[col] = enc.transform(etr_df[col])
    pred_minutes = etr_model.predict(etr_df)[0]
    # to human format
    total_minutes = int(round(pred_minutes))
    hours = total_minutes // 60
    mins = total_minutes % 60
    human = f"{hours} hr {mins} min" if hours>0 else f"{mins} min"
    return pred_minutes, human

# -------------------------
# UI: Left panel for inputs, right for results
# -------------------------
left, right = st.columns([2,3])

with left:
    st.header("ETR dropdowns")
    region_sel = st.selectbox("region_name", region_options)
    circle_sel = st.selectbox("circle_name", circle_options)
    division_sel = st.selectbox("division_name", division_options)

    st.markdown("---")
    st.header("Fault inputs (enter values)")
    st.subheader("Ping statuses")
    feeder_ping = st.selectbox("Feeder_ProcessStatus", ["success","fail"], index=0)
    dtr_ping    = st.selectbox("DTR_ProcessStatus", ["success","fail"], index=0)
    cons_ping   = st.selectbox("Consumer_ProcessStatus", ["success","fail"], index=0)

    st.subheader("Consumer Phase")
    cons_phase = st.selectbox("Consumer_Phase_Id", [1,3], index=1, help="1 = single-phase, 3 = three-phase")

    st.subheader("Feeder voltages (V)")
    f_vr = st.number_input("f_vr", value=230.0, format="%.3f")
    f_vy = st.number_input("f_vy", value=230.0, format="%.3f")
    f_vb = st.number_input("f_vb", value=230.0, format="%.3f")
    st.subheader("Feeder currents (A)")
    f_ir = st.number_input("f_ir", value=0.0, format="%.3f")
    f_iy = st.number_input("f_iy", value=0.0, format="%.3f")
    f_ib = st.number_input("f_ib", value=0.0, format="%.3f")

    st.subheader("DTR voltages (V)")
    d_vr = st.number_input("d_vr", value=230.0, format="%.3f")
    d_vy = st.number_input("d_vy", value=230.0, format="%.3f")
    d_vb = st.number_input("d_vb", value=230.0, format="%.3f")
    st.subheader("DTR currents (A)")
    d_ir = st.number_input("d_ir", value=1.0, format="%.3f")
    d_iy = st.number_input("d_iy", value=1.0, format="%.3f")
    d_ib = st.number_input("d_ib", value=1.0, format="%.3f")

    st.markdown("### Consumer values")
    if cons_phase == 3:
        st.write("3-phase consumer voltages (V)")
        C_tp_vr = st.number_input("C_tp_vr", value=230.0, format="%.3f")
        C_tp_vy = st.number_input("C_tp_vy", value=230.0, format="%.3f")
        C_tp_vb = st.number_input("C_tp_vb", value=230.0, format="%.3f")
        st.write("3-phase consumer currents (A)")
        C_tp_ir = st.number_input("C_tp_ir", value=1.0, format="%.3f")
        C_tp_iy = st.number_input("C_tp_iy", value=1.0, format="%.3f")
        C_tp_ib = st.number_input("C_tp_ib", value=1.0, format="%.3f")
        C_sp_v = 0.0
        C_sp_i = 0.0
    else:
        st.write("Single-phase consumer")
        C_sp_v = st.number_input("C_sp_v", value=230.0, format="%.3f")
        C_sp_i = st.number_input("C_sp_i", value=1.0, format="%.3f")
        # set three-phase fields to NaN
        C_tp_vr = np.nan; C_tp_vy = np.nan; C_tp_vb = np.nan
        C_tp_ir = np.nan; C_tp_iy = np.nan; C_tp_ib = np.nan

    st.markdown("---")
    run_button = st.button("Predict Fault & ETR")

with right:
    st.header("Prediction outputs")
    result_placeholder = st.empty()

# -------------------------
# On click -> compute both predictions and show results together
# -------------------------
if run_button:
    # Validate ETR dropdowns present
    if region_sel in (None, "--select--") or circle_sel in (None, "--select--") or division_sel in (None, "--select--"):
        st.warning("Please select region_name, circle_name and division_name from the dropdowns (ETR inputs).")
    else:
        # Build fault input dictionary (convert ping to expected str)
        fault_vals = {
            'Feeder_ProcessStatus': feeder_ping,
            'DTR_ProcessStatus': dtr_ping,
            'Consumer_ProcessStatus': cons_ping,
            'Consumer_Phase_Id': cons_phase,
            'f_vr': f_vr, 'f_vy': f_vy, 'f_vb': f_vb,
            'f_ir': f_ir, 'f_iy': f_iy, 'f_ib': f_ib,
            'd_vr': d_vr, 'd_vy': d_vy, 'd_vb': d_vb,
            'd_ir': d_ir, 'd_iy': d_iy, 'd_ib': d_ib,
            'C_tp_vr': C_tp_vr, 'C_tp_vy': C_tp_vy, 'C_tp_vb': C_tp_vb,
            'C_tp_ir': C_tp_ir, 'C_tp_iy': C_tp_iy, 'C_tp_ib': C_tp_ib,
            'C_sp_v': C_sp_v, 'C_sp_i': C_sp_i
        }
        # prepare df and predict
        fault_df = prepare_fault_input(fault_vals)
        try:
            fault_label, fault_probs = predict_fault(fault_df)
        except Exception as e:
            st.error(f"Fault model error: {e}")
            raise

        # ETR predict
        try:
            etr_num, etr_human = predict_etr("2025-01" if isinstance(region_sel,str) else 1, 
                                            complaint_time="", region_name=region_sel, circle_name=circle_sel, division_name=division_sel)
            # Note: month used as dummy here (ETR needs month input) — you can add month selection if desired.
        except Exception as e:
            st.error(f"ETR model error: {e}")
            etr_num, etr_human = None, "Error"

        # Display both outputs
        with result_placeholder.container():
            st.subheader("Fault Prediction")
            st.markdown(f"**Label:** `{fault_label}`")
            if fault_probs is not None:
                st.markdown("**Probabilities:**")
                st.dataframe(fault_probs.rename_axis("Class").to_frame("Probability"))

            st.subheader("ETR Prediction")
            st.markdown(f"**ETR (minutes):** `{etr_num}`")
            st.markdown(f"**ETR (human):** `{etr_human}`")

            # Prepare downloadable CSV combining inputs + outputs
            out = fault_df.copy()
            out["Predicted_Fault"] = fault_label
            out["ETR_minutes"] = etr_num
            out["ETR_human"] = etr_human
            if fault_probs is not None:
                for c,p in fault_probs.items():
                    out[f"Prob_{c}"] = p
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download combined result CSV", csv_bytes, "joint_prediction.csv", "text/csv")

