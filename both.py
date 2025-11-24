# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title="DISCOM: Fault + ETR Predictor", layout="wide")

# -------------------------
# CONFIG: local file paths (use your uploaded files)
# -------------------------
FAULT_MODEL_PATH = "/mnt/data/best_model.pkl"
ETR_MODEL_PATH   = "/mnt/data/ETR_MODEL.pkl"
ETR_ENCODERS     = "/mnt/data/ETR_ENCODERS.pkl"
MASTER_CSV_PATH  = "/mnt/data/TRAINING_DATA.csv"

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_fault_pipeline(path=FAULT_MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    # if saved as {'pipeline':..., 'label_encoder':...}
    if isinstance(bundle, dict):
        pipeline = bundle.get("pipeline", bundle)
        label_enc = bundle.get("label_encoder", None)
    else:
        pipeline = bundle
        label_enc = None
    return pipeline, label_enc

@st.cache_resource
def load_etr_model(path=ETR_MODEL_PATH, enc_path=ETR_ENCODERS):
    m = joblib.load(path)
    enc = joblib.load(enc_path)
    return m, enc

fault_pipeline, fault_label_encoder = load_fault_pipeline()
etr_model, etr_encoders = load_etr_model()

# -------------------------
# Load master CSV for dropdowns
# -------------------------
@st.cache_data
def load_master(path=MASTER_CSV_PATH):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()

df_master = load_master()

# -------------------------
# Utility helpers
# -------------------------
FEATURES = [
 'Feeder_ProcessStatus','DTR_ProcessStatus','Consumer_ProcessStatus','Consumer_Phase_Id',
 'f_vr','f_vy','f_vb','f_ir','f_iy','f_ib',
 'd_vr','d_vy','d_vb','d_ir','d_iy','d_ib',
 'C_tp_vr','C_tp_vy','C_tp_vb','C_tp_ir','C_tp_iy','C_tp_ib',
 'C_sp_i','C_sp_v'
]

def prepare_fault_input_from_row(row):
    data = {f: row.get(f, np.nan) for f in FEATURES}
    return pd.DataFrame([data], columns=FEATURES)

def predict_fault(df_one_row):
    pred_raw = fault_pipeline.predict(df_one_row)[0]
    try:
        if fault_label_encoder is not None:
            pred_label = fault_label_encoder.inverse_transform([pred_raw])[0]
        else:
            pred_label = pred_raw
    except Exception:
        pred_label = pred_raw
    proba_df = None
    if hasattr(fault_pipeline, "predict_proba"):
        try:
            probs = fault_pipeline.predict_proba(df_one_row)
            classes = getattr(fault_pipeline, "classes_", None)
            if classes is None:
                classes = list(range(probs.shape[1]))
            proba_df = pd.DataFrame(probs, columns=[str(c) for c in classes])
        except Exception:
            proba_df = None
    return pred_label, proba_df

def get_season(month):
    try:
        m = int(str(month).split("-")[1]) if "-" in str(month) else int(month)
    except:
        try:
            m = int(month)
        except:
            m = 1
    if m in [4,5,6]:
        return "Summer"
    if m in [7,8,9]:
        return "Rainy"
    return "Winter"

def predict_ETR_from_inputs(month_val, complaint_time, region_name, circle_name, division_name):
    # month_val might be "2025-05" or "05"
    try:
        if month_val is None or str(month_val).startswith("--"):
            month_num = 1
        else:
            month_num = int(str(month_val).split("-")[1]) if "-" in str(month_val) else int(month_val)
    except:
        try:
            month_num = int(month_val)
        except:
            month_num = 1
    season = get_season(month_num)
    etr_input = pd.DataFrame({
        "month":[month_num],
        "region_name":[region_name],
        "circle_name":[circle_name],
        "division_name":[division_name],
        "season":[season]
    })
    # encode using saved encoders (append unseen categories if needed)
    for col in ["region_name","circle_name","division_name","season"]:
        enc = etr_encoders.get(col)
        if enc is None:
            continue
        val = etr_input.at[0, col]
        if val not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, val)
        # some encoders expect 1d arrays; try both
        try:
            etr_input[col] = enc.transform(etr_input[[col]])
        except Exception:
            etr_input[col] = enc.transform(etr_input[col])
    pred_num = etr_model.predict(etr_input)[0]
    total_minutes = int(round(pred_num))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    human = f"{hours} hr {minutes} min" if hours>0 else f"{minutes} min"
    return pred_num, human

# -------------------------
# UI Layout
# -------------------------
st.title("⚡ DISCOM Fault + ETR Joint Predictor")
st.write("Select a master row or upload a file. The app will predict Fault label and ETR (Estimated Time to Restore).")

col1, col2 = st.columns([2,3])

with col1:
    st.subheader("Input selection")
    use_upload = st.checkbox("Upload file (use first row)", value=False)
    uploaded_file = st.file_uploader("CSV / Excel file (first row used)", type=["csv","xlsx"]) if use_upload else None

    request_id = None
    if not use_upload:
        if "Request_Id" in df_master.columns:
            ids = ["--select--"] + df_master["Request_Id"].astype(str).tolist()
            request_id = st.selectbox("Select Request_Id (or choose index below)", ids)
        else:
            idxs = ["--select--"] + df_master.index.astype(str).tolist()
            request_id = st.selectbox("Select row index", idxs)
    # ETR overrides
    st.markdown("**ETR inputs (optional override)**")
    month_choices = ["--select--"] + (sorted(df_master["month"].dropna().astype(str).unique().tolist()) if "month" in df_master.columns else [])
    month_sel = st.selectbox("month", month_choices)
    complaint_choices = ["--select--"] + (sorted(df_master["complaint_time"].dropna().astype(str).unique().tolist()) if "complaint_time" in df_master.columns else [])
    complaint_sel = st.selectbox("complaint_time", complaint_choices)
    region_choices = ["--select--"] + (sorted(df_master["region_name"].dropna().astype(str).unique().tolist()) if "region_name" in df_master.columns else [])
    region_sel = st.selectbox("region_name", region_choices)
    circle_choices = ["--select--"] + (sorted(df_master["circle_name"].dropna().astype(str).unique().tolist()) if "circle_name" in df_master.columns else [])
    circle_sel = st.selectbox("circle_name", circle_choices)
    division_choices = ["--select--"] + (sorted(df_master["division_name"].dropna().astype(str).unique().tolist()) if "division_name" in df_master.columns else [])
    division_sel = st.selectbox("division_name", division_choices)

    run = st.button("Run Predictions")

with col2:
    st.subheader("Results")
    result_area = st.empty()
    download_bytes = None

# -------------------------
# Run logic when clicked
# -------------------------
if run:
    try:
        # get row data
        if use_upload and uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df_u = pd.read_csv(uploaded_file)
            else:
                df_u = pd.read_excel(uploaded_file)
            if df_u.shape[0] == 0:
                st.error("Uploaded file is empty.")
                st.stop()
            row = df_u.iloc[0].to_dict()
            source_note = "Uploaded file (first row)"
        else:
            if request_id is None or str(request_id).startswith("--"):
                st.error("Please select a row from master or upload a file.")
                st.stop()
            if "Request_Id" in df_master.columns and str(request_id) in df_master["Request_Id"].astype(str).values:
                row = df_master[df_master["Request_Id"].astype(str) == str(request_id)].iloc[0].to_dict()
            else:
                try:
                    idx = int(request_id)
                    row = df_master.loc[idx].to_dict()
                except Exception as e:
                    st.error(f"Could not find selected row: {e}")
                    st.stop()
            source_note = f"Master row {request_id}"

        st.write(f"**Source:** {source_note}")
        st.write("### Input preview")
        st.write(pd.DataFrame([row]))

        # Fault prediction
        fault_input = prepare_fault_input_from_row(row)
        pred_label, proba_df = predict_fault(fault_input)
        st.markdown(f"### 🔧 Predicted Fault: **{pred_label}**")
        if proba_df is not None:
            st.write("Probabilities (top classes):")
            st.dataframe(proba_df.T)

        # ETR prediction: prefer overrides
        month_val = month_sel if (month_sel and not str(month_sel).startswith("--")) else row.get("month","")
        complaint_time_val = complaint_sel if (complaint_sel and not str(complaint_sel).startswith("--")) else row.get("complaint_time","")
        region_val = region_sel if (region_sel and not str(region_sel).startswith("--")) else row.get("region_name","")
        circle_val = circle_sel if (circle_sel and not str(circle_sel).startswith("--")) else row.get("circle_name","")
        division_val = division_sel if (division_sel and not str(division_sel).startswith("--")) else row.get("division_name","")

        etr_num, etr_human = predict_ETR_from_inputs(month_val, complaint_time_val, region_val, circle_val, division_val)
        st.markdown(f"### ⏱ Predicted ETR: **{etr_human}** ({etr_num} minutes)")

        # Prepare output CSV bytes
        out_df = pd.DataFrame([row])
        out_df["Predicted_Fault"] = pred_label
        out_df["ETR_minutes"] = etr_num
        out_df["ETR_human"] = etr_human
        if proba_df is not None:
            for col in proba_df.columns:
                out_df[f"Prob_{col}"] = proba_df.iloc[0][col]
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")

        st.download_button("Download prediction CSV", csv_bytes, file_name="prediction_output.csv", mime="text/csv")

        # Optional: if uploaded file had true Final_Label, show evaluation metrics
        if use_upload and "Final_Label" in df_u.columns:
            true = df_u["Final_Label"].apply(lambda x: x if isinstance(x, str) else str(x)).values
            # if pipeline returned label strings else map
            pred_for_eval = [pred_label]  # single row
            st.write("Evaluation not applicable for single-row upload (requires many rows).")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# -------------------------
# Footer: master file link (local path shown)
# -------------------------
st.markdown("---")
st.markdown("Master CSV used for dropdowns (local path):")
st.code(f"{MASTER_CSV_PATH}")
