# app_premium.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path

# -------------------------
# Page config & globals
# -------------------------
st.set_page_config(page_title="DISCOM AI — Fault & ETR (Premium)", layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={"About": "Premium DISCOM AI dashboard — built for Esyasoft"})

# Local model files (do not change unless necessary)
FAULT_MODEL_PATH = "best_model.pkl"
ETR_MODEL_PATH   = "ETR_MODEL.pkl"
ETR_ENCODERS     = "ETR_ENCODERS.pkl"
MASTER_CSV_PATH  = "TRAINING_DATA.csv"

# -------------------------
# Inject custom CSS for Electric Blue theme & animations
# -------------------------
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(180deg, #0f1724 0%, #071228 40%, #021223 100%);
        color: #e6f7ff;
    }
    /* Card style */
    .card {
        background: linear-gradient(90deg, rgba(6,24,66,0.9), rgba(3,9,23,0.6));
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(2,10,30,0.6);
        color: #e6f7ff;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .card:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(2,10,40,0.7); }

    /* Header gradient */
    .big-header {
        font-size:28px;
        font-weight:700;
        background: -webkit-linear-gradient(45deg,#00d4ff, #0066ff, #00b4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* animated bar */
    .progress-bar {
        height: 12px;
        border-radius: 8px;
        background: linear-gradient(90deg,#00d4ff, #0066ff);
        animation: grow 1s ease;
    }
    @keyframes grow {
      from {width: 0;}
      to {width: 100%;}
    }

    /* small subtle text */
    .muted { color: #b6dff8; font-size: 0.9rem; }

    /* metric tile */
    .metric-tile {
        border-radius: 12px;
        padding: 12px;
        background: linear-gradient(90deg, rgba(0, 116, 255, 0.12), rgba(0, 212, 255, 0.04));
        box-shadow: inset 0 -2px 12px rgba(0,0,0,0.3);
    }

    /* footer */
    .footer { color: #9fbfe8; font-size:0.85rem; padding-top: 10px; opacity:0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Cached model loading
# -------------------------
@st.cache_resource
def load_fault_model(path=FAULT_MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        pipe = bundle.get("pipeline", bundle)
        label_enc = bundle.get("label_encoder", None)
    else:
        pipe = bundle
        label_enc = None
    return pipe, label_enc

@st.cache_resource
def load_etr_model(path=ETR_MODEL_PATH, enc_path=ETR_ENCODERS):
    m = joblib.load(path)
    enc = joblib.load(enc_path)
    return m, enc

fault_pipeline, fault_label_encoder = load_fault_model()
etr_model, etr_encoders = load_etr_model()

# -------------------------
# Utility / Predict functions
# -------------------------
FEATURES = [
 'Feeder_ProcessStatus','DTR_ProcessStatus','Consumer_ProcessStatus','Consumer_Phase_Id',
 'f_vr','f_vy','f_vb','f_ir','f_iy','f_ib',
 'd_vr','d_vy','d_vb','d_ir','d_iy','d_ib',
 'C_tp_vr','C_tp_vy','C_tp_vb','C_tp_ir','C_tp_iy','C_tp_ib',
 'C_sp_i','C_sp_v'
]

def prepare_fault_input(vals):
    row = {f: vals.get(f, np.nan) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)

def predict_fault_one(df_row):
    raw = fault_pipeline.predict(df_row)[0]
    try:
        label = fault_label_encoder.inverse_transform([raw])[0] if fault_label_encoder is not None else raw
    except Exception:
        label = raw
    probs = None
    if hasattr(fault_pipeline, "predict_proba"):
        p = fault_pipeline.predict_proba(df_row)[0]
        classes = getattr(fault_pipeline, "classes_", None)
        if classes is None:
            classes = [str(i) for i in range(len(p))]
        probs = {str(c): float(prob) for c, prob in zip(classes, p)}
    return label, probs

def get_season(month_num):
    try:
        m = int(month_num)
    except:
        m = 1
    if m in [4,5,6]: return "Summer"
    if m in [7,8,9]: return "Rainy"
    return "Winter"

def predict_etr(region, circle, division, month_input=1):
    season = get_season(month_input)
    df = pd.DataFrame({
        "month":[month_input],
        "region_name":[region],
        "circle_name":[circle],
        "division_name":[division],
        "season":[season]
    })
    for col in ["region_name","circle_name","division_name","season"]:
        enc = etr_encoders.get(col)
        if enc is None:
            continue
        val = df.at[0, col]
        if val not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, val)
        try:
            df[col] = enc.transform(df[[col]])
        except Exception:
            df[col] = enc.transform(df[col])
    pred = etr_model.predict(df)[0]
    minutes = int(round(pred))
    hrs = minutes // 60
    mins = minutes % 60
    human = f"{hrs} hr {mins} min" if hrs>0 else f"{mins} min"
    return minutes, human

# -------------------------
# Load master dropdown values (if available)
# -------------------------
def load_master_csv(path=MASTER_CSV_PATH):
    try:
        dfm = pd.read_csv(path)
        return dfm
    except Exception:
        return pd.DataFrame()

df_master = load_master_csv()

def unique_sorted(col):
    if col in df_master.columns:
        return sorted(df_master[col].dropna().astype(str).unique().tolist())
    return []

# populate ETR dropdown lists from master file or defaults
region_opts = ["--select--"] + unique_sorted("region_name") if len(unique_sorted("region_name"))>0 else ["Rewa","Jabalpur","Sagar","Shahdol"]
circle_opts = ["--select--"] + unique_sorted("circle_name") if len(unique_sorted("circle_name"))>0 else ["Rewa","Satna","Sidhi","Singrauli"]
division_opts = ["--select--"] + unique_sorted("division_name") if len(unique_sorted("division_name"))>0 else ["Rewa (CITY)","Rewa (RURAL)","Satna","Chitrakoot"]

# -------------------------
# Layout - Header
# -------------------------
st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;">'
            '<div><span class="big-header">⚡ DISCOM AI — Fault & ETR (Premium)</span><div class="muted">Electric Blue theme • Esyasoft</div></div>'
            '<div style="text-align:right"><img src="https://raw.githubusercontent.com/analyticsindiamagazine/images/master/ai-images/ai-logo.png" width="90"/></div>'
            '</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Sidebar: quick controls
# -------------------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("**Upload saved input row (optional)**")
    uploaded = st.file_uploader("CSV / Excel (first row used)", type=["csv","xlsx"])
    st.markdown("---")
    st.markdown("**Quick presets**")
    if st.button("Preset: Typical DTHT case"):
        st.session_state['_preset'] = 'dt_ht'
    if st.button("Preset: Typical DTLT case"):
        st.session_state['_preset'] = 'dt_lt'
    st.markdown("---")
    st.markdown("Master CSV (local):")
    st.code(MASTER_CSV_PATH)
    st.markdown('<div class="footer">Powered by DISCOM AI — Esyasoft</div>', unsafe_allow_html=True)

# -------------------------
# Main UI: inputs (left) and outputs (right)
# -------------------------
col_in, col_out = st.columns([1.6, 2.4])

with col_in:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔹 ETR Inputs (Choose region / circle / division)", unsafe_allow_html=True)
    region = st.selectbox("Region Name", region_opts, index=0)
    circle = st.selectbox("Circle Name", circle_opts, index=0)
    division = st.selectbox("Division Name", division_opts, index=0)

    st.markdown("---")
    st.markdown("### 🔧 Fault Inputs", unsafe_allow_html=True)
    # pings
    feeder_ping = st.radio("Feeder Ping", ("success","fail"), index=0, horizontal=True)
    dtr_ping    = st.radio("DTR Ping", ("success","fail"), index=0, horizontal=True)
    cons_ping   = st.radio("Consumer Ping", ("success","fail"), index=0, horizontal=True)
    st.markdown("Consumer Phase")
    phase = st.selectbox("Consumer_Phase_Id", [3,1], index=0)

    st.markdown("#### Feeder voltages (V)")
    f_vr = st.number_input("f_vr", value=230.0, format="%.3f")
    f_vy = st.number_input("f_vy", value=230.0, format="%.3f")
    f_vb = st.number_input("f_vb", value=230.0, format="%.3f")
    st.markdown("#### Feeder currents (A)")
    f_ir = st.number_input("f_ir", value=1.0, format="%.3f")
    f_iy = st.number_input("f_iy", value=1.0, format="%.3f")
    f_ib = st.number_input("f_ib", value=1.0, format="%.3f")

    st.markdown("---")
    st.markdown("#### DTR voltages (V)")
    d_vr = st.number_input("d_vr", value=230.0, format="%.3f")
    d_vy = st.number_input("d_vy", value=230.0, format="%.3f")
    d_vb = st.number_input("d_vb", value=230.0, format="%.3f")
    st.markdown("#### DTR currents (A)")
    d_ir = st.number_input("d_ir", value=1.0, format="%.3f")
    d_iy = st.number_input("d_iy", value=1.0, format="%.3f")
    d_ib = st.number_input("d_ib", value=1.0, format="%.3f")

    st.markdown("---")
    st.markdown("### Consumer readings")
    if phase == 3:
        C_tp_vr = st.number_input("C_tp_vr", value=230.0, format="%.3f")
        C_tp_vy = st.number_input("C_tp_vy", value=230.0, format="%.3f")
        C_tp_vb = st.number_input("C_tp_vb", value=230.0, format="%.3f")
        C_tp_ir = st.number_input("C_tp_ir", value=1.0, format="%.3f")
        C_tp_iy = st.number_input("C_tp_iy", value=1.0, format="%.3f")
        C_tp_ib = st.number_input("C_tp_ib", value=1.0, format="%.3f")
        C_sp_v = 0.0; C_sp_i = 0.0
    else:
        C_sp_v = st.number_input("C_sp_v", value=230.0, format="%.3f")
        C_sp_i = st.number_input("C_sp_i", value=1.0, format="%.3f")
        C_tp_vr=C_tp_vy=C_tp_vb=C_tp_ir=C_tp_iy=C_tp_ib = np.nan

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
    predict_btn = st.button("✨ Predict Fault & ETR", key="run_predict")

with col_out:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Results", unsafe_allow_html=True)

    # Placeholder tiles
    tiles_col1, tiles_col2 = st.columns(2)
    with tiles_col1:
        st.markdown('<div class="metric-tile"><h4 style="margin:0;">🔎 Predicted Fault</h4><div id="fault_label_area"></div></div>', unsafe_allow_html=True)
    with tiles_col2:
        st.markdown('<div class="metric-tile"><h4 style="margin:0;">⏱ Predicted ETR</h4><div id="etr_area"></div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Main result box
    result_box = st.container()

    if predict_btn:
        # Build fault input and run model
        input_vals = {
            "Feeder_ProcessStatus": feeder_ping,
            "DTR_ProcessStatus": dtr_ping,
            "Consumer_ProcessStatus": cons_ping,
            "Consumer_Phase_Id": phase,
            "f_vr": f_vr, "f_vy": f_vy, "f_vb": f_vb,
            "f_ir": f_ir, "f_iy": f_iy, "f_ib": f_ib,
            "d_vr": d_vr, "d_vy": d_vy, "d_vb": d_vb,
            "d_ir": d_ir, "d_iy": d_iy, "d_ib": d_ib,
            "C_tp_vr": C_tp_vr, "C_tp_vy": C_tp_vy, "C_tp_vb": C_tp_vb,
            "C_tp_ir": C_tp_ir, "C_tp_iy": C_tp_iy, "C_tp_ib": C_tp_ib,
            "C_sp_i": C_sp_i, "C_sp_v": C_sp_v
        }
        fault_df = prepare_fault_input(input_vals)
        try:
            fault_label, probs = predict_fault_one(fault_df)
        except Exception as e:
            st.error(f"Fault model error: {e}")
            fault_label, probs = "ERROR", {}

        # ETR
        if region in (None, "--select--") or circle in (None, "--select--") or division in (None, "--select--"):
            st.warning("Pick region / circle / division from dropdowns to compute ETR.")
            etr_minutes, etr_human = None, "N/A"
        else:
            try:
                etr_minutes, etr_human = predict_etr(region, circle, division)
            except Exception as e:
                st.error(f"ETR model error: {e}")
                etr_minutes, etr_human = None, "Error"

        # Display top-level tiles (update metric tiles)
        st.markdown(f"<div class='metric-tile' style='margin-bottom:8px;'><h3 style='margin:0;'>🔥 {fault_label}</h3><div class='muted'>Predicted Fault Class</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-tile'><h3 style='margin:0;'>⏱ {etr_human}</h3><div class='muted'>Estimated Time to Restore</div></div>", unsafe_allow_html=True)

        # Probability table and bar chart
        if probs:
            prob_df = pd.DataFrame({"Fault Class": list(probs.keys()), "Probability": [v*100 for v in probs.values()]})
            prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

            result_box.markdown("#### Class probabilities")
            # show table
            result_box.dataframe(prob_df.style.format({"Probability": "{:.2f}"}), height=240)

            # bar chart
            fig, ax = plt.subplots(figsize=(8,3))
            bars = ax.bar(prob_df["Fault Class"], prob_df["Probability"], color="#00b4ff", edgecolor="#0066ff")
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Fault Class Probability Distribution")
            plt.xticks(rotation=45)
            for bar in bars:
                w = bar.get_width()
            result_box.pyplot(fig)

            # top-3 card
            top3 = prob_df.head(3)
            result_box.markdown("#### Top-3 probable faults")
            cols = result_box.columns(3)
            for i, r in top3.iterrows():
                with cols[i]:
                    st.markdown(f"<div style='padding:12px;border-radius:8px;background:linear-gradient(90deg, rgba(0,212,255,0.06), rgba(0,116,255,0.04));'>"
                                f"<h4 style='margin:0'>{r['Fault Class']}</h4>"
                                f"<div class='muted'>{r['Probability']:.2f}%</div></div>", unsafe_allow_html=True)

        else:
            result_box.info("No probabilities available for this model; predicted class: " + str(fault_label))

        # Prepare downloadable CSV
        out = fault_df.copy()
        out["Predicted_Fault"] = fault_label
        out["ETR_minutes"] = etr_minutes
        out["ETR_human"] = etr_human
        if probs:
            for k,v in probs.items():
                out[f"Prob_{k}"] = v
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        result_box.download_button("📥 Download result CSV", csv_bytes, file_name="joint_prediction_premium.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: Use presets in the left sidebar to populate common scenarios. For production deployment consider running behind reverse proxy with HTTPS.")
