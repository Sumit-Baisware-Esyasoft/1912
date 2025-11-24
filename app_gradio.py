# app_gradio.py
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIG: file paths (already in your environment)
# -------------------------
FAULT_MODEL_PATH = "/mnt/data/best_model.pkl"
ETR_MODEL_PATH   = "/mnt/data/ETR_MODEL.pkl"
ETR_ENCODERS     = "/mnt/data/ETR_ENCODERS.pkl"
MASTER_CSV_PATH  = "/mnt/data/TRAINING_DATA.csv"   # master file for dropdowns

# -------------------------
# LOAD MODELS (cache)
# -------------------------
def load_fault_pipeline(path=FAULT_MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    # handle both formats: either pipeline or {'pipeline':..., 'label_encoder':...}
    pipeline = bundle.get("pipeline", bundle) if isinstance(bundle, dict) else bundle
    label_enc = bundle.get("label_encoder", None) if isinstance(bundle, dict) else None
    # some saved bundles use different keys; try to find label encoder
    if label_enc is None:
        label_enc = bundle.get("label_encoder", None) if isinstance(bundle, dict) else None
    return pipeline, label_enc

def load_etr_model(path=ETR_MODEL_PATH, enc_path=ETR_ENCODERS):
    m = joblib.load(path)
    enc = joblib.load(enc_path)
    return m, enc

fault_pipeline, fault_label_encoder = load_fault_pipeline()
etr_model, etr_encoders = load_etr_model()

# -------------------------
# Load master CSV for dropdown values
# -------------------------
if Path(MASTER_CSV_PATH).exists():
    df_master = pd.read_csv(MASTER_CSV_PATH)
else:
    df_master = pd.DataFrame()  # empty fallback

# Useful lists for dropdowns (safe guards)
def unique_values(col):
    if col in df_master.columns:
        vals = df_master[col].dropna().astype(str).unique().tolist()
        return sorted(vals)
    return []

months_list = unique_values("month")
complaint_times = unique_values("complaint_time")
regions = unique_values("region_name")
circles = unique_values("circle_name")
divisions = unique_values("division_name")
request_ids = unique_values("Request_Id") if "Request_Id" in df_master.columns else [str(i) for i in df_master.index.astype(str)]

# -------------------------
# FEATURES used by fault model (same as before)
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
def prepare_fault_input_from_row(row):
    data = {f: row.get(f, np.nan) for f in FEATURES}
    return pd.DataFrame([data], columns=FEATURES)

def predict_fault(df_one_row):
    pred_raw = fault_pipeline.predict(df_one_row)[0]
    # try to inverse transform
    try:
        if fault_label_encoder is not None:
            pred_label = fault_label_encoder.inverse_transform([pred_raw])[0]
        else:
            pred_label = pred_raw
    except Exception:
        # sometimes pipeline outputs strings already
        pred_label = pred_raw
    probs = None
    try:
        if hasattr(fault_pipeline, "predict_proba"):
            probs = fault_pipeline.predict_proba(df_one_row)
            # column names: try pipeline.classes_ or fallback to numeric indices
            classes = getattr(fault_pipeline, "classes_", None)
            if classes is None:
                classes = list(range(probs.shape[1]))
            prob_df = pd.DataFrame(probs, columns=[str(c) for c in classes])
        else:
            prob_df = None
    except Exception:
        prob_df = None
    return pred_label, prob_df

def get_season(month):
    try:
        month = int(str(month).split("-")[1]) if "-" in str(month) else int(month)
    except:
        try:
            month = int(month)
        except:
            month = 1
    if month in [4,5,6]:
        return "Summer"
    if month in [7,8,9]:
        return "Rainy"
    return "Winter"

def predict_ETR_from_inputs(month_val, complaint_time, region_name, circle_name, division_name):
    month_num = None
    try:
        if month_val is None or month_val == "" or str(month_val).startswith("--"):
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
    # apply encoders
    for col in ["region_name","circle_name","division_name","season"]:
        enc = etr_encoders.get(col)
        if enc is None:
            continue
        val = etr_input.at[0, col]
        if val not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, val)
        # encoder.transform often expects 1D array-like
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
# Main processing function (for Gradio)
# -------------------------
def predict_from_master(request_id_or_index, month_sel, complaint_time_sel, region_sel, circle_sel, division_sel, uploaded_file):
    """
    If uploaded_file provided: use first row of uploaded csv/xlsx to predict.
    Else use the selected Request_Id (or index) from master.
    Returns: tuple of strings and CSV bytes for download
    """
    # choose source row
    if uploaded_file is not None:
        # read uploaded file into a dataframe; get first row
        try:
            if isinstance(uploaded_file, str):
                # path-like (in local tests)
                if uploaded_file.lower().endswith(".csv"):
                    df_u = pd.read_csv(uploaded_file)
                else:
                    df_u = pd.read_excel(uploaded_file)
            else:
                # gradio file object (temp file-like)
                df_u = pd.read_csv(uploaded_file.name) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file.name)
        except Exception as e:
            return f"Error reading uploaded file: {e}", "", None
        if df_u.shape[0] == 0:
            return "Uploaded file is empty", "", None
        row = df_u.iloc[0].to_dict()
        source_note = "Using uploaded file (first row)"
    else:
        # use master
        if request_id_or_index is None or str(request_id_or_index).startswith("--"):
            return "No input selected and no file uploaded. Please select Request_Id or upload file.", "", None
        # try by Request_Id or index
        if "Request_Id" in df_master.columns and str(request_id_or_index) in df_master["Request_Id"].astype(str).values:
            row = df_master[df_master["Request_Id"].astype(str) == str(request_id_or_index)].iloc[0].to_dict()
        else:
            # maybe it's index
            try:
                idx = int(request_id_or_index)
                row = df_master.loc[idx].to_dict()
            except Exception as e:
                return f"Could not find Request_Id or index: {e}", "", None
        source_note = f"Using master row for {request_id_or_index}"

    # Prepare fault input
    fault_input = prepare_fault_input_from_row(row)

    # Predict fault
    try:
        fault_label, prob_df = predict_fault(fault_input)
    except Exception as e:
        return f"Fault prediction error: {e}", "", None

    # Determine ETR inputs: prefer user selected dropdowns if provided, else fallback to row values
    month_val = month_sel if (month_sel and not str(month_sel).startswith("--")) else row.get("month","")
    complaint_time_val = complaint_time_sel if (complaint_time_sel and not str(complaint_time_sel).startswith("--")) else row.get("complaint_time","")
    region_val = region_sel if (region_sel and not str(region_sel).startswith("--")) else row.get("region_name","")
    circle_val = circle_sel if (circle_sel and not str(circle_sel).startswith("--")) else row.get("circle_name","")
    division_val = division_sel if (division_sel and not str(division_sel).startswith("--")) else row.get("division_name","")

    # Predict ETR
    try:
        etr_num, etr_human = predict_ETR_from_inputs(month_val, complaint_time_val, region_val, circle_val, division_val)
    except Exception as e:
        etr_num, etr_human = None, f"ETR prediction error: {e}"

    # build result summary text
    res_text = f"{source_note}\n\nPredicted Fault: {fault_label}\n"
    if prob_df is not None:
        # transform prob_df columns to strings
        proba_str = "\n".join([f"{col}: {prob_df.iloc[0][col]:.3f}" for col in prob_df.columns])
        res_text += f"Probabilities:\n{proba_str}\n"
    res_text += f"\nPredicted ETR (minutes): {etr_num}\nPredicted ETR (human): {etr_human}"

    # Build CSV for download: combine original row + predictions + probabilities
    out_df = pd.DataFrame([row])
    out_df["Predicted_Fault"] = fault_label
    out_df["ETR_minutes"] = etr_num
    out_df["ETR_human"] = etr_human
    if prob_df is not None:
        for col in prob_df.columns:
            out_df[f"Prob_{col}"] = prob_df.iloc[0][col]
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    return res_text, out_df.head(1).to_json(orient="records"), csv_bytes

# -------------------------
# Build Gradio UI
# -------------------------
with gr.Blocks(title="DISCOM Joint Prediction (Fault + ETR)") as demo:
    gr.Markdown("# ⚡ DISCOM: Fault & ETR Joint Predictor (Gradio)")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Select input row from master or upload a file")
            req_dropdown = gr.Dropdown(choices=["--select--"] + request_ids, label="Request_Id / Index", value="--select--")
            uploaded_file = gr.File(label="Upload CSV/XLSX (use first row only)", file_types=[".csv", ".xlsx"])
            run_btn = gr.Button("Run Predictions")
            download_btn = gr.DownloadButton("Download result CSV", visible=False, label="Download CSV")
        with gr.Column(scale=3):
            gr.Markdown("### ETR Inputs (optional - overrides selected row values)")
            month_dd = gr.Dropdown(choices=["--select--"]+months_list, label="month", value="--select--")
            complaint_time_dd = gr.Dropdown(choices=["--select--"]+complaint_times, label="complaint_time", value="--select--")
            region_dd = gr.Dropdown(choices=["--select--"]+regions, label="region_name", value="--select--")
            circle_dd = gr.Dropdown(choices=["--select--"]+circles, label="circle_name", value="--select--")
            division_dd = gr.Dropdown(choices=["--select--"]+divisions, label="division_name", value="--select--")
    with gr.Row():
        output_text = gr.Textbox(label="Prediction Summary", lines=12)
        output_json = gr.JSON(label="Output Row (JSON)")
    # event
    def on_run(req_id, month_sel, ct_sel, reg_sel, circ_sel, div_sel, file_obj):
        txt, json_out, csv_bytes = predict_from_master(req_id, month_sel, ct_sel, reg_sel, circ_sel, div_sel, file_obj)
        # set download visibility and payload
        visible = csv_bytes is not None
        return txt, gr.JSON.update(value=json_out), gr.File.update(value=csv_bytes, visible=visible)
    # hook up: when run_btn clicked, call on_run and update outputs. Gradio DownloadButton uses update() with file content.
    run_btn.click(fn=on_run,
                  inputs=[req_dropdown, month_dd, complaint_time_dd, region_dd, circle_dd, division_dd, uploaded_file],
                  outputs=[output_text, output_json, download_btn])

    gr.Markdown("### Notes")
    gr.Markdown("- If you upload a file, the first row of that file is used as input. - If you choose dropdown values they override master row values for ETR inputs.")
    gr.Markdown(f"- Master file used (if present): `{MASTER_CSV_PATH}`")
    gr.Markdown("")

# -------------------------
# Launch
# -------------------------
if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
