import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title="DISCOM ML Prediction App", layout="wide")

# ============================
#  LOAD TRAINED MODEL
# ============================
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["pipeline"], bundle["label_encoder"]

model, label_encoder = load_model()

st.title("⚡ DISCOM Fault Prediction — ML App")
st.write("Upload consumer/DTR/feeder data and get predictions using your trained LightGBM/ML pipeline.")

# ============================
#  DROP COLUMNS (YOUR UPDATED LIST)
# ============================
DROP_COLS = [
    'Request_Id','Feeder_MSN','DTR_MSN','Consumer_MSN',
    'Feeder_ResponseTime','DTR_ResponseTime','Consumer_ResponseTime',
    'Label_Reason','Unnamed: 34',
    'F_mf','D_mf','C_mf',
    'C_ping','D_ping','F_ping'
]

# ============================
#  FILE UPLOAD
# ============================
uploaded = st.file_uploader("📤 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded:
    # Read
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Detect evaluation mode (if actual labels included)
    evaluation_mode = "Final_Label" in df.columns

    # ============================
    #  PREPARE FEATURES
    # ============================
    st.subheader("🔧 Preparing Data")

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')

    if evaluation_mode:
        X = X.drop(columns=["Final_Label"], errors='ignore')

    st.write("Using these features for prediction:")
    st.write(list(X.columns))

    # ============================
    #  RUN PREDICTIONS
    # ============================
    try:
        st.subheader("🔮 Predictions")
        preds = model.predict(X)
        pred_labels = label_encoder.inverse_transform(preds)

        df_results = df.copy()
        df_results["Predicted_Label"] = pred_labels

        # Probability scores
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            proba_df = pd.DataFrame(
                proba,
                columns=[f"Prob_{c}" for c in label_encoder.classes_]
            )
            df_results = pd.concat([df_results, proba_df], axis=1)

        st.dataframe(df_results.head(25))

        # Allow download
        st.download_button(
            "📥 Download Results as CSV",
            df_results.to_csv(index=False).encode('utf-8'),
            file_name="predictions_output.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")


    # ============================
    #  EVALUATION MODE (If Final_Label exists)
    # ============================
    if evaluation_mode:
        st.subheader("📊 Model Evaluation (Using Provided True Labels)")

        y_true = label_encoder.transform(df["Final_Label"])
        y_pred = preds

        # ------ Confusion Matrix ------
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(label_encoder.classes_)))
        ax.set_xticklabels(label_encoder.classes_, rotation=45, ha="right")
        ax.set_yticks(range(len(label_encoder.classes_)))
        ax.set_yticklabels(label_encoder.classes_)
        for (i,j),v in np.ndenumerate(cm):
            ax.text(j,i,v,ha='center',va='center')
        st.pyplot(fig)

        # ------ Classification Report ------
        st.write("### Classification Report")
        report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # ------ ROC Curves ------
        if hasattr(model, "predict_proba"):
            st.write("### ROC Curves (One-vs-Rest)")

            y_bin = label_binarize(y_true, classes=np.arange(len(label_encoder.classes_)))
            probs = model.predict_proba(X)

            for i, cls in enumerate(label_encoder.classes_):
                fpr, tpr, _ = roc_curve(y_bin[:,i], probs[:,i])
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax.plot([0,1],[0,1],"--",color="gray")
                ax.set_title(f"ROC Curve — {cls}")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
                st.pyplot(fig)

    # ============================
    #  FEATURE IMPORTANCES
    # ============================
    st.subheader("🌟 Feature Importances")

    try:
        clf = model.named_steps['clf']
        pre = model.named_steps['pre']

        if hasattr(clf, "feature_importances_"):

            # numeric feature names
            num_names = pre.transformers_[0][2]

            # categorical names
            cat_cols = pre.transformers_[1][2]
            ohe = pre.named_transformers_['cat'].named_steps['onehot']
            cat_names = ohe.get_feature_names_out(cat_cols).tolist()

            feat_names = list(num_names) + cat_names

            importances = clf.feature_importances_
            idxs = np.argsort(importances)[-20:][::-1]

            fig, ax = plt.subplots(figsize=(8,6))
            ax.barh([feat_names[i] for i in idxs], importances[idxs])
            ax.set_title("Top 20 Important Features")
            st.pyplot(fig)

        else:
            st.info("This model does not support feature importances.")
    except Exception as e:
        st.error(f"Feature importance error: {e}")

