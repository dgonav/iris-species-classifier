import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
)

# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[t] for t in iris.target]
    df["target"]  = iris.target

    X = df[iris.feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall":    recall_score(y_test, y_pred, average="weighted"),
        "F1 Score":  f1_score(y_test, y_pred, average="weighted"),
    }

    cm = confusion_matrix(y_test, y_pred)
    feature_imp = pd.Series(model.feature_importances_, index=iris.feature_names)

    return model, scaler, df, iris.feature_names, iris.target_names, metrics, cm, feature_imp

model, scaler, df, feature_names, target_names, metrics, cm, feature_imp = load_and_train()

COLORS = {"setosa": "#636EFA", "versicolor": "#EF553B", "virginica": "#00CC96"}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;'>🌸 Iris Species Classification</h1>
<p style='text-align:center; color:grey;'>
    Data Mining · Universidad de la Costa · Prof. José Escorcia-Gutierrez, Ph.D.
</p>
<p style='text-align:center; color:grey; font-size:0.85rem;'>
    <b>Group:</b> Diego Navarro Gómez &nbsp;|&nbsp; Juan Félix &nbsp;|&nbsp;
    Dinellys García &nbsp;|&nbsp; Kimberly Ochoa
</p>
<hr>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Model Metrics", "🔮 Predict Species", "🔍 Data Exploration"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model Performance — Random Forest (100 trees)")

    # KPI cards
    cols = st.columns(4)
    icons = {"Accuracy": "✅", "Precision": "🎯", "Recall": "📡", "F1 Score": "⚖️"}
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(f"{icons[name]} {name}", f"{val:.4f}")

    st.markdown("---")
    col_cm, col_fi = st.columns(2)

    # Confusion matrix
    with col_cm:
        st.markdown("**Confusion Matrix**")
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=target_names, y=target_names,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_layout(height=350, margin=dict(t=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importances
    with col_fi:
        st.markdown("**Feature Importances**")
        fi_df = feature_imp.sort_values(ascending=True).reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Teal",
        )
        fig_fi.update_layout(height=350, margin=dict(t=20), showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Workflow description
    with st.expander("📋 Methodology / Workflow"):
        st.markdown("""
**Pipeline steps:**

1. **Data Understanding** — Iris dataset (150 samples, 3 balanced classes, 4 numerical features). No missing values.
2. **Preprocessing** — StandardScaler applied to normalize features before training.
3. **Train/Test Split** — 75 % train / 25 % test, stratified to preserve class distribution.
4. **Modeling** — Random Forest Classifier (100 estimators, random_state=42).
   - *Why Random Forest?* Robust to outliers, handles correlated features well, provides
     feature importances, and consistently achieves >95 % accuracy on Iris with minimal tuning.
5. **Evaluation** — Weighted Accuracy, Precision, Recall, and F1-Score on the held-out test set.
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Enter Flower Measurements")

    c1, c2 = st.columns([1, 2])

    with c1:
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sw = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
        pw = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

        if st.button("🔮 Predict", use_container_width=True, type="primary"):
            st.session_state["predict"] = True

    with c2:
        if st.session_state.get("predict"):
            sample = np.array([[sl, sw, pl, pw]])
            sample_s = scaler.transform(sample)
            pred_idx  = model.predict(sample_s)[0]
            pred_name = target_names[pred_idx]
            proba     = model.predict_proba(sample_s)[0]

            color = COLORS[pred_name]
            st.markdown(f"""
<div style='background:{color}22; border-left:5px solid {color};
            padding:1rem; border-radius:8px; margin-bottom:1rem;'>
    <h3 style='color:{color}; margin:0;'>🌺 Predicted: <b><i>Iris {pred_name}</i></b></h3>
</div>
""", unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Species": [f"Iris {n}" for n in target_names],
                "Probability": proba,
            })
            fig_prob = px.bar(
                prob_df, x="Species", y="Probability",
                color="Species",
                color_discrete_map={f"Iris {k}": v for k, v in COLORS.items()},
                range_y=[0, 1],
                title="Class Probabilities",
            )
            fig_prob.update_layout(showlegend=False, height=250, margin=dict(t=35))
            st.plotly_chart(fig_prob, use_container_width=True)

        # 3D Scatter
        st.markdown("**3D Scatter — Dataset + New Sample**")
        fig3d = px.scatter_3d(
            df,
            x="petal length (cm)", y="petal width (cm)", z="sepal length (cm)",
            color="species",
            color_discrete_map=COLORS,
            opacity=0.7,
            title="Dataset distribution (petal L × petal W × sepal L)",
        )

        if st.session_state.get("predict"):
            fig3d.add_trace(go.Scatter3d(
                x=[pl], y=[pw], z=[sl],
                mode="markers",
                marker=dict(size=10, color="yellow", symbol="diamond",
                            line=dict(color="black", width=2)),
                name="⭐ Your Sample",
            ))

        fig3d.update_layout(height=420, margin=dict(t=40))
        st.plotly_chart(fig3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Exploratory Data Analysis")

    col_h, col_s = st.columns(2)

    with col_h:
        feat_sel = st.selectbox("Feature for histogram", feature_names)
        fig_hist = px.histogram(
            df, x=feat_sel, color="species",
            color_discrete_map=COLORS,
            barmode="overlay", opacity=0.7,
            nbins=20,
            title=f"Distribution of {feat_sel}",
        )
        fig_hist.update_layout(height=350, margin=dict(t=40))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_s:
        ax_x = st.selectbox("X axis", feature_names, index=2)
        ax_y = st.selectbox("Y axis", feature_names, index=3)
        fig_sc = px.scatter(
            df, x=ax_x, y=ax_y, color="species",
            color_discrete_map=COLORS,
            title=f"{ax_x} vs {ax_y}",
        )
        fig_sc.update_layout(height=350, margin=dict(t=40))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("**Scatter Matrix (Pair Plot)**")
    fig_pm = px.scatter_matrix(
        df,
        dimensions=list(feature_names),
        color="species",
        color_discrete_map=COLORS,
        opacity=0.6,
    )
    fig_pm.update_traces(diagonal_visible=False, marker=dict(size=3))
    fig_pm.update_layout(height=550, margin=dict(t=20))
    st.plotly_chart(fig_pm, use_container_width=True)

    with st.expander("📄 Raw Dataset"):
        st.dataframe(df.drop(columns="target"), use_container_width=True)
