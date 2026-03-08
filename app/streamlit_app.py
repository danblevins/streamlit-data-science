"""
Prompt Complexity Classifier — Full Data Science Workflow App
Streamlit web application showcasing EDA, modeling, explainability, and live prediction.
"""
import os, sys, json, ast, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_features, parse_techniques, TECHNIQUES, PROMPT_TYPES
from app.ui_components import render_section_header, render_metric_cards, card

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Complexity Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .pc-card {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 0.75rem;
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55);
        margin-bottom: 1.25rem;
        transition: transform 0.15s ease-out, box-shadow 0.15s ease-out,
                    border-color 0.15s ease-out, background 0.15s ease-out;
    }
    .pc-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 22px 55px rgba(0, 0, 0, 0.65);
        border-color: rgba(76, 175, 80, 0.7);
        background: rgba(255, 255, 255, 0.03);
    }
    .pc-card-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }
    .pc-metric-card {
        text-align: left;
        padding: 0.9rem 1rem;
        box-shadow: none;
        border-radius: 0.65rem;
        margin-bottom: 0.75rem;
    }
    .pc-metric-label {
        font-size: 0.78rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .pc-metric-value {
        font-size: 1.15rem;
        font-weight: 700;
    }
    .pc-subtitle {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-top: -0.4rem;
        margin-bottom: 0.9rem;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2.5rem;
        padding-left: 3.5rem;
        padding-right: 3.5rem;
    }
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Constants ─────────────────────────────────────────────────────────────────
PALETTE = {'low': '#4CAF50', 'medium': '#2196F3', 'high': '#F44336'}
CLASS_ORDER = ['low', 'medium', 'high']

# ─── Load resources ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/prompt_examples_dataset.csv')
    return df

@st.cache_resource
def load_models():
    models = {}
    model_names = {
        'Random Forest': 'models/random_forest.pkl',
        'Gradient Boosting': 'models/gradient_boosting.pkl',
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Decision Tree': 'models/decision_tree.pkl',
        'MLP': 'models/mlp.pkl',
        'SVM (RBF)': 'models/svm_rbf.pkl',
        'K-Nearest Neighbors': 'models/k-nearest_neighbors.pkl',
    }
    for name, path in model_names.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    le = joblib.load('models/label_encoder.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return models, le, feature_names

@st.cache_data
def load_metrics():
    with open('models/best_model_metrics.json') as f:
        return json.load(f)

@st.cache_data
def load_comparison():
    return pd.read_csv('models/model_comparison.csv', index_col=0)

# ─── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
models, le, feature_names = load_models()
metrics = load_metrics()
comparison_df = load_comparison()

# ─── Top-level navigation tabs ────────────────────────────────────────────────
overview_tab, eda_tab, models_tab, explain_tab, predict_tab = st.tabs([
    "🏠 Overview",
    "📊 EDA",
    "🤖 Models",
    "🔍 Explainability",
    "⚡ Predict",
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with overview_tab:
    st.title("🤖 Prompt Complexity Classifier")
    render_section_header(
        "End-to-end prompt complexity dashboard",
        "Explore the dataset, compare models, understand feature importance, and run live predictions in one place.",
    )

    overview_metrics = {
        "📦 Dataset Size": "1,450 prompts",
        "🔢 Features": "71 engineered",
        "🏆 Best Model": "Random Forest",
        "✅ Test Accuracy": f"{metrics['accuracy']:.1%}",
        "📈 Macro F1": f"{metrics['macro_f1']:.3f}",
    }
    render_metric_cards(overview_metrics)

    col_main, col_side = st.columns([2, 1])
    with col_main:
        with card("Data science workflow"):
            st.markdown(
                """
                This application walks through a complete prompt-complexity project:

                1. **Data collection** — 1,450 prompts across 16 task types and 3 complexity levels.  
                2. **EDA** — class balance, text lengths, prompting techniques, and correlations.  
                3. **Feature engineering** — 71 structural and NLP features.  
                4. **Model training** — 5 classifiers with 5-fold stratified CV.  
                5. **Explainability** — feature importance and per-feature distributions.  
                6. **Deployment** — this interactive Streamlit dashboard.  
                """
            )

        with card("📋 Sample data"):
            st.dataframe(
                df[["task_description", "complexity", "prompt_type", "prompting_techniques"]].head(10),
                use_container_width=True,
            )

    with col_side:
        with card("⚡ Jump into live prediction"):
            st.markdown(
                """
                Try the model on your own prompts:

                - Provide **task description** and prompts  
                - Choose **prompt type** and **techniques**  
                - See predicted complexity with confidence  
                """
            )
            st.markdown("Use the **⚡ Predict** tab at the top to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
with eda_tab:
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Class Distribution", "Text Lengths", "Prompting Techniques",
        "Complexity × Type", "Feature Correlations"
    ])

    with tab1:
        st.subheader("Class & Type Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            counts = df["complexity"].value_counts()[CLASS_ORDER].reset_index()
            counts.columns = ["complexity", "count"]
            counts["percent"] = counts["count"] / len(df) * 100
            counts["label"] = counts.apply(
                lambda r: f"{int(r['count'])} ({r['percent']:.1f}%)", axis=1
            )
            fig = px.bar(
                counts,
                x="complexity",
                y="count",
                color="complexity",
                text="label",
                color_discrete_map=PALETTE,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                title="Complexity Distribution",
                xaxis_title="Complexity",
                yaxis_title="Count",
                showlegend=False,
            )
            st.caption("This bar chart highlights how the target variable (low, medium, high complexity) is distributed across the dataset, revealing moderate class imbalance.")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            type_counts = df["prompt_type"].value_counts().reset_index()
            type_counts.columns = ["prompt_type", "count"]
            fig = px.bar(
                type_counts,
                x="count",
                y="prompt_type",
                orientation="h",
                title="Prompt Type Distribution",
            )
            fig.update_layout(xaxis_title="Count", yaxis_title="")
            st.caption("This chart highlights which prompt types (e.g., Informational, Q&A) appear most frequently in the dataset.")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Key findings:**
        - Dataset is **moderately imbalanced** — medium (50.4%) dominates, low (23.3%) is underrepresented
        - 16 unique prompt types, with Informational, Q&A, and Code Generation being most common
        - Class imbalance handled via `class_weight='balanced'` in models
        """)

    with tab2:
        st.subheader("Text Length by Complexity")
        cols = [
            ("task_description", "Task Description"),
            ("bad_prompt", "Bad Prompt"),
            ("good_prompt", "Good Prompt"),
        ]
        for col, label in cols:
            tmp = df[[col, "complexity"]].copy()
            tmp["length"] = tmp[col].astype(str).str.len()
            fig = px.histogram(
                tmp,
                x="length",
                color="complexity",
                nbins=40,
                barmode="overlay",
                opacity=0.65,
                color_discrete_map=PALETTE,
                title=f"{label} Length by Complexity",
            )
            fig.update_layout(
                xaxis_title="Character Count",
                yaxis_title="Frequency",
            )
            st.caption(f"This histogram highlights how character length for {label} varies by complexity level, with high-complexity prompts typically longer.")
            st.plotly_chart(fig, use_container_width=True)

        # Stats table
        length_stats = df.groupby('complexity')[['task_description', 'good_prompt', 'bad_prompt']].apply(
            lambda g: g.applymap(lambda x: len(str(x))).mean()
        ).round(1)
        st.dataframe(length_stats, use_container_width=True)
        st.info("💡 **Insight:** High-complexity prompts tend to have longer good prompts, more structure, and larger length ratios vs. bad prompts.")

    with tab3:
        st.subheader("Prompting Techniques Analysis")
        all_techs = []
        for _, row in df.iterrows():
            for t in parse_techniques(row['prompting_techniques']):
                all_techs.append({'technique': t, 'complexity': row['complexity']})
        tech_df = pd.DataFrame(all_techs)
        tech_counts = tech_df['technique'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            freq_df = tech_counts.reset_index()
            freq_df.columns = ["technique", "count"]
            fig = px.bar(
                freq_df,
                x="count",
                y="technique",
                orientation="h",
                title="Overall Prompting Technique Frequency",
            )
            fig.update_layout(xaxis_title="Count", yaxis_title="")
            st.caption("This chart highlights which prompting techniques (e.g., CHAIN_OF_THOUGHT, ROLE_PROMPTING) are used most often across the dataset.")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            tech_complex = tech_df.groupby(['technique', 'complexity']).size().unstack(fill_value=0)
            top_techs = tech_counts.head(10).index
            tc = tech_complex.loc[tech_complex.index.isin(top_techs)]
            tc_long = tc.reset_index().melt(
                id_vars="technique", var_name="complexity", value_name="count"
            )
            fig = px.bar(
                tc_long,
                x="count",
                y="technique",
                color="complexity",
                orientation="h",
                barmode="stack",
                color_discrete_map=PALETTE,
                title="Techniques by Complexity Level (Top 10)",
            )
            fig.update_layout(xaxis_title="Count", yaxis_title="")
            st.caption("This stacked bar chart highlights how the top 10 techniques split across low, medium, and high complexity—revealing which techniques associate with harder prompts.")
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Insight:** CHAIN_OF_THOUGHT and ROLE_PROMPTING are most common. High-complexity prompts more frequently use TREE_OF_THOUGHTS and multi-technique combinations.")

    with tab4:
        st.subheader("Complexity vs. Prompt Type Heatmap")
        pivot = df.groupby(['prompt_type', 'complexity']).size().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        
        ordered = pivot_pct[["low", "medium", "high"]]
        fig = px.imshow(
            ordered,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(
            title="Complexity Distribution by Prompt Type (%)",
            xaxis_title="Complexity",
            yaxis_title="Prompt Type",
        )
        st.caption("This heatmap highlights the percentage of each prompt type that falls into low, medium, or high complexity—showing which task types tend toward simpler or harder prompts.")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** TRANSLATION and CONVERSATIONAL prompts skew low-complexity. ANALYSIS_CRITIQUE and CREATIVE_WRITING prompts lean high-complexity.")

    with tab5:
        st.subheader("Feature Correlation Matrix")
        feat_df = extract_features(df)
        key_feats = [c for c in feat_df.columns if not c.startswith('tech_') and not c.startswith('type_')]
        corr = feat_df[key_feats].corr()
        
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis_title="Feature",
            yaxis_title="Feature",
        )
        st.caption("This heatmap highlights pairwise correlations between engineered features, revealing redundancy (high correlation) and independent signals (low correlation).")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** Length-based features are highly correlated with each other. Instruction density is a more independent signal.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING & COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with models_tab:
    st.title("🤖 Model Training & Comparison")
    
    st.markdown("""
    Five classifiers were trained using **5-fold stratified cross-validation** on 71 engineered features.
    The best model was then evaluated on a held-out **20% test set**.
    """)

    st.subheader("📊 Cross-Validation Results (Macro F1)")
    
    cv_df = comparison_df.reset_index().rename(columns={"index": "Model"})
    fig = px.bar(
        cv_df,
        x="Model",
        y="CV F1 Mean",
        error_y="CV F1 Std",
        text="CV F1 Mean",
        title="Model Comparison: 5-Fold Cross-Validation (Macro F1)",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.add_hline(y=0.33, line_dash="dash", line_color="red", opacity=0.4)
    fig.update_layout(
        yaxis_title="Macro F1 Score (5-fold CV)",
        xaxis_title="Model",
    )
    st.caption("This bar chart highlights each model's cross-validation Macro F1 score and variability, with the red line marking the random baseline.")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Full Comparison Table")
    styled = comparison_df.sort_values('Macro F1', ascending=False).style\
        .background_gradient(subset=['Macro F1', 'Test Accuracy', 'CV F1 Mean'], cmap='YlGn')\
        .format({'Test Accuracy': '{:.4f}', 'Macro F1': '{:.4f}',
                 'CV F1 Mean': '{:.4f}', 'CV F1 Std': '{:.4f}'})
    st.dataframe(styled, use_container_width=True)

    st.subheader("🏆 Best Model Evaluation — Random Forest")
    
    # Feature engineering summary
    with st.expander("📐 Feature Engineering Summary (71 features)"):
        st.markdown("""
        **Text-based features** (×3 for good_prompt, bad_prompt, task_description):
        - Character length, word count, sentence count, average word length
        - Question marks, bullet points, numbered items, code blocks
        - Example mentions, role assignments, step instructions
        - Constraint word count, instruction density
        
        **Derived ratio features:**
        - length ratio (good / bad), word ratio, sentence ratio
        
        **Technique indicator features:**
        - Binary flags for 12 known prompting techniques
        - Number of techniques used
        
        **Prompt type one-hot encoding:**
        - 16 binary features for prompt categories
        """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    col3.metric("Macro Precision", f"{metrics['macro_precision']:.3f}")
    col4.metric("Macro Recall", f"{metrics['macro_recall']:.3f}")

    st.subheader("Confusion Matrix")
    test_pred_path = "models/test_predictions.csv"
    if os.path.exists(test_pred_path):
        pred_df = pd.read_csv(test_pred_path)
        if {"true_complexity", "pred_complexity"}.issubset(pred_df.columns):
            cm = confusion_matrix(
                pred_df["true_complexity"],
                pred_df["pred_complexity"],
                labels=CLASS_ORDER,
            )
            cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
            fig = px.imshow(
                cm_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                xaxis_title="Predicted label",
                yaxis_title="True label",
                title="Confusion Matrix (Counts)",
            )
            st.caption("This heatmap highlights how often the model correctly vs. incorrectly predicted each complexity class, with diagonal cells showing correct predictions.")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Per-Class Metrics")
            report = classification_report(
                pred_df["true_complexity"],
                pred_df["pred_complexity"],
                labels=CLASS_ORDER,
                target_names=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            )
            per_class = {
                c: report[c] for c in CLASS_ORDER if c in report
            }
            per_class_df = pd.DataFrame(per_class).T[["precision", "recall", "f1-score"]]
            per_class_df = per_class_df.reset_index().rename(columns={"index": "Class"})

            melt_df = per_class_df.melt(
                id_vars="Class",
                var_name="Metric",
                value_name="Score",
            )
            fig2 = px.bar(
                melt_df,
                x="Class",
                y="Score",
                color="Metric",
                barmode="group",
                title="Per-Class Precision, Recall, F1-Score",
            )
            fig2.update_layout(yaxis=dict(range=[0, 1.0]))
            st.caption("This chart highlights precision, recall, and F1-score for each complexity class, revealing which labels the model handles best.")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Test predictions file found but columns are missing.")
    else:
        st.info("Run the training script to generate test predictions and confusion matrix data.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with explain_tab:
    st.title("🔍 Model Explainability")
    
    st.markdown("""
    Understanding *why* a model makes predictions is critical for trust and debugging.
    Here we analyze **SHAP values**, **feature importances**, and how individual features correlate with complexity.
    """)
    
    # ─── SHAP Explainability ───────────────────────────────────────────────────
    st.subheader("🔬 SHAP Explainability")
    TREE_MODELS = {"Decision Tree", "Random Forest", "Gradient Boosting"}
    shap_available = [m for m in models.keys() if m in TREE_MODELS]
    if not shap_available:
        st.info("SHAP is available for Decision Tree, Random Forest, and Gradient Boosting. Train those models first.")
    else:
        shap_model_choice = st.selectbox(
            "Select model for SHAP analysis",
            shap_available,
            index=shap_available.index("Random Forest") if "Random Forest" in shap_available else 0,
        )
        st.caption("SHAP is available for Decision Tree, Random Forest, and Gradient Boosting. For other models, use the Feature Importance Explorer below.")
        shap_model = models.get(shap_model_choice)
        if shap_model is not None:
            try:
                import shap
                X_shap = extract_features(df)
                n_background = min(150, len(X_shap) - 1)
                background_sampled = X_shap.sample(n=n_background, random_state=42)
                background_np = background_sampled.to_numpy()
                explainer = shap.TreeExplainer(shap_model, background_np)
                shap_vals = explainer(background_np)
                if hasattr(shap_vals, "values"):
                    vals = shap_vals.values
                else:
                    vals = np.array(shap_vals)
                if vals.ndim == 3:
                    mean_abs_shap = np.abs(vals).mean(axis=(0, 1))
                else:
                    mean_abs_shap = np.abs(vals).mean(axis=0)
                shap_series = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
                top_shap = st.slider("Top features for SHAP", 5, 40, 20, key="shap_top")
                top_df = shap_series.head(top_shap).reset_index()
                top_df.columns = ["Feature", "Mean |SHAP|"]
                fig = px.bar(
                    top_df,
                    x="Mean |SHAP|",
                    y="Feature",
                    orientation="h",
                    title=f"SHAP Summary — {shap_model_choice} (mean |SHAP value|)",
                    color="Mean |SHAP|",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(xaxis_title="Mean |SHAP value|", yaxis_title="")
                st.caption("This chart highlights which features most strongly influence the model's predictions on average, with higher bars indicating greater impact.")
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("SHAP values table"):
                    st.dataframe(
                        shap_series.reset_index().rename(columns={"index": "Feature", 0: "Mean |SHAP|"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.subheader("📉 SHAP Waterfall (single-instance explanation)")
                sample_idx = st.number_input(
                    "Row index to explain (0–{})".format(len(X_shap) - 1),
                    min_value=0,
                    max_value=len(X_shap) - 1,
                    value=0,
                    step=1,
                    key="shap_waterfall_idx",
                )
                if st.button("Show waterfall", key="shap_waterfall_btn"):
                    row = X_shap.iloc[[sample_idx]]
                    row_np = row.to_numpy()
                    single_shap = explainer(row_np)
                    try:
                        if hasattr(single_shap, "values") and single_shap.values.ndim == 3:
                            pred_class = int(shap_model.predict(row)[0])
                            exp = shap.Explanation(
                                values=single_shap.values[0, :, pred_class],
                                base_values=(
                                    explainer.expected_value[pred_class]
                                    if isinstance(explainer.expected_value, (list, np.ndarray))
                                    else explainer.expected_value
                                ),
                                data=row_np[0],
                                feature_names=feature_names,
                            )
                        else:
                            exp = single_shap[0]
                        st.caption("This waterfall highlights how each feature pushes the prediction for one specific instance from the base value to the final output.")
                        fig_wf, _ = plt.subplots(figsize=(10, 8))
                        shap.plots.waterfall(exp, max_display=15, show=False)
                        plt.tight_layout()
                        st.pyplot(fig_wf)
                        plt.close()
                    except Exception as wf_err:
                        st.warning(f"Waterfall plot failed: {wf_err}")
            except Exception as e:
                st.warning(f"SHAP could not be computed for {shap_model_choice}: {e}")
    
    # Interactive feature importance
    st.subheader("📊 Feature Importance Explorer (tree-based)")
    rf_model = models.get('Random Forest')
    if rf_model is not None:
        importances = rf_model.feature_importances_
        feat_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        top_n = st.slider("Number of top features to show", 5, 50, 20)
        top = feat_series.head(top_n)

        top_df = top.reset_index()
        top_df.columns = ["Feature", "Importance"]
        fig = px.bar(
            top_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top {top_n} Feature Importances — Random Forest",
            color="Importance",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(xaxis_title="Importance score", yaxis_title="")
        st.caption("This chart highlights which features the Random Forest model uses most when making split decisions, based on Gini importance.")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("See all feature importances as table"):
            st.dataframe(feat_series.reset_index().rename(
                columns={'index': 'Feature', 0: 'Importance'}
            ).style.background_gradient(subset=['Importance'], cmap='YlOrRd'),
            use_container_width=True)

    # Feature vs complexity boxplots (interactive)
    st.subheader("📦 Feature Distribution by Complexity")
    feat_df = extract_features(df)
    feat_df['complexity'] = df['complexity'].values
    
    numeric_feats = [c for c in feat_df.columns 
                     if not c.startswith('tech_') and not c.startswith('type_') and c != 'complexity']
    
    selected_feat = st.selectbox("Choose a feature to visualize:", numeric_feats, index=0)

    fig = px.box(
        feat_df,
        x="complexity",
        y=selected_feat,
        color="complexity",
        category_orders={"complexity": CLASS_ORDER},
        color_discrete_map=PALETTE,
        title=f"{selected_feat} by Complexity",
    )
    fig.update_layout(xaxis_title="Complexity", yaxis_title="Value", showlegend=False)
    st.caption(f"This boxplot highlights how the selected feature ({selected_feat}) is distributed across the three complexity levels, revealing separation or overlap.")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with target
    from scipy.stats import f_oneway
    groups = [feat_df[feat_df['complexity'] == c][selected_feat].dropna() for c in CLASS_ORDER]
    try:
        f_stat, p_val = f_oneway(*groups)
        if p_val < 0.001:
            st.success(f"✅ ANOVA: F={f_stat:.2f}, p={p_val:.4f} — **Statistically significant** difference across complexity levels.")
        elif p_val < 0.05:
            st.warning(f"⚠️ ANOVA: F={f_stat:.2f}, p={p_val:.4f} — Marginally significant difference.")
        else:
            st.info(f"ℹ️ ANOVA: F={f_stat:.2f}, p={p_val:.4f} — No statistically significant difference.")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with predict_tab:
    st.title("⚡ Live Prediction")
    render_section_header(
        "Test the model on your own prompts",
        "Describe the task, provide good/bad prompts, and see the predicted complexity with confidence scores.",
    )

    left, right = st.columns([2, 1])

    with left:
        with card("Prompt details"):
            selected_model_name = st.selectbox("🤖 Select model", list(models.keys()))

            task_desc = st.text_area(
                "📋 Task description",
                placeholder="e.g., Write a Python function to parse JSON and handle errors gracefully...",
                height=80,
            )

            good_prompt = st.text_area(
                "✅ Good (improved) prompt",
                placeholder="The detailed, well-structured version of the prompt...",
                height=150,
            )

            bad_prompt = st.text_area(
                "❌ Bad (original) prompt",
                placeholder="The vague or under-specified version...",
                height=100,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                prompt_type = st.selectbox("📂 Prompt type", PROMPT_TYPES)
            with col_b:
                selected_techs = st.multiselect("🔧 Prompting techniques", TECHNIQUES)

            predict_btn = st.button("🚀 Predict complexity", type="primary", use_container_width=True)

    with right:
        with card("📖 Quick guide"):
            st.markdown(
                """
                **Low complexity** prompts  
                - Simple, direct requests  
                - Single task, clear scope  

                **Medium complexity** prompts  
                - Multi-step instructions  
                - Some context required  

                **High complexity** prompts  
                - Role-playing or expert framing  
                - Multiple constraints  
                - Often encourages reasoning  
                """
            )

        with card("📊 Model performance"):
            for name, row in comparison_df.iterrows():
                st.markdown(f"**{name}** · Macro F1: `{row['Macro F1']:.3f}`")

    prediction_made = False
    if predict_btn:
        if not good_prompt.strip():
            st.error("Please provide at least the good prompt.")
        else:
            tech_str = str(selected_techs)
            row = pd.DataFrame(
                [
                    {
                        "task_description": task_desc or "generic task",
                        "good_prompt": good_prompt,
                        "bad_prompt": bad_prompt or "tell me about it",
                        "prompting_techniques": tech_str,
                        "prompt_type": prompt_type,
                    }
                ]
            )

            X = extract_features(row)

            model = models[selected_model_name]
            pred_enc = model.predict(X)[0]
            pred_label = le.inverse_transform([pred_enc])[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
            else:
                proba = np.array([0.33, 0.34, 0.33])

            prediction_made = True

            with card("🎯 Prediction results"):
                emoji_map = {"low": "🟢", "medium": "🔵", "high": "🔴"}
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted complexity", f"{emoji_map[pred_label]} {pred_label.upper()}")
                c2.metric("Model used", selected_model_name)
                c3.metric("Model Macro F1", f"{comparison_df.loc[selected_model_name, 'Macro F1']:.3f}")

            with card("📊 Prediction confidence"):
                prob_df = pd.DataFrame(
                    {
                        "Class": CLASS_ORDER,
                        "Probability": [proba[le.transform([c])[0]] for c in CLASS_ORDER],
                    }
                )

                fig = px.bar(
                    prob_df,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    color="Class",
                    color_discrete_map=PALETTE,
                    text="Probability",
                    title="Class Probabilities",
                )
                fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                fig.update_layout(
                    xaxis_title="Probability",
                    yaxis_title="",
                    xaxis=dict(range=[0, 1.05]),
                    showlegend=False,
                )
                st.caption("This chart highlights the model's confidence across the three complexity classes for the current prompt.")
                st.plotly_chart(fig, use_container_width=True)

            with card("🔍 Key features for this prediction"):
                key_feats = {
                    "Good prompt length": X["good_len"].values[0],
                    "Good prompt words": X["good_word_count"].values[0],
                    "Good prompt sentences": X["good_sentence_count"].values[0],
                    "Length ratio (good/bad)": round(X["len_ratio"].values[0], 2),
                    "Num. techniques": X["num_techniques"].values[0],
                    "Constraint words": X["good_constraints"].values[0],
                    "Bullet points": X["good_bullet_points"].values[0],
                    "Has role assignment": bool(X["good_has_role"].values[0]),
                    "Instruction density": round(X["good_instruction_density"].values[0], 4),
                }
                feat_disp = pd.DataFrame(list(key_feats.items()), columns=["Feature", "Value"])
                st.dataframe(feat_disp, use_container_width=True, hide_index=True)

    st.divider()
    with card("📚 Try these examples"):
        examples = [
            {
                "name": "🟢 Low — simple count",
                "task": "Count to 10",
                "good": "Generate the sequence of positive integers starting from 1 and ending at 10. Display each integer on a separate line.",
                "bad": "Count to 10",
                "type": "INSTRUCTIONAL",
                "techniques": [],
            },
            {
                "name": "🔵 Medium — recipe generation",
                "task": "One-pot vegetarian pasta recipes for busy nights",
                "good": "Create a list of 5 easy one-pot vegetarian pasta recipes suitable for weeknight dinners. Each recipe should include: a clear title, a brief description highlighting the key flavors, a concise ingredient list with approximate measurements, and step-by-step cooking instructions (3-5 steps max). Format as a numbered list.",
                "bad": "Give me pasta recipes",
                "type": "INSTRUCTIONAL",
                "techniques": ["STRUCTURED_OUTPUT"],
            },
            {
                "name": "🔴 High — creative with role",
                "task": "Develop a creative narrative about neural prosthetics",
                "good": "Imagine you are a science fiction author renowned for your detailed and emotionally resonant narratives. Craft a story set 50 years in the future where neural prosthetics are commonplace. First, brainstorm three distinct scenarios involving neural prosthetics and their impact on society. Consider both positive and negative consequences. For each scenario, outline the key characters, the central conflict, and the resolution. Then select the most compelling scenario and develop it into a short story of approximately 500 words.",
                "bad": "Tell me a story about neural prosthetics.",
                "type": "CREATIVE_WRITING",
                "techniques": ["ROLE_PROMPTING", "TREE_OF_THOUGHTS"],
            },
        ]

        names = [ex["name"] for ex in examples]
        choice = st.selectbox("Load an example prompt", ["– Select –"] + names)
        if choice != "– Select –":
            ex = next(e for e in examples if e["name"] == choice)
            st.session_state["ex_task"] = ex["task"]
            st.session_state["ex_good"] = ex["good"]
            st.session_state["ex_bad"] = ex["bad"]
            st.session_state["ex_type"] = ex["type"]
            st.session_state["ex_techniques"] = ex["techniques"]
            st.rerun()
