# 🤖 Prompt Complexity Classifier

> **End-to-End Data Science Project** — Predicting the complexity of LLM prompts using engineered NLP features and ensemble machine learning models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.18-purple.svg)](https://plotly.com)

## 📋 Project Overview

LIVE PROJECT LINK: https://danblevins-ds.streamlit.app/

This project applies the **complete data science workflow** to classify LLM prompts into three complexity levels: **low**, **medium**, and **high**. Understanding prompt complexity is valuable for:

- Routing prompts to appropriately-sized language models
- Estimating inference cost and latency
- Training prompt engineers to write better prompts
- Building quality control systems for LLM workflows

## 📦 Dataset

| Property | Value |
|---|---|
| Source | `prompt_examples_dataset.csv` |
| Size | 1,450 prompts |
| Target | Complexity (low / medium / high) |
| Prompt Types | 16 categories |
| Columns | task description, bad prompt, good prompt, techniques, type |

**Class distribution:**
- Medium: 731 (50.4%)
- High: 381 (26.3%)
- Low: 338 (23.3%)

## 🔧 Feature Engineering

71 features extracted from raw text across three text fields (`task_description`, `good_prompt`, `bad_prompt`):

| Category | Features |
|---|---|
| Text metrics | Length, word count, sentence count, avg word length |
| Structural | Bullet points, numbered items, code blocks, examples |
| Linguistic | Role assignments, step instructions, constraint words, instruction density |
| Ratios | Good/bad length ratio, word ratio, sentence ratio |
| Techniques | 12 binary technique flags + technique count |
| Prompt type | 16 one-hot encoded categories |

## 🏆 Model Results

Seven classifiers trained with **5-fold stratified cross-validation** and **GridSearchCV** hyperparameter tuning:

| Model | Test Accuracy | Macro F1 | CV F1 Mean | Tuning |
|---|---|---|---|---|
| **Random Forest** 🏆 | 66.2% | 0.652 | 0.677 | GridSearchCV |
| Gradient Boosting | 68.3% | 0.669 | 0.665 | GridSearchCV |
| Decision Tree | 62.1% | 0.612 | 0.598 | GridSearchCV |
| MLP (Neural Network) | 61.4% | 0.605 | 0.589 | Early stopping |
| SVM (RBF) | 63.4% | 0.634 | 0.620 | Manual |
| Logistic Regression | 59.3% | 0.598 | 0.591 | Baseline |
| K-Nearest Neighbors | 49.7% | 0.422 | 0.457 | Manual |

*GridSearchCV tuning details:*
- **Decision Tree**: `max_depth` ∈ {3, 5, 7, 10, 15}, `min_samples_leaf` ∈ {5, 10, 20, 50}
- **Random Forest**: `n_estimators` ∈ {50, 100, 200}, `max_depth` ∈ {3, 5, 8, None}
- **Gradient Boosting**: `n_estimators` ∈ {50, 100, 200}, `max_depth` ∈ {3, 4, 5, 6}, `learning_rate` ∈ {0.01, 0.05, 0.1}

## 🔍 Key Findings (EDA)

1. **Prompt length** is the strongest predictor — high-complexity prompts are 3-5× longer than low-complexity ones
2. **Length ratio** (good/bad prompt) effectively captures how much a prompt improves with better engineering
3. **Technique count** correlates positively with complexity
4. **CHAIN_OF_THOUGHT** and **ROLE_PROMPTING** are the most commonly used techniques
5. **ANALYSIS_CRITIQUE** and **CREATIVE_WRITING** tasks skew toward high complexity

## 🗂️ Project Structure

```
prompt-complexity-classifier/
├── data/
│   ├── prompt_examples_dataset.csv    # Main dataset
│   ├── labeled_train_final.csv        # Supplementary training data
│   └── labeled_validation_final.csv   # Supplementary validation data
├── src/
│   ├── features.py                    # Feature engineering
│   ├── eda.py                         # EDA & visualization scripts
│   └── train.py                       # Model training with GridSearchCV
├── models/
│   ├── best_model.pkl                 # Best saved model (Random Forest)
│   ├── random_forest.pkl              # Individual model files
│   ├── gradient_boosting.pkl
│   ├── decision_tree.pkl
│   ├── mlp.pkl
│   ├── svm_rbf.pkl
│   ├── logistic_regression.pkl
│   ├── k-nearest_neighbors.pkl
│   ├── label_encoder.pkl              # Label encoder
│   ├── feature_names.pkl              # Feature name list
│   ├── model_comparison.csv           # All model metrics
│   ├── best_model_metrics.json        # Best model test metrics
│   ├── best_hyperparameters.json      # GridSearchCV best params
│   ├── mlp_history.json               # MLP training loss curve
│   └── test_predictions.csv           # Test set predictions
├── app/
│   ├── streamlit_app.py               # Main Streamlit application
│   └── ui_components.py               # Reusable UI components
├── .streamlit/
│   └── config.toml                    # Streamlit theme configuration
├── requirements.txt
└── README.md
```

## 🚀 Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run EDA

```bash
python src/eda.py
```

### 3. Train models

```bash
python src/train.py
```

### 4. Launch Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## 📊 Streamlit App Features

The deployed app includes 5 interactive tabs:

### 🏠 Overview (Executive Summary)
- Dataset description and prediction task explanation
- Why prompt complexity matters (the "so what")
- Key findings and approach summary
- Quick metrics dashboard

### 📊 EDA (Descriptive Analytics)
- Target distribution visualization
- Text length analysis by complexity
- Prompting techniques breakdown
- Complexity × prompt type heatmap
- Feature correlation matrix

### 🤖 Models (Predictive Analytics)
- Model comparison bar chart with error bars
- Full metrics comparison table
- Confusion matrix visualization
- Per-class precision/recall/F1 chart
- **ROC curves** with AUC scores (multi-class)
- **Best hyperparameters** table
- **MLP training history** (loss curve)
- Written model comparison analysis

### 🔍 Explainability (SHAP Analysis)
- **SHAP bar plot** (mean |SHAP| values)
- **SHAP beeswarm plot** (feature impact direction)
- **SHAP waterfall plot** (single-instance explanation)
- Written interpretation of SHAP results
- Feature importance explorer
- Feature distribution boxplots by complexity
- ANOVA statistical tests

### ⚡ Predict (Interactive Prediction)
- Model selection dropdown
- Text input for task description, good prompt, bad prompt
- Prompt type and technique selectors
- Real-time complexity prediction with confidence scores
- Key feature values for the prediction
- **SHAP waterfall for custom input** (tree models)
- Example prompts to try

## 🛠️ Technical Stack

- **Data Processing:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (GridSearchCV, cross-validation)
- **Explainability:** `shap` (TreeExplainer, beeswarm, waterfall)
- **Visualization:** `plotly`, `matplotlib`, `seaborn`
- **Model Persistence:** `joblib`
- **Statistics:** `scipy`
- **Web App:** `streamlit`

## 📈 Future Work

- Incorporate TF-IDF or sentence embedding features
- Add XGBoost / LightGBM models
- Hyperparameter tuning with Optuna
- Fine-tune a small BERT model for end-to-end classification
- Deploy to Streamlit Community Cloud
