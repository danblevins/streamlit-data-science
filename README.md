# 🤖 Prompt Complexity Classifier

> **End-to-End Data Science Project** — Predicting the complexity of LLM prompts using engineered NLP features and ensemble machine learning models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)

## 📋 Project Overview

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

| Model | Test Accuracy | Macro F1 | CV F1 Mean |
|---|---|---|---|
| **Random Forest** 🏆 | 66.2% | 0.652 | 0.677 |
| Gradient Boosting | 68.3% | 0.669 | 0.665 |
| SVM (RBF) | 63.4% | 0.634 | 0.620 |
| Logistic Regression | 59.3% | 0.598 | 0.591 |
| K-Nearest Neighbors | 49.7% | 0.422 | 0.457 |

*5-fold stratified cross-validation used for model selection.*

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
│   └── labeled_validation_final.csv  # Supplementary validation data
├── src/
│   ├── features.py                    # Feature engineering
│   ├── eda.py                         # EDA & visualization scripts
│   └── train.py                       # Model training & evaluation
├── models/
│   ├── best_model.pkl                 # Best saved model (Random Forest)
│   ├── label_encoder.pkl              # Label encoder
│   ├── feature_names.pkl             # Feature name list
│   ├── model_comparison.csv          # All model metrics
│   └── best_model_metrics.json       # Best model test metrics
├── figures/                           # Generated EDA & model plots
├── app/
│   └── streamlit_app.py              # Main Streamlit application
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

The deployed app includes 5 interactive sections:

1. **🏠 Overview** — Project summary, dataset stats, workflow steps
2. **📊 EDA** — Interactive distribution plots, text length analysis, technique breakdowns
3. **🤖 Model Comparison** — CV results, confusion matrices, per-class metrics
4. **🔍 Explainability** — Feature importances, interactive boxplots, ANOVA tests
5. **⚡ Live Prediction** — Enter any prompt and get real-time complexity classification

## 🛠️ Technical Stack

- **Data Processing:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Model Persistence:** `joblib`
- **Statistics:** `scipy`
- **Web App:** `streamlit`

## 📈 Future Work

- Incorporate TF-IDF or sentence embedding features
- Add XGBoost / LightGBM models
- Hyperparameter tuning with Optuna
- True SHAP explanations for individual predictions
- Fine-tune a small BERT model for end-to-end classification

---
*Project developed as part of an end-to-end data science homework assignment.*
