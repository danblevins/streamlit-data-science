"""
Exploratory Data Analysis for Prompt Complexity Classification
Saves all plots to figures/ directory
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import ast
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.features import extract_features, parse_techniques

os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
PALETTE = {'low': '#4CAF50', 'medium': '#2196F3', 'high': '#F44336'}

def load_data():
    df = pd.read_csv('data/prompt_examples_dataset.csv')
    return df

# ─── 1. Class Distribution ─────────────────────────────────────────────────────
def plot_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Complexity
    counts = df['complexity'].value_counts()[['low', 'medium', 'high']]
    bars = axes[0].bar(counts.index, counts.values, 
                        color=[PALETTE[c] for c in counts.index], edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    axes[0].set_title('Prompt Complexity Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Complexity Level')
    axes[0].set_ylabel('Count')
    axes[0].set_ylim(0, 850)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Prompt type
    type_counts = df['prompt_type'].value_counts()
    colors = plt.cm.tab20(np.linspace(0, 1, len(type_counts)))
    axes[1].barh(type_counts.index[::-1], type_counts.values[::-1], color=colors[::-1])
    axes[1].set_title('Prompt Type Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Count')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/01_class_distribution.png', bbox_inches='tight')
    plt.close()
    print("Saved: 01_class_distribution.png")


# ─── 2. Text Length Analysis ──────────────────────────────────────────────────
def plot_text_length(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    cols = [('task_description', 'Task Description'),
            ('bad_prompt', 'Bad Prompt'),
            ('good_prompt', 'Good Prompt')]
    
    for ax, (col, label) in zip(axes, cols):
        for complexity in ['low', 'medium', 'high']:
            subset = df[df['complexity'] == complexity][col].str.len().dropna()
            ax.hist(subset, bins=40, alpha=0.6, label=complexity, color=PALETTE[complexity])
        ax.set_title(f'{label} Length', fontweight='bold')
        ax.set_xlabel('Character Count')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Text Length Distributions by Complexity', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/02_text_lengths.png', bbox_inches='tight')
    plt.close()
    print("Saved: 02_text_lengths.png")


# ─── 3. Prompting Techniques Analysis ─────────────────────────────────────────
def plot_techniques(df):
    all_techs = []
    for idx, row in df.iterrows():
        techs = parse_techniques(row['prompting_techniques'])
        for t in techs:
            all_techs.append({'technique': t, 'complexity': row['complexity']})
    
    tech_df = pd.DataFrame(all_techs)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall technique frequency
    tech_counts = tech_df['technique'].value_counts()
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(tech_counts)))
    axes[0].barh(tech_counts.index[::-1], tech_counts.values[::-1], color=colors[::-1])
    axes[0].set_title('Overall Prompting Technique Frequency', fontweight='bold')
    axes[0].set_xlabel('Count')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Technique by complexity (stacked bar)
    tech_complex = tech_df.groupby(['technique', 'complexity']).size().unstack(fill_value=0)
    top_techs = tech_counts.head(10).index
    tech_complex_top = tech_complex.loc[tech_complex.index.isin(top_techs)]
    
    bottom = np.zeros(len(tech_complex_top))
    for complexity in ['low', 'medium', 'high']:
        if complexity in tech_complex_top.columns:
            vals = tech_complex_top[complexity].values
            axes[1].barh(tech_complex_top.index, vals, left=bottom,
                         label=complexity, color=PALETTE[complexity], alpha=0.85)
            bottom += vals
    
    axes[1].set_title('Techniques by Complexity Level (Top 10)', fontweight='bold')
    axes[1].set_xlabel('Count')
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/03_techniques_analysis.png', bbox_inches='tight')
    plt.close()
    print("Saved: 03_techniques_analysis.png")


# ─── 4. Feature Correlation Heatmap ───────────────────────────────────────────
def plot_feature_correlations(df):
    feat_df = extract_features(df)
    
    # Select key numeric features
    key_feats = [c for c in feat_df.columns if not c.startswith('tech_') and not c.startswith('type_')]
    
    corr = feat_df[key_feats].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                ax=ax, vmin=-1, vmax=1, linewidths=0.3)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/04_correlations.png', bbox_inches='tight')
    plt.close()
    print("Saved: 04_correlations.png")


# ─── 5. Key Feature Boxplots by Complexity ────────────────────────────────────
def plot_feature_boxplots(df):
    feat_df = extract_features(df)
    feat_df['complexity'] = df['complexity'].values
    
    key_features = ['good_len', 'good_word_count', 'good_sentence_count',
                    'len_ratio', 'num_techniques', 'good_constraints',
                    'good_bullet_points', 'good_instruction_density']
    labels = ['Good Prompt Length', 'Good Prompt Words', 'Good Prompt Sentences',
              'Length Ratio (Good/Bad)', 'Num. Techniques', 'Constraint Words',
              'Bullet Points', 'Instruction Density']
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    order = ['low', 'medium', 'high']
    for ax, feat, label in zip(axes, key_features, labels):
        data = [feat_df[feat_df['complexity'] == c][feat].dropna() for c in order]
        bp = ax.boxplot(data, labels=order, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch, c in zip(bp['boxes'], order):
            patch.set_facecolor(PALETTE[c])
            patch.set_alpha(0.7)
        ax.set_title(label, fontweight='bold', fontsize=10)
        ax.set_xlabel('Complexity')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Key Feature Distributions by Complexity Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/05_feature_boxplots.png', bbox_inches='tight')
    plt.close()
    print("Saved: 05_feature_boxplots.png")


# ─── 6. Complexity × Prompt Type Heatmap ──────────────────────────────────────
def plot_complexity_type_heatmap(df):
    pivot = df.groupby(['prompt_type', 'complexity']).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(pivot_pct[['low', 'medium', 'high']], annot=True, fmt='.1f',
                cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title('Complexity Distribution by Prompt Type (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Complexity')
    ax.set_ylabel('Prompt Type')
    plt.tight_layout()
    plt.savefig('figures/06_type_complexity_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Saved: 06_type_complexity_heatmap.png")


if __name__ == '__main__':
    print("Running EDA...")
    df = load_data()
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Complexity distribution:\n{df['complexity'].value_counts()}\n")
    
    plot_class_distribution(df)
    plot_text_length(df)
    plot_techniques(df)
    plot_feature_correlations(df)
    plot_feature_boxplots(df)
    plot_complexity_type_heatmap(df)
    
    print("\nEDA complete. All figures saved to figures/")
