"""
Feature engineering for prompt complexity classification.
Extracts NLP-based and structural features from prompt text.
"""
import re
import ast
import pandas as pd
import numpy as np


TECHNIQUES = [
    'ROLE_PROMPTING', 'CHAIN_OF_THOUGHT', 'TREE_OF_THOUGHTS',
    'CODE_PROMPTING', 'CONTEXTUAL_PROMPTING', 'ONE_SHOT_FEW_SHOT',
    'ZERO_SHOT', 'SELF_CONSISTENCY', 'STEP_BY_STEP',
    'STRUCTURED_OUTPUT', 'CONSTRAINT_PROMPTING', 'EXAMPLE_PROMPTING'
]

PROMPT_TYPES = [
    'INFORMATIONAL', 'QUESTION_ANSWERING', 'PROGRAMMING_CODE_GENERATION',
    'SUMMARIZATION', 'INSTRUCTIONAL', 'ANALYSIS_CRITIQUE',
    'CONVERSATIONAL', 'COMPARISON', 'CREATIVE_WRITING',
    'CODE_EXPLANATION', 'DATA_EXTRACTION', 'ROLE_PLAYING',
    'TRANSLATION', 'STYLE_TONE_CHANGE', 'COMPLETION',
    'CLASSIFICATION_TAGGING'
]


def count_sentences(text):
    if not isinstance(text, str):
        return 0
    return max(1, len(re.split(r'[.!?]+', text.strip())))


def avg_word_length(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0
    words = text.split()
    if not words:
        return 0
    return np.mean([len(w) for w in words])


def count_question_marks(text):
    if not isinstance(text, str):
        return 0
    return text.count('?')


def count_bullet_points(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'^\s*[-*•]', text, re.MULTILINE))


def count_numbered_items(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE))


def count_code_blocks(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'```', text))


def count_examples(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'\bexample\b|\binstance\b|\bsample\b', text, re.IGNORECASE))


def has_role_assignment(text):
    if not isinstance(text, str):
        return 0
    return int(bool(re.search(r'\byou are\b|\bact as\b|\bimagine you\b|\bpretend\b', text, re.IGNORECASE)))


def has_step_instruction(text):
    if not isinstance(text, str):
        return 0
    return int(bool(re.search(r'\bstep[- ]by[- ]step\b|\bfirst.*then.*finally\b|\bbegin by\b', text, re.IGNORECASE)))


def constraint_count(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'\bmust\b|\bshould\b|\bdo not\b|\bavoid\b|\bensure\b|\brequire\b', text, re.IGNORECASE))


def instruction_density(text):
    """Ratio of imperative/instruction words to total words."""
    if not isinstance(text, str) or len(text.split()) == 0:
        return 0
    imperatives = re.findall(r'\bwrite\b|\bcreate\b|\banalyze\b|\bexplain\b|\blist\b|\bgenerate\b|\bdescribe\b|\bcompare\b|\bsummarize\b|\bcalculate\b', text, re.IGNORECASE)
    return len(imperatives) / max(1, len(text.split()))


def parse_techniques(tech_str):
    """Parse prompting techniques from string representation of list."""
    if not isinstance(tech_str, str):
        return []
    try:
        return ast.literal_eval(tech_str)
    except Exception:
        return []


def extract_features(df):
    """
    Extract all features from the dataframe.
    Returns a feature DataFrame.
    """
    features = {}

    for col, prefix in [('good_prompt', 'good'), ('bad_prompt', 'bad'), ('task_description', 'task')]:
        text = df[col].fillna('')
        features[f'{prefix}_len'] = text.str.len()
        features[f'{prefix}_word_count'] = text.str.split().str.len()
        features[f'{prefix}_sentence_count'] = text.apply(count_sentences)
        features[f'{prefix}_avg_word_len'] = text.apply(avg_word_length)
        features[f'{prefix}_question_marks'] = text.apply(count_question_marks)
        features[f'{prefix}_bullet_points'] = text.apply(count_bullet_points)
        features[f'{prefix}_numbered_items'] = text.apply(count_numbered_items)
        features[f'{prefix}_code_blocks'] = text.apply(count_code_blocks)
        features[f'{prefix}_examples'] = text.apply(count_examples)
        features[f'{prefix}_has_role'] = text.apply(has_role_assignment)
        features[f'{prefix}_has_steps'] = text.apply(has_step_instruction)
        features[f'{prefix}_constraints'] = text.apply(constraint_count)
        features[f'{prefix}_instruction_density'] = text.apply(instruction_density)

    # Ratio features
    features['len_ratio'] = (features['good_len'] / (features['bad_len'] + 1))
    features['word_ratio'] = (features['good_word_count'] / (features['bad_word_count'] + 1))
    features['sentence_ratio'] = (features['good_sentence_count'] / (features['bad_sentence_count'] + 1))

    # Technique-based features
    techniques_list = df['prompting_techniques'].apply(parse_techniques)
    features['num_techniques'] = techniques_list.apply(len)
    for tech in TECHNIQUES:
        features[f'tech_{tech}'] = techniques_list.apply(lambda t: int(tech in t))

    # Prompt type one-hot
    for pt in PROMPT_TYPES:
        features[f'type_{pt}'] = (df['prompt_type'] == pt).astype(int)

    return pd.DataFrame(features)


def get_feature_names():
    """Return list of all feature names (for use without a DataFrame)."""
    names = []
    for prefix in ['good', 'bad', 'task']:
        for suffix in ['len', 'word_count', 'sentence_count', 'avg_word_len',
                       'question_marks', 'bullet_points', 'numbered_items',
                       'code_blocks', 'examples', 'has_role', 'has_steps',
                       'constraints', 'instruction_density']:
            names.append(f'{prefix}_{suffix}')
    names += ['len_ratio', 'word_ratio', 'sentence_ratio', 'num_techniques']
    for tech in TECHNIQUES:
        names.append(f'tech_{tech}')
    for pt in PROMPT_TYPES:
        names.append(f'type_{pt}')
    return names
