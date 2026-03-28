from unicodedata import name

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def age_cat(years):

    """Categorizes age into discrete bins."""

    if years <= 20:
        return '0-20'
    elif years > 20 and years <= 30:
        return '20-30'
    elif years > 30 and years <= 40:
        return '30-40'
    elif years > 40 and years <= 50:
        return '40-50'
    elif years > 50 and years <= 60:
        return '50-60'
    elif years > 60 and years <= 70:
        return '60-70'
    else:
        return '70+'

def feature_engineering(df: pd.DataFrame, campaign_limit: float = None) -> pd.DataFrame:

    """Groups categories and creates new features to reduce cardinality and noise."""
    df_copy = df.copy()

    if campaign_limit is None:
        campaign_limit = df_copy['campaign'].quantile(0.99)
    df_copy['campaign'] = df_copy['campaign'].clip(upper=campaign_limit)

    education_dict = {
                    'basic.9y' : 'basic',
                    'basic.4y' : 'basic',
                    'basic.6y' : 'basic',
                    'illiterate' : 'illiterate',
                    'unknown' : 'unknown',
                    'professional.course' : 'professional',
                    'high.school' : 'high',
                    'university.degree' : 'university'            
         }

    edu_order = {
        'illiterate': 0,
        'basic': 1,
        'high': 2,
        'professional': 3,
        'university': 4,
        'unknown': 2  
    }
    
    job_dict = {
                    'admin.': 'office worker',
                    'management': 'office worker',
                    'technician': 'office worker',
                    'blue-collar': 'blue-collar',
                    'services': 'blue-collar',
                    'housemaid': 'blue-collar',
                    'self-employed': 'self-employed',
                    'entrepreneur': 'self-employed',
                    'retired': 'retired',
                    'student': 'student',
                    'unemployed': 'unemployed',
                    'unknown': 'unknown'
                }
    
    df_copy['education_grouped'] = df_copy['education'].map(education_dict).fillna('unknown')
    df_copy['education_rank'] = df_copy['education_grouped'].map(edu_order)
    df_copy['job_grouped'] = df_copy['job'].map(job_dict).fillna('unknown')
    df_copy['age_category'] = df_copy['age'].apply(age_cat)
    df_copy['is_contacted'] = df_copy['pdays'].apply(lambda x: 0 if x == 999 else 1)

    return df_copy

def get_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numeric and categorical columns, excluding identifiers and target.
    """
    target_col = 'y'
    drop_cols = [target_col,  'job', 'education', 'day_of_week', 'duration', 'pdays', 'education_grouped']  
    
    input_cols = [col for col in df.columns if col not in drop_cols]
    numeric_cols = df[input_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df[input_cols].select_dtypes(include='object').columns.tolist()
    
    return numeric_cols, categorical_cols

# --- Imputation ---

def fit_imputers(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[SimpleImputer, SimpleImputer]:
    """
    Fits numeric (median) and categorical (most_frequent) imputers on the dataframe.
    """
    imputer_num = SimpleImputer(strategy='median').fit(df[numeric_cols])
    imputer_cat = SimpleImputer(strategy='most_frequent').fit(df[categorical_cols])
    return imputer_num, imputer_cat

def apply_imputation(df: pd.DataFrame, imputer_num: SimpleImputer, imputer_cat: SimpleImputer, 
                     numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """
    Applies pre-fitted imputers to the dataframe.
    """
    df_copy = df.copy()
    df_copy[numeric_cols] = imputer_num.transform(df_copy[numeric_cols])
    df_copy[categorical_cols] = imputer_cat.transform(df_copy[categorical_cols])
    return df_copy

# --- Encoding ---

def fit_encoder(df: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fits a OneHotEncoder on the categorical columns.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary').fit(df[categorical_cols])
    return encoder

def apply_encoding(df: pd.DataFrame, encoder: OneHotEncoder, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Applies OneHotEncoding and adds new encoded columns to the dataframe.
    """
    df_copy = df.copy()
    encoded_data = encoder.transform(df_copy[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    # Create a DataFrame with encoded columns and merge
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_copy.index)
    df_copy = df_copy.drop(columns=categorical_cols)

    return pd.concat([df_copy, encoded_df], axis=1)

# --- Scaling ---

def fit_scaler(df: pd.DataFrame, numeric_cols: List[str]) -> StandardScaler:
    """
    Fits a StandardScaler on the numeric columns.
    """
    return StandardScaler().fit(df[numeric_cols])

def apply_scaling(df: pd.DataFrame, scaler: StandardScaler, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Applies pre-fitted StandardScaler to the numeric columns.
    """
    df_copy = df.copy()
    df_copy[numeric_cols] = scaler.transform(df_copy[numeric_cols])
    return df_copy


def split_and_engineer(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits raw data into train/val and applies feature engineering to each part separately.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['y'])
    
    campaign_limit = train_df['campaign'].quantile(0.99)

    train_df = feature_engineering(train_df,campaign_limit=campaign_limit)
    val_df = feature_engineering(val_df,campaign_limit=campaign_limit)
    
    return train_df, val_df, campaign_limit



# --- Main Pipelines ---

def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical: bool = True, scaler_numeric: bool = True) -> Dict[str, Any]:
    """
    Full training pipeline: splits data, fits all processors on train, and transforms both train and val.
    """

    train_df = train_df.copy()
    val_df = val_df.copy()

    target_col = 'y'
    numeric_cols, categorical_cols = get_feature_cols(train_df)
    
    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()

    # 1. Impute
    imputer_num, imputer_cat = fit_imputers(train_df, numeric_cols, categorical_cols)
    train_df = apply_imputation(train_df, imputer_num, imputer_cat, numeric_cols, categorical_cols)
    val_df = apply_imputation(val_df, imputer_num, imputer_cat, numeric_cols, categorical_cols)

    # 2. Encode
    encoder = None
    if categorical:
        encoder = fit_encoder(train_df, categorical_cols)
        train_df = apply_encoding(train_df, encoder, categorical_cols)
        val_df = apply_encoding(val_df, encoder, categorical_cols)
        encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    else:
        encoded_cols = categorical_cols
        
        for col_name in encoded_cols:
            train_df[col_name] = train_df[col_name].astype('category')
            val_df[col_name] = val_df[col_name].astype('category')


    # 3. Scale
    scaler = None
    if scaler_numeric:
        scaler = fit_scaler(train_df, numeric_cols)
        train_df = apply_scaling(train_df, scaler, numeric_cols)
        val_df = apply_scaling(val_df, scaler, numeric_cols)

    # Select final features
    final_cols = numeric_cols + encoded_cols
    
    return {
        'train_X': train_df[final_cols],
        'train_y': train_targets,
        'val_X': val_df[final_cols],
        'val_y': val_targets,
        'imputer_num': imputer_num,
        'imputer_cat': imputer_cat,
        'scaler': scaler,
        'encoder': encoder,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoded_cols': encoded_cols
    }

def preprocess_new_data(
    df: pd.DataFrame,
    imputer_num: SimpleImputer,
    imputer_cat: SimpleImputer,
    encoder: OneHotEncoder,
    scaler: Optional[StandardScaler],
    numeric_cols: List[str],
    categorical_cols: List[str],
    campaign_limit: float = None
) -> pd.DataFrame:
    """
    Inference pipeline: applies pre-fitted processors to new data (e.g., test.csv).
    """
    df = feature_engineering(df, campaign_limit=campaign_limit)

    # 1. Impute
    df = apply_imputation(df, imputer_num, imputer_cat, numeric_cols, categorical_cols)
    
    # 2. Encode
    if encoder is not None:
        df = apply_encoding(df, encoder, categorical_cols)
        encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    else:
        encoded_cols = categorical_cols
        
        for col_name in encoded_cols:
            df[col_name] = df[col_name].astype('category')

    # 3. Scale
    if scaler is not None:
        df = apply_scaling(df, scaler, numeric_cols)
        
    return df[numeric_cols + encoded_cols]

def evaluate_model(inputs, targets, model, name=''):
    preds = model.predict(inputs)

    y_pred_proba = None
    roc_auc = 0.0

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(inputs)
        y_pred_proba_1 = y_pred_proba[:, 1]

        roc_auc = roc_auc_score(targets, y_pred_proba_1)

    else:
        print(f'AUROC for {name}: Not available')

    f1_score_ = f1_score(targets, preds, pos_label=1, zero_division=0)

    precision = precision_score(targets, preds, pos_label=1, zero_division=0)
    
    recall = recall_score(targets, preds, pos_label=1, zero_division=0)

    metrics_df = pd.DataFrame({
            'model': [name],
            'roc_auc': [roc_auc],
            'f1_score': [f1_score_],
            'precision': [precision],
            'recall': [recall]
        })

    return metrics_df

def predict_and_plot_cf(inputs, targets, model, ax=None, normalize='true', name=''):

    preds = model.predict(inputs)
    cf = confusion_matrix(targets, preds, normalize=normalize)

    if ax is None:
            plt.figure(figsize=(5, 4))
            ax = plt.gca()

    sns.heatmap(cf, annot=True, cmap='Blues', fmt='.2f', ax=ax)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Target')
    ax.set_title(f'Confusion Matrix - {name}')

def plot_one_model_cf(inputs_train, targets_train, val_inputs, val_targets, model, model_name=''):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    predict_and_plot_cf(inputs_train, targets_train, model, ax=axes[0,0], normalize='true', name=f'Model {model_name} Train (%)')
   
    predict_and_plot_cf(inputs_train, targets_train, model, ax=axes[0,1], normalize=None, name=f'Model {model_name} Train (count)')
    
    predict_and_plot_cf(val_inputs, val_targets, model, ax=axes[1,0], normalize='true', name=f'Model{model_name} Val (%)')
    
    predict_and_plot_cf(val_inputs, val_targets, model, ax=axes[1,1], normalize=None, name=f'Model {model_name} Val (count)')
    plt.tight_layout()
    plt.show()

def predict_and_plot_roc_auc_all(trained_models: Dict[str, Tuple[Any, pd.DataFrame, pd.Series]]):
    plt.figure(figsize=(8, 6))
    for name, (model, X_val_, y_val_) in trained_models.items():
        y_pred_proba = model.predict_proba(X_val_)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val_, y_pred_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} ROC curve (area = {roc_auc:.3f})')


    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — all models (validation set)')
    plt.legend(loc="lower right")
    plt.show()

def update_metrics_dict(models_dict, X_train_boost, X_val_boost, train_targets_boost, val_targets_boost, results_df=None, trained_models=None):
        
    all_metrics_list = []

    if trained_models is None:
        trained_models = {}
    
    if results_df is None:
        results_df = pd.DataFrame(columns=['model', 'dataset', 'roc_auc', 'f1_score', 'precision', 'recall'])


    for name, model in models_dict.items():
        trained_models[name] = (model, X_val_boost, val_targets_boost)
        train_metrics = evaluate_model(X_train_boost, train_targets_boost, model, name)
        train_metrics['dataset'] = 'train'
        val_metrics = evaluate_model(X_val_boost, val_targets_boost, model, name)
        val_metrics['dataset'] = 'validation'

        all_metrics_list.extend([train_metrics, val_metrics])

    new_df = pd.concat(all_metrics_list, ignore_index=True)
    results_df = pd.concat([results_df, new_df], ignore_index=True)
    
    cols = ['model', 'dataset', 'roc_auc', 'f1_score', 'precision', 'recall']
    results_df = results_df.reindex(columns=cols)

    return results_df, trained_models