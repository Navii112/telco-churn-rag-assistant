"""Telco churn: 80% train / 20% test. Selection uses training-only CV.
Run: python churn_pipeline.py --out outputs_cli
Shared learning functions are also imported by churn_analysis.ipynb.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
N_SPLITS = 3

def load_and_clean(csv_path):
    """Fixed transformations only; no statistics learned before splitting."""
    df = pd.read_csv(csv_path)
    if df['customerID'].duplicated().any():
        raise ValueError('Investigate duplicate customer IDs before splitting.')
    target = df['Churn'].astype('string').str.strip()
    if target.isna().any() or not target.isin(['Yes', 'No']).all():
        raise ValueError('Churn must contain only Yes/No.')
    df['Churn'] = target.map({'Yes': 1, 'No': 0}).astype(int)
    if df['Churn'].nunique() != 2:
        raise ValueError('Both classes are required.')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    return df.drop(columns='customerID')


def split_data(df):
    """One holdout split; CV validation folds stay inside training data."""
    X, y = df.drop(columns='Churn'), df['Churn']
    return train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)


def make_features(X, variant='original'):
    """Row-wise formulas use neither targets nor dataset statistics."""
    result = X.copy()
    if variant == 'drop_phone':
        return result.drop(columns='PhoneService')
    allowed = ['original', 'tenure_band', 'average_charge', 'charge_gap', 'engineered']
    if variant not in allowed:
        raise ValueError(variant)
    if variant in ['tenure_band', 'engineered']:
        result['TenureBand'] = pd.cut(result['tenure'], [-1, 12, 24, 48, np.inf],
            labels=['0-12', '13-24', '25-48', '49+']).astype(str)
    average = result['TotalCharges'].div(result['tenure'].replace(0, np.nan))
    if variant in ['average_charge', 'engineered']:
        result['AvgChargePerMonth'] = average
    if variant in ['charge_gap', 'engineered']:
        result['ChargeGap'] = result['MonthlyCharges'] - average
    return result


def make_preprocessor(X):
    numerical = X.select_dtypes(include='number').columns.tolist()
    categorical = X.columns.difference(numerical).tolist()
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), numerical),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))]), categorical)])


def make_pipeline(X, model):
    return Pipeline([('preprocessor', make_preprocessor(X)), ('model', model)])


def xgb_model(**settings):
    params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
        eval_metric='logloss', tree_method='hist', importance_type='gain',
        random_state=SEED, n_jobs=2)
    params.update(settings)
    return XGBClassifier(**params)


def baseline_models():
    return {'LogisticRegression': LogisticRegression(max_iter=2000),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=8,
            min_samples_leaf=5, random_state=SEED, n_jobs=2),
        'XGBoost': xgb_model()}


def metric_scores(y, probabilities):
    predicted = (np.asarray(probabilities) >= 0.5).astype(int)
    return dict(Accuracy=accuracy_score(y, predicted),
        Precision=precision_score(y, predicted, zero_division=0),
        Recall=recall_score(y, predicted, zero_division=0),
        F1=f1_score(y, predicted, zero_division=0),
        **{'ROC-AUC': roc_auc_score(y, probabilities),
           'AP': average_precision_score(y, probabilities)})

def cv_experiment(X, y, model, with_importance=False):
    """Each row is predicted by a fold model that did not train on it."""
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold_auc, importances = [], []
    for train_idx, val_idx in folds.split(X, y):
        pipeline = make_pipeline(X.iloc[train_idx], clone(model))
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        probability = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = probability
        fold_auc.append(roc_auc_score(y.iloc[val_idx], probability))
        if with_importance:
            imp = permutation_importance(pipeline, X.iloc[val_idx], y.iloc[val_idx],
                scoring='roc_auc', n_repeats=3, random_state=SEED, n_jobs=1)
            importances.append(imp.importances)
    assert np.isfinite(oof).all()
    scores = metric_scores(y, oof)
    scores['CV_AUC_mean'] = float(np.mean(fold_auc))
    scores['CV_AUC_std'] = float(np.std(fold_auc))
    frame = None
    if importances:
        values = np.concatenate(importances, axis=1)
        frame = pd.DataFrame({'Feature': X.columns, 'Mean': values.mean(axis=1),
            'Std': values.std(axis=1)}).sort_values('Mean', ascending=False)
    return scores, oof, frame

def compare_candidates(X_train, y_train, out):
    """No test data is passed to this function."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    specs = {name: ('original', model) for name, model in baseline_models().items()}
    # Fixed hypotheses from the earlier report; not chosen on test results.
    specs['XGB_drop_PhoneService'] = ('drop_phone', xgb_model())
    specs['XGB_engineered'] = ('engineered', xgb_model())
    for depth in [2, 3]:
        for trees in [150, 300]:
            if depth == 3 and trees == 300:
                continue  # Same settings as baseline: reuse its CV score.
            specs[f'XGB_depth{depth}_trees{trees}'] = ('original', xgb_model(max_depth=depth, n_estimators=trees))
    # 한 번에 하나씩 바꿔야 어떤 변경의 결과인지 설명하기 쉽습니다.
    for variant in ['tenure_band', 'average_charge', 'charge_gap']:
        specs['XGB_' + variant] = (variant, xgb_model())
    specs['RF_balanced'] = ('original', RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=SEED, n_jobs=2))
    rows, predictions, importance = [], {}, None
    for name, (variant, model) in specs.items():
        print(f'CV: {name}', flush=True)
        scores, oof, frame = cv_experiment(make_features(X_train, variant), y_train,
            model, with_importance=(name == 'XGBoost'))
        rows.append(dict(Model=name, Variant=variant, **scores))
        predictions[name] = oof
        if frame is not None:
            importance = frame
    table = pd.DataFrame(rows).set_index('Model').sort_values('CV_AUC_mean', ascending=False, kind='stable')
    table.to_csv(out / 'cv_comparison.csv', encoding='utf-8-sig')
    importance.to_csv(out / 'cv_permutation_importance.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(predictions, index=X_train.index).to_csv(out / 'cv_oof_predictions.csv')
    return specs, table, importance


def run_grid_searches(X_train, y_train, specs, cv_table, out):
    """Fit the fixed search spaces using training data only; save every setting."""
    out = Path(out)
    # 1. 후보 범위를 미리 정합니다. 테스트 결과를 보고 범위를 수정하지 않습니다.
    search_spaces = {
        'RF_GridSearch': (baseline_models()['RandomForest'], {
            'model__n_estimators': [150, 300],
            'model__max_depth': [4, 8],
            'model__min_samples_leaf': [2, 5],
        }),
        'XGB_GridSearch': (xgb_model(), {
            'model__n_estimators': [150, 300],
            'model__max_depth': [2, 3],
            'model__learning_rate': [0.03, 0.05],
            'model__reg_lambda': [1, 5],
        }),
    }
    searches = {}
    tuning_results = {}
    for name, (base_model, param_grid) in search_spaces.items():
        print('자동 튜닝:', name, flush=True)
        # 2. 원본 학습 80%만 전달합니다. 전처리도 fold 안에서 fit합니다.
        search = GridSearchCV(
            estimator=make_pipeline(X_train, clone(base_model)),
            param_grid=param_grid,
            scoring={'AUC': 'roc_auc', 'Recall': 'recall', 'F1': 'f1'},
            refit='AUC',
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
            n_jobs=1, error_score='raise', return_train_score=True,
        )
        search.fit(X_train, y_train)
        searches[name] = search
        # 3. 가장 좋은 설정의 모델을 기존 후보 목록에 추가합니다.
        best_model = clone(search.best_estimator_.named_steps['model'])
        specs[name] = ('original', best_model)
        scores, oof, _ = cv_experiment(X_train, y_train, best_model)
        np.testing.assert_allclose(scores['CV_AUC_mean'], search.best_score_, atol=1e-12)
        cv_table.loc[name] = {'Variant': 'original', **scores}
        oof_table = pd.read_csv(out/'cv_oof_predictions.csv', index_col=0)
        np.testing.assert_array_equal(oof_table.index, X_train.index)
        oof_table[name] = oof
        oof_table.to_csv(out/'cv_oof_predictions.csv')
        details = pd.DataFrame(search.cv_results_)
        cols = ['params', 'mean_test_AUC', 'std_test_AUC', 'mean_test_Recall',
                'mean_test_F1', 'mean_train_AUC', 'rank_test_AUC']
        tuning_results[name] = details[cols].sort_values('rank_test_AUC')
        print('선택한 설정:', search.best_params_)
        print('최고 평균 CV AUC:', round(search.best_score_, 4))
        tables = out / 'notebook_tables'
        tables.mkdir(parents=True, exist_ok=True)
        tuning_results[name].to_csv(tables / (name + '_results.csv'), index=False, encoding='utf-8-sig')
    cv_table = cv_table.sort_values('CV_AUC_mean', ascending=False, kind='stable')
    # 기존 CV 결과 파일에 추가 후보도 반영합니다.
    cv_table.to_csv(out/'cv_comparison.csv', encoding='utf-8-sig')
    return specs, cv_table, searches, tuning_results

def final_evaluation(X_train, X_test, y_train, y_test, specs, table, out):
    """Lock selection on CV before examining any test outcome."""
    out = Path(out)
    selected_name = table.index[0]
    variant, model = specs[selected_name]
    X_fit = make_features(X_train, variant)
    selected = make_pipeline(X_fit, clone(model)).fit(X_fit, y_train)
    final_prob = selected.predict_proba(make_features(X_test, variant))[:, 1]
    fitted, test_rows, predictions = {}, [], {}
    for name in [*baseline_models(), 'Selected']:
        if name == 'Selected':
            pipeline, probability = selected, final_prob
        else:
            pipeline = make_pipeline(X_train, clone(specs[name][1])).fit(X_train, y_train)
            probability = pipeline.predict_proba(X_test)[:, 1]
        fitted[name] = pipeline
        predictions[name] = probability
        test_rows.append(dict(Model=name, **metric_scores(y_test, probability)))
    test = pd.DataFrame(test_rows).set_index('Model')
    test.to_csv(out / 'test_metrics.csv', encoding='utf-8-sig')
    pd.DataFrame(predictions, index=X_test.index).assign(Actual=y_test).to_csv(out / 'test_predictions.csv')
    bundle = dict(pipeline=selected, variant=variant, selected_name=selected_name,
        input_columns=X_train.columns.tolist(), threshold=0.5)
    joblib.dump(bundle, out / 'selected_model.joblib')
    restored = joblib.load(out / 'selected_model.joblib')
    reload_prob = restored['pipeline'].predict_proba(make_features(X_test, restored['variant']))[:, 1]
    np.testing.assert_allclose(final_prob, reload_prob, rtol=0, atol=0)
    example = X_test.iloc[[0]]
    summary = dict(selected_name=selected_name, selected_variant=variant,
        selection_criterion='highest mean training 3-fold ROC-AUC',
        best_cv_auc=float(table.iloc[0]['CV_AUC_mean']), threshold=0.5,
        train_rows=len(X_train), test_rows=len(X_test), train_churn=int(y_train.sum()),
        test_churn=int(y_test.sum()), train_missing_total=int(X_train['TotalCharges'].isna().sum()),
        transformed_features=len(selected.named_steps['preprocessor'].get_feature_names_out()),
        confusion_matrix=confusion_matrix(y_test, final_prob >= 0.5, labels=[0, 1]).tolist(),
        reloaded_predictions_identical=True,
        example=dict(row_index=int(example.index[0]), tenure=int(example.iloc[0]['tenure']),
            contract=example.iloc[0]['Contract'], monthly_charges=float(example.iloc[0]['MonthlyCharges']),
            probability=float(final_prob[0]), prediction=int(final_prob[0] >= 0.5)))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return test, fitted, summary


def run(csv_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(csv_path)
    X_train, X_test, y_train, y_test = split_data(df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(df)
    pd.DataFrame({'RowIndex': df.index, 'Split': np.where(df.index.isin(X_train.index),
        'Train', 'Test')}).to_csv(out / 'split_membership.csv', index=False)
    specs, table, importance = compare_candidates(X_train, y_train, out)
    specs, table, _, _ = run_grid_searches(X_train, y_train, specs, table, out)
    selection = dict(selected_name=table.index[0], variant=specs[table.index[0]][0],
        criterion='highest mean training 3-fold ROC-AUC', threshold=0.5)
    (out/'selection_before_test.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(table.round(4).to_string(), flush=True)
    test, fitted, summary = final_evaluation(X_train, X_test, y_train, y_test, specs, table, out)
    metadata = dict(python=platform.python_version(), pandas=pd.__version__,
        sklearn=sklearn.__version__, xgboost=xgboost.__version__,
        csv_sha256=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        random_state=SEED, test_size=0.2, cv_splits=N_SPLITS,
        note='Same holdout as prior experiments; not a new independent external validation.')
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nLocked selection:', summary['selected_name'])
    print(test.round(4).to_string())
    print('Reload verification: passed')
    return df, X_train, X_test, y_train, y_test, table, test, fitted, importance, summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path(__file__).parent / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'outputs_cli')
    args = parser.parse_args()
    run(args.csv, args.out)

if __name__ == '__main__':
    main()
