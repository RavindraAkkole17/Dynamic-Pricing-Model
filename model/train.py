import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import joblib
import os
import json
from config import MODEL_FOLDER
from model.preprocess import DataPreprocessor


MODELS = {
    'random_forest': RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    ),
    'xgboost': XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        random_state=42, n_jobs=-1
    ),
    'lightgbm': LGBMRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbose=-1
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        random_state=42
    ),
    'linear_regression': LinearRegression(),
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=1.0),
}


def train_model(filepath, target_column, model_name='xgboost',
                test_size=0.2):
    # Load data
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df, target_column)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train
    model = MODELS.get(model_name, MODELS['xgboost'])
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    metrics = {
        'model_name': model_name,
        'target_column': target_column,
        'feature_columns': preprocessor.feature_columns,
        'num_samples': len(df),
        'num_features': len(preprocessor.feature_columns),
        'train_metrics': {
            'mae': round(mean_absolute_error(y_train, y_pred_train), 4),
            'rmse': round(
                np.sqrt(mean_squared_error(y_train, y_pred_train)), 4
            ),
            'r2': round(r2_score(y_train, y_pred_train), 4),
            'mape': round(
                mean_absolute_percentage_error(y_train, y_pred_train) * 100, 2
            ),
        },
        'test_metrics': {
            'mae': round(mean_absolute_error(y_test, y_pred_test), 4),
            'rmse': round(
                np.sqrt(mean_squared_error(y_test, y_pred_test)), 4
            ),
            'r2': round(r2_score(y_test, y_pred_test), 4),
            'mape': round(
                mean_absolute_percentage_error(y_test, y_pred_test) * 100, 2
            ),
        },
        'cv_r2_mean': round(cv_scores.mean(), 4),
        'cv_r2_std': round(cv_scores.std(), 4),
        'actual_vs_predicted': {
            'actual': y_test[:50].tolist(),
            'predicted': y_pred_test[:50].tolist()
        }
    }

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feat_imp = dict(zip(preprocessor.feature_columns, importance.tolist()))
        feat_imp = dict(
            sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        )
        metrics['feature_importance'] = feat_imp

    # Save model, preprocessor, metrics
    joblib.dump(model, os.path.join(MODEL_FOLDER, 'pricing_model.pkl'))
    preprocessor.save()

    with open(os.path.join(MODEL_FOLDER, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def compare_models(filepath, target_column, test_size=0.2):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    results = {}
    for name, model in MODELS.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'mae': round(mean_absolute_error(y_test, y_pred), 4),
                'rmse': round(
                    np.sqrt(mean_squared_error(y_test, y_pred)), 4
                ),
                'r2': round(r2_score(y_test, y_pred), 4),
                'mape': round(
                    mean_absolute_percentage_error(y_test, y_pred) * 100, 2
                ),
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    return results