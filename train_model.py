"""
==============================================
  RUN THIS ONCE BEFORE STARTING THE WEBSITE
  python train_model.py
==============================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
import joblib
import json
import os

from config import CSV_FILE_PATH, MODEL_FOLDER
from model.preprocess import PricingPreprocessor


def train():
    print("=" * 60)
    print("  DynamicPriceX — Model Training")
    print("=" * 60)

    # ── Load CSV ──
    if not os.path.exists(CSV_FILE_PATH):
        print(f"\n❌ CSV file not found at: {CSV_FILE_PATH}")
        print(f"   Place your CSV file at: data/pricing_data.csv")
        return

    df = pd.read_csv(CSV_FILE_PATH)
    print(f"\n✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {df.columns.tolist()}")

    # ── Detect target column ──
    target_candidates = [
        'selling_price', 'price', 'dynamic_price', 'final_price',
        'optimized_price', 'predicted_price', 'target_price',
        'sale_price', 'actual_price'
    ]
    target_col = None
    for col in df.columns:
        if col.lower().strip().replace(' ', '_') in target_candidates:
            target_col = col
            break

    if target_col is None:
        print("\n⚠️  Could not auto-detect target column.")
        print(f"   Available columns: {df.columns.tolist()}")
        target_col = input("   Enter the target (price) column name: ").strip()

    print(f"\n🎯 Target column: {target_col}")

    # ── Preprocess ──
    preprocessor = PricingPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col)
    print(f"   Features used: {preprocessor.feature_columns}")

    # ── Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train multiple models and pick best ──
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=15, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            random_state=42
        ),
    }

    best_model = None
    best_name = ''
    best_r2 = -999

    print("\n📊 Training models...\n")
    print(f"   {'Model':<22} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'MAPE':>8}")
    print("   " + "-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        marker = ""
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            marker = " ⭐"

        print(f"   {name:<22} {r2:>8.4f} {mae:>10.2f} {rmse:>10.2f} {mape:>7.2f}%{marker}")

    # ── Save best model ──
    print(f"\n🏆 Best Model: {best_name} (R² = {best_r2:.4f})")

    y_pred_final = best_model.predict(X_test)

    metrics = {
        'model_name': best_name,
        'r2': round(r2_score(y_test, y_pred_final), 4),
        'mae': round(mean_absolute_error(y_test, y_pred_final), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_final)), 4),
        'mape': round(
            mean_absolute_percentage_error(y_test, y_pred_final) * 100, 2
        ),
        'num_samples': len(df),
        'num_features': len(preprocessor.feature_columns),
        'feature_columns': preprocessor.feature_columns,
        'target_column': target_col,
    }

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        imp = dict(zip(
            preprocessor.feature_columns,
            best_model.feature_importances_.tolist()
        ))
        metrics['feature_importance'] = dict(
            sorted(imp.items(), key=lambda x: x[1], reverse=True)
        )

    joblib.dump(best_model, os.path.join(MODEL_FOLDER, 'pricing_model.pkl'))
    preprocessor.save()

    with open(os.path.join(MODEL_FOLDER, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Model saved to: {MODEL_FOLDER}/")
    print(f"   - pricing_model.pkl")
    print(f"   - preprocessor.pkl")
    print(f"   - metrics.json")
    print(f"\n✅ Training complete! Now run: python app.py")
    print("=" * 60)


if __name__ == '__main__':
    train()