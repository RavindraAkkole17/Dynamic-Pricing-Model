import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from config import MODEL_FOLDER


class PricingPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_cols = []
        self.numerical_cols = []

    def fit_transform(self, df, target_col='selling_price'):
        df = df.copy().dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col].values

        # Remove non-predictive columns
        drop_cols = [c for c in X.columns if c.lower() in
                     ['id', 'product_id', 'product_name', 'name']]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

        self.categorical_cols = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        self.numerical_cols = X.select_dtypes(
            include=['number']
        ).columns.tolist()

        # Fill missing
        for col in self.numerical_cols:
            X[col] = X[col].fillna(X[col].median())
        for col in self.categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0])

        # Encode categorical
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        self.feature_columns = X.columns.tolist()

        # Scale
        X[self.numerical_cols] = self.scaler.fit_transform(
            X[self.numerical_cols]
        )

        return X.values, y

    def transform_single(self, input_dict):
        df = pd.DataFrame([input_dict])

        # Remove non-predictive columns
        drop_cols = [c for c in df.columns if c.lower() in
                     ['id', 'product_id', 'product_name', 'name']]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                     errors='ignore')

        # Add missing columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_columns]

        # Fill missing
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')

        # Encode
        for col in self.categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = 0

        # Scale
        num_cols = [c for c in self.numerical_cols if c in df.columns]
        if num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df.values

    def save(self):
        joblib.dump(self, os.path.join(MODEL_FOLDER, 'preprocessor.pkl'))

    @staticmethod
    def load():
        path = os.path.join(MODEL_FOLDER, 'preprocessor.pkl')
        if os.path.exists(path):
            return joblib.load(path)
        return None