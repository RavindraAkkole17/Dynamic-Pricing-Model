import joblib
import os
from config import MODEL_FOLDER
from model.preprocess import PricingPreprocessor


def predict_price(input_data: dict):
    model_path = os.path.join(MODEL_FOLDER, 'pricing_model.pkl')
    if not os.path.exists(model_path):
        raise Exception('Model not trained yet. Run train_model.py first.')

    model = joblib.load(model_path)
    preprocessor = PricingPreprocessor.load()
    if not preprocessor:
        raise Exception('Preprocessor not found. Run train_model.py first.')

    X = preprocessor.transform_single(input_data)
    prediction = model.predict(X)
    return round(float(prediction[0]), 2)


def is_model_ready():
    return (
        os.path.exists(os.path.join(MODEL_FOLDER, 'pricing_model.pkl'))
        and os.path.exists(os.path.join(MODEL_FOLDER, 'preprocessor.pkl'))
    )