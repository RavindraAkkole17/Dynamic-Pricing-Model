import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY = 'dynamicpricex-secret-2024-change-this'
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'pricing.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False

DATA_FOLDER = os.path.join(BASE_DIR, 'data')
MODEL_FOLDER = os.path.join(BASE_DIR, 'saved_models')
CSV_FILE_PATH = os.path.join(DATA_FOLDER, 'pricing_data.csv')

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)