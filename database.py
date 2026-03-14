from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(150), nullable=True)
    company = db.Column(db.String(150), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    predictions = db.relationship('PredictionLog', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class PredictionLog(db.Model):
    __tablename__ = 'prediction_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    product_category = db.Column(db.String(50))
    brand = db.Column(db.String(50))
    base_price = db.Column(db.Float)
    competitor_price = db.Column(db.Float)
    cost_price = db.Column(db.Float)
    stock_quantity = db.Column(db.Integer)
    daily_views = db.Column(db.Integer)
    daily_sales = db.Column(db.Integer)
    add_to_cart_count = db.Column(db.Integer)
    customer_rating = db.Column(db.Float)
    review_count = db.Column(db.Integer)
    discount_percentage = db.Column(db.Float)
    advertising_spend = db.Column(db.Float)
    promotion_type = db.Column(db.String(50))
    day_of_week = db.Column(db.Integer)
    season = db.Column(db.String(50))
    demand_index = db.Column(db.Integer)
    price_elasticity = db.Column(db.Float)

    predicted_price = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)