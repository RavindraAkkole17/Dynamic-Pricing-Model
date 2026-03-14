from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash
)
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user
)
from datetime import datetime
import json
import os

from config import SECRET_KEY, SQLALCHEMY_DATABASE_URI, MODEL_FOLDER
from database import db, User, PredictionLog
from model.predict import predict_price, is_model_ready

# ─── App Setup ───
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to continue.'
login_manager.login_message_category = 'warning'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


def get_metrics():
    path = os.path.join(MODEL_FOLDER, 'metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ─── AUTH ───

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)

        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()

        if user and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user, remember=bool(remember))
            flash(f'Welcome back, {user.full_name or user.username}!',
                  'success')
            return redirect(request.args.get('next') or url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        company = request.form.get('company', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        errors = []
        if len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if '@' not in email:
            errors.append('Enter a valid email.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first():
            errors.append('Username already taken.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')

        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html')

        user = User(
            username=username, email=email,
            full_name=full_name, company=company
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


# ─── PAGES ───

@app.route('/')
@login_required
def index():
    model_ready = is_model_ready()
    metrics = get_metrics()
    total_preds = PredictionLog.query.filter_by(
        user_id=current_user.id
    ).count()
    return render_template(
        'index.html',
        model_ready=model_ready,
        metrics=metrics,
        total_predictions=total_preds
    )


@app.route('/dashboard')
@login_required
def dashboard():
    metrics = get_metrics()
    total_preds = PredictionLog.query.filter_by(
        user_id=current_user.id
    ).count()
    recent = PredictionLog.query.filter_by(
        user_id=current_user.id
    ).order_by(PredictionLog.created_at.desc()).limit(5).all()
    return render_template(
        'dashboard.html',
        metrics=metrics,
        total_predictions=total_preds,
        recent_predictions=recent
    )


@app.route('/history')
@login_required
def history():
    predictions = PredictionLog.query.filter_by(
        user_id=current_user.id
    ).order_by(PredictionLog.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)


# ─── API ───

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Build feature dict (must match CSV columns exactly)
        features = {
            'product_category': data.get('product_category', ''),
            'brand': data.get('brand', ''),
            'base_price': float(data.get('base_price', 0)),
            'competitor_price': float(data.get('competitor_price', 0)),
            'cost_price': float(data.get('cost_price', 0)),
            'stock_quantity': int(data.get('stock_quantity', 0)),
            'daily_views': int(data.get('daily_views', 0)),
            'daily_sales': int(data.get('daily_sales', 0)),
            'add_to_cart_count': int(data.get('add_to_cart_count', 0)),
            'customer_rating': float(data.get('customer_rating', 0)),
            'review_count': int(data.get('review_count', 0)),
            'discount_percentage': float(data.get('discount_percentage', 0)),
            'advertising_spend': float(data.get('advertising_spend', 0)),
            'promotion_type': data.get('promotion_type', ''),
            'day_of_week': int(data.get('day_of_week', 0)),
            'season': data.get('season', ''),
            'demand_index': int(data.get('demand_index', 5)),
            'price_elasticity': float(data.get('price_elasticity', 1.0)),
        }

        price = predict_price(features)

        # Save to database
        log = PredictionLog(
            user_id=current_user.id,
            product_category=features['product_category'],
            brand=features['brand'],
            base_price=features['base_price'],
            competitor_price=features['competitor_price'],
            cost_price=features['cost_price'],
            stock_quantity=features['stock_quantity'],
            daily_views=features['daily_views'],
            daily_sales=features['daily_sales'],
            add_to_cart_count=features['add_to_cart_count'],
            customer_rating=features['customer_rating'],
            review_count=features['review_count'],
            discount_percentage=features['discount_percentage'],
            advertising_spend=features['advertising_spend'],
            promotion_type=features['promotion_type'],
            day_of_week=features['day_of_week'],
            season=features['season'],
            demand_index=features['demand_index'],
            price_elasticity=features['price_elasticity'],
            predicted_price=price
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            'success': True,
            'predicted_price': price,
            'prediction_id': log.id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
@login_required
def api_model_info():
    metrics = get_metrics()
    if metrics:
        return jsonify(metrics)
    return jsonify({'error': 'No model found'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)