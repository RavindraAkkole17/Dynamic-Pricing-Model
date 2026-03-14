async function predictPrice() {
    const data = {
        product_category: document.getElementById('product_category').value,
        brand: document.getElementById('brand').value,
        base_price: document.getElementById('base_price').value,
        competitor_price: document.getElementById('competitor_price').value,
        cost_price: document.getElementById('cost_price').value,
        stock_quantity: document.getElementById('stock_quantity').value,
        daily_views: document.getElementById('daily_views').value,
        daily_sales: document.getElementById('daily_sales').value,
        add_to_cart_count: document.getElementById('add_to_cart_count').value,
        customer_rating: document.getElementById('customer_rating').value,
        review_count: document.getElementById('review_count').value,
        discount_percentage: document.getElementById('discount_percentage').value,
        advertising_spend: document.getElementById('advertising_spend').value,
        promotion_type: document.getElementById('promotion_type').value,
        day_of_week: document.getElementById('day_of_week').value,
        season: document.getElementById('season').value,
        demand_index: document.getElementById('demand_index').value,
        price_elasticity: document.getElementById('price_elasticity').value,
    };

    // Validate required fields
    if (!data.base_price || !data.competitor_price || !data.cost_price) {
        alert('Please fill in Base Price, Competitor Price, and Cost Price.');
        return;
    }

    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('result-section').classList.add('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        document.getElementById('loading').classList.add('hidden');

        if (result.error) {
            alert('Error: ' + result.error);
            return;
        }

        const basePrice = parseFloat(data.base_price);
        const costPrice = parseFloat(data.cost_price);
        const predicted = result.predicted_price;
        const margin = ((predicted - costPrice) / predicted * 100).toFixed(1);

        document.getElementById('res-base').textContent = '₹' + basePrice.toFixed(2);
        document.getElementById('res-cost').textContent = '₹' + costPrice.toFixed(2);
        document.getElementById('res-predicted').textContent = '₹' + predicted.toFixed(2);

        const marginEl = document.getElementById('res-margin');
        marginEl.textContent = margin + '%';
        if (margin > 20) {
            marginEl.style.color = '#55efc4';
        } else if (margin > 0) {
            marginEl.style.color = '#ffeaa7';
        } else {
            marginEl.style.color = '#fab1a0';
        }

        document.getElementById('result-section').classList.remove('hidden');
        document.getElementById('result-section').scrollIntoView({
            behavior: 'smooth', block: 'center'
        });

    } catch (err) {
        document.getElementById('loading').classList.add('hidden');
        alert('Prediction failed: ' + err.message);
    }
}