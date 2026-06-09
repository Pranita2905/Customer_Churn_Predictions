import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load your GaussianNB model
with open('modelN.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # Ensure you have an index.html in a 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Categorical mappings (adjust these based on your training encoding)
        gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
        city_map = {'City A': 0, 'City B': 1, 'City C': 2}
        sub_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}

        # Extracting the 10 features defined in the model metadata
        features = [
            float(request.form.get('customer_id', 0)),
            float(request.form.get('age', 0)),
            gender_map.get(request.form.get('gender'), 0),
            city_map.get(request.form.get('city'), 0),
            float(request.form.get('tenure_months', 0)),
            float(request.form.get('avg_order_value', 0)),
            float(request.form.get('total_orders', 0)),
            float(request.form.get('last_purchase_days_ago', 0)),
            float(request.form.get('support_tickets', 0)),
            sub_map.get(request.form.get('subscription_type'), 0)
        ]

        # Reshape for a single prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        # Determine result label (0 or 1)
        result = "Positive" if prediction[0] == 1 else "Negative"
        
        return render_template('index.html', 
                               prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
