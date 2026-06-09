import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the GaussianNB model 
with open('modelN.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mapping Categorical inputs to numbers (Adjust based on your training data)
        gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
        city_map = {'City A': 0, 'City B': 1, 'City C': 2}
        sub_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}

        # Extracting the 10 features expected by the model 
        features = [
            float(request.form['customer_id']),
            float(request.form['age']),
            gender_map.get(request.form['gender'], 0),
            city_map.get(request.form['city'], 0),
            float(request.form['tenure_months']),
            float(request.form['avg_order_value']),
            float(request.form['total_orders']),
            float(request.form['last_purchase_days_ago']),
            float(request.form['support_tickets']),
            sub_map.get(request.form['subscription_type'], 0)
        ]

        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        # Output label based on prediction (Adjust labels as necessary)
        output = "Likely to Churn" if prediction[0] == 1 else "Likely to Stay"
        color = "#e74c3c" if prediction[0] == 1 else "#2ecc71"

        return render_template('index.html', 
                               prediction_text=f'Analysis Result: {output}',
                               res_color=color)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
