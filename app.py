from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("modelN.pkl", "rb") as f:
    model = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>

    *{
        margin:0;
        padding:0;
        box-sizing:border-box;
        font-family:Arial, sans-serif;
    }

    body{
        background:linear-gradient(135deg,#0f172a,#1e293b,#2563eb);
        min-height:100vh;
        display:flex;
        justify-content:center;
        align-items:center;
        padding:20px;
    }

    .container{
        width:100%;
        max-width:1000px;
        background:white;
        border-radius:20px;
        padding:30px;
        box-shadow:0px 10px 30px rgba(0,0,0,0.2);
    }

    h1{
        text-align:center;
        color:#2563eb;
        margin-bottom:10px;
    }

    p{
        text-align:center;
        color:gray;
        margin-bottom:20px;
    }

    .grid{
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:15px;
    }

    input{
        width:100%;
        padding:12px;
        border:1px solid #ddd;
        border-radius:10px;
        font-size:15px;
    }

    button{
        width:100%;
        margin-top:20px;
        padding:14px;
        border:none;
        border-radius:10px;
        background:#2563eb;
        color:white;
        font-size:18px;
        cursor:pointer;
    }

    button:hover{
        background:#1d4ed8;
    }

    .result{
        margin-top:20px;
        padding:15px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
        border-radius:10px;
        background:#f1f5f9;
    }

    @media(max-width:768px){
        .grid{
            grid-template-columns:1fr;
        }
    }

    </style>
</head>

<body>

<div class="container">

    <h1>Customer Churn Prediction</h1>
    <p>Machine Learning Powered Customer Analytics</p>

    <form method="POST">

        <div class="grid">

            <input type="number" name="customer_id"
            placeholder="Customer ID" required>

            <input type="number" name="age"
            placeholder="Age" required>

            <input type="number" name="gender"
            placeholder="Gender Encoded Value" required>

            <input type="number" name="city"
            placeholder="City Encoded Value" required>

            <input type="number" name="tenure_months"
            placeholder="Tenure Months" required>

            <input type="number" step="0.01"
            name="avg_order_value"
            placeholder="Average Order Value" required>

            <input type="number" name="total_orders"
            placeholder="Total Orders" required>

            <input type="number"
            name="last_purchase_days_ago"
            placeholder="Last Purchase Days Ago" required>

            <input type="number"
            name="support_tickets"
            placeholder="Support Tickets" required>

            <input type="number"
            name="subscription_type"
            placeholder="Subscription Type Encoded Value" required>

        </div>

        <button type="submit">
            Predict Customer Churn
        </button>

    </form>

    {% if prediction %}
    <div class="result">
        {{ prediction }}
    </div>
    {% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        features = np.array([[
            float(request.form["customer_id"]),
            float(request.form["age"]),
            float(request.form["gender"]),
            float(request.form["city"]),
            float(request.form["tenure_months"]),
            float(request.form["avg_order_value"]),
            float(request.form["total_orders"]),
            float(request.form["last_purchase_days_ago"]),
            float(request.form["support_tickets"]),
            float(request.form["subscription_type"])
        ]])

        pred = model.predict(features)[0]

        if pred == 1:
            prediction = "⚠️ Customer Likely To Churn"
        else:
            prediction = "✅ Customer Likely To Stay"

    return render_template_string(
        HTML,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
