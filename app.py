from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("modelN.pkl", "rb") as f:
    model = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Customer Churn Prediction</title>

<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    font-family:'Segoe UI',sans-serif;
}

body{
    min-height:100vh;
    display:flex;
    justify-content:center;
    align-items:center;
    background:linear-gradient(135deg,#0f172a,#1e293b,#2563eb);
    padding:20px;
}

.container{
    width:100%;
    max-width:1000px;
    background:rgba(255,255,255,0.1);
    backdrop-filter:blur(15px);
    border-radius:20px;
    padding:35px;
    box-shadow:0 8px 32px rgba(0,0,0,.3);
    color:white;
}

.header{
    text-align:center;
    margin-bottom:25px;
}

.header h1{
    font-size:2.3rem;
}

.header p{
    color:#dbeafe;
    margin-top:10px;
}

.form-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:15px;
}

.input-box{
    position:relative;
}

.input-box i{
    position:absolute;
    left:15px;
    top:15px;
    color:#2563eb;
}

input,select{
    width:100%;
    padding:12px 12px 12px 42px;
    border:none;
    border-radius:10px;
    outline:none;
}

.btn{
    width:100%;
    margin-top:20px;
    padding:14px;
    border:none;
    border-radius:10px;
    background:#2563eb;
    color:white;
    font-size:18px;
    cursor:pointer;
    font-weight:bold;
}

.btn:hover{
    background:#1d4ed8;
}

.result{
    margin-top:25px;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    background:rgba(255,255,255,0.12);
}

.footer{
    margin-top:25px;
    text-align:center;
    color:#cbd5e1;
}

@media(max-width:768px){
    .form-grid{
        grid-template-columns:1fr;
    }
}

</style>
</head>

<body>

<div class="container">

<div class="header">
<h1><i class="fas fa-users"></i> Customer Churn Prediction</h1>
<p>Machine Learning Powered Customer Retention Analytics</p>
</div>

<form method="POST">

<div class="form-grid">

<div class="input-box">
<i class="fas fa-id-card"></i>
<input type="number" name="customer_id" placeholder="Customer ID" required>
</div>

<div class="input-box">
<i class="fas fa-user"></i>
<input type="number" name="age" placeholder="Age" required>
</div>

<div class="input-box">
<i class="fas fa-venus-mars"></i>
<select name="gender">
<option value="0">Male</option>
<option value="1">Female</option>
<option value="2">Other</option>
</select>
</div>

<div class="input-box">
<i class="fas fa-city"></i>
<input type="number" name="city" placeholder="City Encoded Value" required>
</div>

<div class="input-box">
<i class="fas fa-calendar"></i>
<input type="number" name="tenure_months" placeholder="Tenure Months" required>
</div>

<div class="input-box">
<i class="fas fa-dollar-sign"></i>
<input type="number" step="0.01" name="avg_order_value" placeholder="Average Order Value" required>
</div>

<div class="input-box">
<i class="fas fa-cart-shopping"></i>
<input type="number" name="total_orders" placeholder="Total Orders" required>
</div>

<div class="input-box">
<i class="fas fa-clock"></i>
<input type="number" name="last_purchase_days_ago" placeholder="Last Purchase Days Ago" required>
</div>

<div class="input-box">
<i class="fas fa-headset"></i>
<input type="number" name="support_tickets" placeholder="Support Tickets" required>
</div>

<div class="input-box">
<i class="fas fa-crown"></i>
<select name="subscription_type">
<option value="0">Basic</option>
<option value="1">Standard</option>
<option value="2">Premium</option>
</select>
</div>

</div>

<button class="btn" type="submit">
<i class="fas fa-chart-line"></i>
 Predict Churn
</button>

</form>

{% if prediction %}
<div class="result">
{{ prediction }}
</div>
{% endif %}

<div class="footer">
Developed by Pranita | Data Analyst & Machine Learning Project
</div>

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
    app.run(host="0.0.0.0", port=5000)
