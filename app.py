```python
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
background:linear-gradient(135deg,#0f172a,#1e293b,#2563eb);
display:flex;
justify-content:center;
align-items:center;
padding:20px;
}

.container{
width:100%;
max-width:1100px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(15px);
padding:40px;
border-radius:25px;
box-shadow:0 8px 32px rgba(0,0,0,0.3);
color:white;
}

.header{
text-align:center;
margin-bottom:30px;
}

.header h1{
font-size:2.5rem;
margin-bottom:10px;
}

.header p{
color:#cbd5e1;
}

.form-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:20px;
}

.input-box{
position:relative;
}

.input-box i{
position:absolute;
left:15px;
top:16px;
color:#2563eb;
}

input,select{
width:100%;
padding:14px 14px 14px 45px;
border:none;
border-radius:12px;
outline:none;
font-size:15px;
}

.btn{
width:100%;
margin-top:25px;
padding:15px;
border:none;
border-radius:12px;
font-size:18px;
font-weight:bold;
background:#2563eb;
color:white;
cursor:pointer;
transition:.3s;
}

.btn:hover{
background:#1d4ed8;
transform:translateY(-2px);
}

.result{
margin-top:25px;
padding:20px;
border-radius:15px;
text-align:center;
font-size:24px;
font-weight:bold;
background:rgba(255,255,255,0.1);
}

.stay{
color:#22c55e;
}

.churn{
color:#ef4444;
}

.footer{
margin-top:30px;
text-align:center;
color:#cbd5e1;
}

@media(max-width:768px){

.form-grid{
grid-template-columns:1fr;
}

.header h1{
font-size:2rem;
}

}

</style>
</head>

<body>

<div class="container">

<div class="header">
<h1><i class="fas fa-users"></i> Customer Churn Prediction</h1>
<p>Machine Learning Powered Customer Retention Analysis</p>
</div>

<form method="POST">

<div class="form-grid">

<div class="input-box">
<i class="fas fa-id-card"></i>
<input type="number" name="customer_id"
placeholder="Customer ID" required>
</div>

<div class="input-box">
<i class="fas fa-user"></i>
<input type="number" name="age"
placeholder="Age" required>
</div>

<div class="input-box">
<i class="fas fa-venus-mars"></i>
<select name="gender">
<option>Male</option>
<option>Female</option>
<option>Other</option>
</select>
</div>

<div class="input-box">
<i class="fas fa-city"></i>
<select name="city">
<option>City A</option>
<option>City B</option>
<option>City C</option>
</select>
</div>

<div class="input-box">
<i class="fas fa-calendar"></i>
<input type="number" name="tenure_months"
placeholder="Tenure Months" required>
</div>

<div class="input-box">
<i class="fas fa-dollar-sign"></i>
<input type="number" step="0.01"
name="avg_order_value"
placeholder="Average Order Value" required>
</div>

<div class="input-box">
<i class="fas fa-cart-shopping"></i>
<input type="number" name="total_orders"
placeholder="Total Orders" required>
</div>

<div class="input-box">
<i class="fas fa-clock"></i>
<input type="number"
name="last_purchase_days_ago"
placeholder="Days Since Last Purchase" required>
</div>

<div class="input-box">
<i class="fas fa-headset"></i>
<input type="number"
name="support_tickets"
placeholder="Support Tickets" required>
</div>

<div class="input-box">
<i class="fas fa-crown"></i>
<select name="subscription_type">
<option>Basic</option>
<option>Standard</option>
<option>Premium</option>
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

gender_map = {
    "Male": 0,
    "Female": 1,
    "Other": 2
}

city_map = {
    "City A": 0,
    "City B": 1,
    "City C": 2
}

subscription_map = {
    "Basic": 0,
    "Standard": 1,
    "Premium": 2
}

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        features = np.array([[
            float(request.form["customer_id"]),
            float(request.form["age"]),
            gender_map.get(request.form["gender"],0),
            city_map.get(request.form["city"],0),
            float(request.form["tenure_months"]),
            float(request.form["avg_order_value"]),
            float(request.form["total_orders"]),
            float(request.form["last_purchase_days_ago"]),
            float(request.form["support_tickets"]),
            subscription_map.get(
                request.form["subscription_type"],0
            )
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
```
