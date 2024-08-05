from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('models/lasso.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    # Extract input fields from form_data
    features = [
        float(form_data['temperature']),
        float(form_data['RH']),
        float(form_data['Ws']),
        float(form_data['Rain']),
        float(form_data['FFMC']),
        float(form_data['DMC']),
        float(form_data['ISI']),
        float(form_data['classes']),
        float(form_data['region'])
    ]
    # Preprocess the input features
    features_scaled = scaler.transform([features])
    # Make prediction for FWI
    fwi_prediction = model.predict(features_scaled)
    # Render result template with the prediction
    return render_template('result.html', prediction=fwi_prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
