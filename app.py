from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler object
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    features = [float(x) for x in request.form.values()]
    
    # Ensure that all 8 features are present
    if len(features) != 8:
        return "Error: All features are required", 400
    
    # Preprocess the input data
    input_features = pd.DataFrame([features], columns=['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'])
    input_features_scaled = scaler.transform(input_features)
    
    # Make prediction
    prediction = model.predict(input_features_scaled)
    
    # Convert prediction to human-readable format
    result = 'Demented' if prediction[0] == 1 else 'Nondemented'
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)