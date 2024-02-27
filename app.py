from flask import Flask, request, jsonify
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

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the JSON request
    data = request.get_json()

    # Validate input data
    if data is None:
        return jsonify({"error": "No JSON data received"}), 400

    # Check if all required features are present
    required_features = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    if not all(feature in data for feature in required_features):
        return jsonify({"error": "All features are required"}), 400

    # Extract input features
    features = [data[feature] for feature in required_features]

    try:
        # Preprocess the input data
        input_features = pd.DataFrame([features], columns=required_features)
        input_features_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_features_scaled)

        # Convert prediction to human-readable format
        result = 'Demented' if prediction[0] == 1 else 'Non-demented'

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
