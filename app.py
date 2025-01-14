from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# run_with_ngrok(app)  # Uncomment if you're using ngrok

# Load the trained model
try:
    model = joblib.load('optimized_gradient_boosting_model.pkl')
except FileNotFoundError:
    model = None
    print("Model file not found. Ensure 'optimized_gradient_boosting_model.pkl' is in the same directory.")

@app.route('/', methods=['GET'])
def home():
    return "Flask Deployment is running successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    # Logic for predictions
    return jsonify({"prediction": prediction, "probabilities": probabilities})

    # Parse the input JSON
    input_data = request.get_json()

    if not input_data or 'features' not in input_data:
        return jsonify({"error": "Invalid input! Please include 'features' in the request JSON."}), 400

    try:
        # Extract features and reshape for prediction
        features = np.array(input_data['features']).reshape(1, -1)
        
        # Validate input shape
        if features.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Model expects {model.n_features_in_} features, but received {features.shape[1]}."}), 400

        # Make predictions
        prediction = model.predict(features).tolist()
        probabilities = model.predict_proba(features).tolist()

        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
