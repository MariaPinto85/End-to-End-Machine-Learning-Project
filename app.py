from flask import Flask, request, jsonify
import joblib
import numpy as np


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
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the model file."}), 500

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

if __name__ == '__main__':
    app.run(debug=True)
