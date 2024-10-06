from flask import Flask, request, jsonify
import numpy as np
import joblib  # Assuming you have a pre-trained model saved as a .pkl file

# Load your pre-trained model
model = joblib.load('trained_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Root route
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Flask Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)
    # Extract the input for prediction
    input_data = np.array(data['input'])  # Convert the input to numpy array
    
    # Make predictions using your model
    predictions = model.predict(input_data)
    
    # Convert predictions to a list to send it as a JSON response
    output = predictions.tolist()
    
    # Send the prediction as JSON response
    return jsonify({'predictions': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Start the Flask server
