from flask import Flask, request, jsonify
import numpy as np
import joblib  # Assuming you have a pre-trained model saved as a .pkl file

# Load your pre-trained model (modify the path to your model file)
# model = joblib.load('C:\\Users\\abdul\\Desktop\\trained_model.pkl')  # Make sure the model is in the correct path
model = joblib.load('trained_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

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
