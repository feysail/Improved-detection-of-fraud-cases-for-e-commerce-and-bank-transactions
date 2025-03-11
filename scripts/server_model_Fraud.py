from flask import Flask, request, jsonify
import joblib
import logging


logging.basicConfig(level=logging.INFO)
app = Flask(__name__)


model_fraud = joblib.load(r'C:\Users\ASUS VIVO\Desktop\e-commerce\Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions\model\model_fraud.pkl') 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features'] 
    prediction = model_fraud.predict([features])
    app.logger.info(f'Input features: {features}, Prediction: {prediction[0]}')
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)