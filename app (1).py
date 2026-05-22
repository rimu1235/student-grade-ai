from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

try:
    model = joblib.load('student_model.pkl')
except Exception as e:
    model = None

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'success'}), 200
        
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model file not found'}), 500
        
    try:
        data = request.get_json()
        if not data or 'hours' not in data:
            return jsonify({'status': 'error', 'message': 'No input data'}), 400
            
        study_hours = float(data['hours'])
        input_data = pd.DataFrame({'Hours': [study_hours]})
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
        
        return jsonify({
            'status': 'success',
            'result': int(prediction),
            'pass_probability': round(probability, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run()
