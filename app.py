from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import joblib
import pandas as pd
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ১. আপনার জেমিনি এপিআই কি এখানে বসান
GEMINI_API_KEY = "AIzaSyCBptnMPgVzEPeZf2caeGPwhCW-ceaofjE"
genai.configure(api_key=GEMINI_API_KEY)

# ২. পুরনো ট্রেইন করা পাস-ফেল নির্ধারণী মডেলটি লোড করা
try:
    student_model = joblib.load('student_model.pkl')
except Exception as e:
    print("Warning: student_model.pkl loaded locally or not found. System will fallback to Gemini.")
    student_model = None

try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    gemini_model = None

# ইউজারের ইনপুট থেকে পড়াশোনার ঘণ্টা (সংখ্যা) খুঁজে বের করার ফাংশন
def extract_hours(text):
    text = str(text).strip()
    # যদি ইউজার শুধু সংখ্যা লেখে (যেমন: 5 বা 6.5)
    if re.match(r"^\d+(\.\d+)?$", text):
        return float(text)
    
    # যদি ইউজার লেখে "৫ ঘন্টা" বা "6 hours"
    match = re.search(r"(\d+(\.\d+)?)\s*(ঘণ্টা|ঘন্টা|hour|hours)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'success'}), 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
            
        # চ্যাটবক্স থেকে মেসেজটি নেওয়া
        user_message = data.get('message', data.get('hours', ''))
        
        if not user_message:
            return jsonify({'status': 'error', 'message': 'Empty message'}), 400
            
        # 🧠 ফিল্টার লজিক: ইউজার কি পড়াশোনার ঘণ্টা জানতে চাচ্ছে নাকি সাধারণ প্রশ্ন করছে?
        hours_found = extract_hours(user_message)
        
        if hours_found is not None and student_model is not None:
            # ✅ ইউজার সংখ্যা বা ঘণ্টা দিয়েছে -> আমাদের পুরনো মডেল দিয়ে পাস-ফেল হিসাব হবে
            input_data = pd.DataFrame({'Hours': [hours_found]})
            prediction = student_model.predict(input_data)[0]
            probability = student_model.predict_proba(input_data)[0][1] * 100
            
            result_text = "পাস (Pass) 🎉" if int(prediction) == 1 else "ফেল (Fail) ❌"
            reply = f"আপনি জানিয়েছেন দৈনিক {hours_found} ঘণ্টা পড়াশোনা করেন। আমাদের AI মডেলের হিসাব অনুযায়ী আপনার পরীক্ষার ফলাফল অনুমান: {result_text} এবং আপনার পাস করার সম্ভাবনা: {round(probability, 2)}%"
            
            return jsonify({
                'status': 'success',
                'result': int(prediction),
                'pass_probability': round(probability, 2),
                'reply': reply
            })
            
        else:
            # ✅ ইউজার সাধারণ প্রশ্ন করেছে -> জেমিনি এআই উত্তর দেবে
            if gemini_model is None:
                return jsonify({'status': 'error', 'message': 'Gemini AI not initialized'}), 500
                
            response = gemini_model.generate_content(str(user_message))
            return jsonify({
                'status': 'success',
                'reply': response.text
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run()
