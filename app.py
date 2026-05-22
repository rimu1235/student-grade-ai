from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import joblib
import pandas as pd
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ১. আপনার টেস্ট করা সচল জেমিনি এপিআই কি এখানে বসান
GEMINI_API_KEY = "AIzaSyDHmqR46WNafwIwmh3UOP3yu0d-whBvSlA"
genai.configure(api_key=GEMINI_API_KEY)

# ২. পুরনো ট্রেইন করা পাস-ফেল নির্ধারণী মডেলটি লোড করা
try:
    student_model = joblib.load('student_model.pkl')
except Exception as e:
    student_model = None

# এখানে আমরা ১.৫ ফ্ল্যাশ ভার্সন ব্যবহার করছি, যা একদম নিরাপদ ও স্ট্যাবল
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    gemini_model = None

# ইউজারের ইনপুট থেকে পড়াশোনার ঘণ্টা (সংখ্যা) খুঁজে বের করার ফাংশন
def extract_hours(text):
    text = str(text).strip()
    if re.match(r"^\d+(\.\d+)?$", text):
        return float(text)
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
            
        user_message = data.get('message', data.get('hours', ''))
        if not user_message:
            return jsonify({'status': 'error', 'message': 'Empty message'}), 400
            
        # 🧠 ফিল্টার লজিক: ইউজার কি পড়াশোনার ঘণ্টা দিয়েছে নাকি সাধারণ প্রশ্ন করেছে?
        hours_found = extract_hours(user_message)
        
        if hours_found is not None and student_model is not None:
            # ✅ ইউজার সংখ্যা দিয়েছে -> আমাদের পুরনো মডেল দিয়ে পাস-ফেল হিসাব হবে
            input_data = pd.DataFrame({'Hours': [hours_found]})
            prediction = student_model.predict(input_data)[0]
            probability = student_model.predict_proba(input_data)[0][1] * 100
            
            result_text = "পাস (Pass) 🎉" if int(prediction) == 1 else "ফেল (Fail) ❌"
            reply = f"আপনি জানিয়েছেন দৈনিক {hours_found} ঘণ্টা পড়াশোনা করেন। আমাদের AI মডেলের হিসাব অনুযায়ী আপনার পরীক্ষার ফলাফল অনুমান: {result_text} এবং আপনার পাস করার সম্ভাবনা: {round(probability, 2)}%"
            
            return jsonify({
                'status': 'success',
                'reply': reply
            })
            
        else:
            # ✅ ইউজার সাধারণ প্রশ্ন করেছে -> জেমিনি এআই (1.5) উত্তর দেবে
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
