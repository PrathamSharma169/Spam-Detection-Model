from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

app = Flask(__name__)

# Load the trained models and necessary objects
classifier = joblib.load('spam_classifier_model.pkl')
le = joblib.load('label_encoder.pkl')
cv = joblib.load('countvectorizer.pkl')
classifier_email = joblib.load('spam_classifier_model_email.pkl')
cv_email = joblib.load('countvectorizer_email.pkl')

def classify_message(message):
    sp = re.sub('[^a-zA-Z]', ' ', message).lower().split()
    all_stopwords = stopwords.words('english')
    ps = PorterStemmer()
    sp = [ps.stem(word) for word in sp if word not in all_stopwords]
    sp = ' '.join(sp)
    x_new = cv.transform([sp]).toarray()
    prediction = classifier.predict(x_new)
    prediction_label = le.inverse_transform(prediction)
    return "Not Spam" if prediction_label[0] == 'ham' else "Spam"

def classify_message_email(message):
    sp = re.sub('[^a-zA-Z]', ' ', message).lower().split()
    all_stopwords = stopwords.words('english')
    ps = PorterStemmer()
    sp = [ps.stem(word) for word in sp if word not in all_stopwords]
    sp = ' '.join(sp)
    x_new = cv_email.transform([sp]).toarray()
    prediction = classifier_email.predict(x_new)
    return "Not Spam" if prediction[0] == 0 else "Spam"

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spam Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-image: url(https://tycoonsuccess.com/wp-content/uploads/2022/03/spam.jpg);
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 10px;
                width: 400px;
                padding: 30px;
                background-color: rgba(255, 255, 255, .15);  
                backdrop-filter: blur(10px);
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
            }
            label {
                display: block;
                align-items: center;
                justify-content: center;
                margin-bottom: 8px;
            }
            input[type="text"] {
                padding: 8px;
                margin-bottom: 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            button {
                align-items: center;
                justify-content: center;
                padding: 10px;
                display: inline-block;
                background-color: #007bff;
                color: #fff;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                width: 25%;
                align-self: center;
            }
            button:hover {
                background-color: #0056b3;
            }
            #result {
                margin-top: 20px;
                font-weight: bold;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Spam Detection</h1>
            <label for="type">Choose type:</label>
            <select id="type">
                <option value="message">Message</option>
                <option value="email">Email</option>
            </select>
            <label for="input">Enter your text:</label>
            <input type="text" id="input" placeholder="Type your text here...">
            <button onclick="classify()">Predict</button>
            <div id="result"></div>
        </div>
        <script>
            function classify() {
                const type = document.getElementById('type').value;
                const input = document.getElementById('input').value;

                fetch('/classify', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ type: type, input: input })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').innerText = "Result: " + data.result;

                    // Hide the result after 5 seconds
                    setTimeout(() => {
                        document.getElementById('result').innerText = '';
                    }, 5000);
                })
                .catch(error => console.error('Error:', error));
            }
        </script>
    </body>
    </html>
    """

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    if data['type'] == 'message':
        result = classify_message(data['input'])
    elif data['type'] == 'email':
        result = classify_message_email(data['input'])
    else:
        result = "Invalid type selected"
    return jsonify({"result": result})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)