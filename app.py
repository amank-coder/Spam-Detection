
from flask import Flask, render_template, request
import pickle

from sklearn import feature_extraction

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    input_text = request.form['text']

    # Make the prediction using the loaded model
    # prediction = model.predict([input_text])
    input_data_features = tfidf.transform([input_text])

    # Make the prediction using the loaded model
    prediction = model.predict(input_data_features)
    # Determine the prediction label
    if prediction[0] == 1:
        result = "Ham mail"
    else:
        result = "Spam mail"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
