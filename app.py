# Dependencies used to build the web app
from flask import Flask, render_template, jsonify, request
from werkzeug.exceptions import HTTPException
import traceback
import io

# Used to load and run the model
# Packages: scikit-learn, joblib
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
model = None


def load_model():


    global model
    if not model:
        # print("--->>>> loading the model...")
        # TODO: Change the filename to match your model's filename
        model = joblib.load("heart_classifier.pkl")
    return model


# Homepage: The heart health form
@app.route('/')
def form():
    # Get the values if specified in the URL (so we can edit them)
    values = request.values

    return render_template('form.html', form_values=values)


@app.route('/process_form', methods=["POST"])
def process_form():
    # Get the values that were submitted in the form, and
    # convert them to correct numeric format (integer or floats)
    values = {
        'age': int(request.form['age']),
        'bp_systolic': int(request.form['bp_systolic']),
        'bp_diastolic': int(request.form['bp_diastolic']),
        'weight_kg': float(request.form['weight_kg']),
        'height_cm': float(request.form['height_cm']),
        'cholesterol': int(request.form['cholesterol'])
    }

    
    bmi = calculate_bmi(values['height_cm'], values['weight_kg'])

    
    cholesterol_descriptions = {
        0: "Normal",
        1: "Above Normal",
        2: "Well Above Normal",
    }

    
    input_values = {
        "Age": values['age'],
        "Blood Pressure": "%s/%s" % (values['bp_systolic'], values['bp_diastolic']),
        "Weight": "%s kg" % values['weight_kg'],
        "Height": "%s cm" % values['height_cm'],
        "BMI": bmi,
        "Cholesterol": cholesterol_descriptions[values['cholesterol']]
    }

    
    model = load_model()
    model_params = [[
        values['bp_systolic'],
        values['bp_diastolic'],
        values['age'],
        values['cholesterol'],
        bmi
    ]]

    
    prediction = model.predict(model_params)[0]

    
    probabilities = model.predict_proba(model_params)[0]

    return render_template('results.html', prediction=prediction, probabilities=probabilities, input_values=input_values, form_values=values)


def calculate_bmi(height_cm, weight_kg):
    """
    Calculates BMI given height (in kg), and weight (in cm)
    BMI Formula: kg / m^2
    Output is BMI, rounded to one decimal digit
    """

    
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


# Start the server
if __name__ == "__main__":
    print("* Starting Flask server..."
          "please wait until server has fully started")
    # debug=True options allows us to view our changes without restarting the server.
    app.run(host='0.0.0.0', debug=True)
