from flask import Flask, request, abort
import joblib
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

@app.route("/predict")
def predict():
  nik = request.args.get("nik")
  phone = request.args.get("phone")

  if (nik == None or phone == None):
    abort(400, "Bad Request")

  feature1 = int(nik)
  feature2 = int(phone)
  
  try:
    model = joblib.load("./trained_model.joblib")
  except Exception as e:
    print(e)
    abort(500)

  raw_data = pd.read_csv("all_data.csv")
  target_variable = "fraud_v2_status"
  try:
      input_data = raw_data[(raw_data['ktp']==feature1) & (raw_data['phone']==feature2)]
      name  = input_data.name.iloc[0]
      phone = input_data.phone.iloc[0]
      ktp   = input_data.ktp.iloc[0]
      input_data = input_data[[col for col in raw_data.columns if (col != target_variable and col not in ['name','phone','ktp'])]]
  except ValueError:
      abort(404, "Not Found")

  # Make predictions using the loaded model
  predictions = model.predict(input_data.values)
  predictions_proba = np.max(model.predict_proba(input_data.values),axis=1)
  prediction_result = 'Non-Fraud' if predictions[0] == 0 else 'Fraud'

  if prediction_result == 'Fraud' and np.round(1-predictions_proba[0],2)*100 >= 0 and np.round(1-predictions_proba[0],2)*100 <= 20:
      grade = 'E'
  elif prediction_result == 'Fraud' and np.round(1-predictions_proba[0],2)*100 >= 21 and np.round(1-predictions_proba[0],2)*100 <= 40:
      grade = 'D'
  elif prediction_result == 'Fraud' and np.round(1-predictions_proba[0],2)*100 >= 41 and np.round(1-predictions_proba[0],2)*100 <= 50:
      grade = 'C'
  elif prediction_result == 'Non-Fraud' and np.round(predictions_proba[0],2)*100 >= 51 and np.round(predictions_proba[0],2)*100 <= 60:
      grade = 'C'
  elif prediction_result == 'Non-Fraud' and np.round(predictions_proba[0],2)*100 >= 61 and np.round(predictions_proba[0],2)*100 <= 80:
      grade = 'B'
  elif prediction_result == 'Non-Fraud' and np.round(predictions_proba[0],2)*100 >= 81 and np.round(predictions_proba[0],2)*100 <= 100:
      grade = 'A'

  if prediction_result == 'Fraud':
      score = np.round(1-predictions_proba[0],2)*100
  else:
      score = np.round(predictions_proba[0],2)*100

  feature_contribution = np.array(input_data.values) * model.coef_
  feature_names = input_data.columns
  contributions_list = list(zip(feature_names, feature_contribution[0]))
  sorted_contributions = sorted(contributions_list, key=lambda x: abs(x[1]), reverse=True)

  if prediction_result == 'Fraud':
      fraud_indicator = [(key, value) for key, value in sorted_contributions if value < 0]
  elif prediction_result == 'Non-Fraud':
      fraud_indicator = [(key, value) for key, value in sorted_contributions if value > 0]

  result = {
      'name': name,
      'phone': phone,
      'ktp': ktp,
      'prediction_result': prediction_result,
      'grade': grade,
      'score': score,
      'fraud_indicator': fraud_indicator
  }

  return result