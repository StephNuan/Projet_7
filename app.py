import pandas as pd
from flask import Flask, jsonify, request
import joblib
import pickle
import json
import zipfile

app = Flask(__name__)

zip_file_path = 'df_test.zip'
df = 'df_test.csv'

# Charger les données depuis le fichier ZIP lors du démarrage de l'application
with zipfile.ZipFile(zip_file_path, 'r') as zipf:
    with zipf.open(df) as file_in_zip:
        data = pd.read_csv(file_in_zip)

lgbm = pickle.load(open('best_final_prediction.pickle', 'rb'))

@app.route('/')
def index():
    return 'Welcome to my Flask API!'

@app.route('/check_id/', methods=['GET'])
def check_id():
    id_client = int(request.args.get('id_client', default=100028))  # Obtenir l'id_client de la requête

    all_id_client = list(data['SK_ID_CURR'].unique())

    if id_client not in all_id_client:
        return jsonify({})  # Renvoyer une réponse JSON vide
    else:
        check_id = data[data['SK_ID_CURR'] == id_client]
        json_check_id = json.dumps(check_id.to_dict(orient='records'), allow_nan=True)
        return json_check_id

@app.route('/credit/', methods=["GET"])
def credit():
    id_client = int(request.args.get('id_client', default=100028))  # Obtenir l'id_client de la requête
    X = data[data['SK_ID_CURR'] == id_client]
    json_predict = json.dumps(X.to_dict(orient='records'), allow_nan=True)

    ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in data.columns if col not in ignore_features]
    X = X[relevant_features]

    proba = lgbm.predict_proba(X)
    prediction = lgbm.predict(X)
    pred_proba = {
        'prediction': int(prediction[0]),
        'proba': float(proba[0][1])  # Mettez l'indice correct pour obtenir la probabilité de la classe positive
    }
    
    return jsonify(pred_proba)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
