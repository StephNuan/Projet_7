import pandas as pd
from flask import Flask, jsonify
import joblib
import pickle
import streamlit as st 
from flask import request
import json

app = Flask(__name__)


#PATH = ''

#df = pd.read_csv(PATH+'test_df_2.csv')
df = pd.read_csv('df_test.csv', encoding='utf-8')
#print('df shape = ', df.shape)
print('df shape = ', df.shape)


#Chargement du modèle
#load_clf = joblib.load(PATH+r"trained_model_sample_.joblib")
lgbm = pickle.load(open('Modele', 'rb'))


#Premiers pas sur l'API
@app.route('/')
def index():
    return 'Welcome to my Flask API!'

#id_list = df["SK_ID_CURR"].values
#id_client = st.selectbox("Sélectionner l'identifiant du client", id_list)
#id_client = 100001

# Définir une route pour la validation de l'id_client
@app.route('/check_id/', methods=['GET'])#, methods=['POST']
def check_id(id_client= 100028):
    data_id = pd.read_csv('df_test_imputed.csv', encoding ='utf-8')
    all_id_client = list(data_id['SK_ID_CURR'].unique())
    
    # Vérifier si l'ID client est présent dans la liste des ID client du jeu de données
    id_client = int(id_client)
    if id_client not in all_id_client:

        return  jsonify()  # Renvoyer une réponse JSON vide
    else:
        # Sélectionner les données correspondantes à l'ID client
        check_id = data_id[data_id['SK_ID_CURR'] == id_client] # qui contient uniquement les colonnes à correspond à l'ID client spécifié
                
        # Transformer le dataset en dictionnaire
        # Convertir le dictionnaire en JSON
        json_check_id = json.dumps(check_id.to_dict(orient='records'), allow_nan=True)

        # Renvoyer les données clients
        return json_check_id

@app.route('/credit/', methods=["GET"])
def credit(id_client= 100028):   
    #id_client = 100028
    #id_client = df[df['SK_ID_CURR']]
# # Récupérer l'ID du client à partir des paramètres de la requête
#     id_client = request.args.get('id_client')
#     print('id client = ', id_client)

# Récupération des données du client en question
    ID = int(id_client)
    X = df[df['SK_ID_CURR'] == ID]
    json_predict = json.dumps(X.to_dict(orient='records'), allow_nan=True)

#Isolement des features non utilisées
    ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df.columns if col not in ignore_features]
    X = X[relevant_features]
    print('X shape = ', X.shape)

# Prédiction
    proba = lgbm.predict_proba(X)
    prediction = lgbm.predict(X)
    pred_proba = {
          'prediction': int(prediction),
          'proba': float(proba[0][0])
      }
    #pred_proba = json.dumps(pred_proba)
    #json_predict = json.dumps(pred_proba)
    print('Nouvelle Prédiction : \n', pred_proba)

    return jsonify(pred_proba)
    #return json_predict

# def main():
#     credit(id_client= 100001)

# Lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)