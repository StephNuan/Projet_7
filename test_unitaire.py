#import unittest
import pandas as pd
import requests
import pytest


# TEST NOMBRE DE COLONNES ET LIGNES

# class TestUnitaire(unittest.TestCase):

#     def test_dataframe_size(self):    
#         app_train_df = pd.read_csv('application_train.csv')
#         num_rows, num_cols = app_train_df.shape
#         self.assertEqual(num_rows, 307511)  
#         self.assertEqual(num_cols, 122) 

# TEST CONNEXION API
# URL de l'API
api_url = "http://127.0.0.1:5000/credit/"

# ID client de test
client_id = 100028

def test_api_connection():
    # Envoi d'une requête GET à l'API
    response = requests.get(api_url, params={"id_client": client_id})

    # Vérification du code de statut HTTP de la réponse
    assert response.status_code == 200, f"La requête a échoué avec le code de statut {response.status_code}"

    # Vérification du contenu de la réponse (vous pouvez adapter cela en fonction de la réponse réelle de votre API)
    response_data = response.json()
    assert "prediction" in response_data, "La réponse ne contient pas la clé 'prediction'"
    assert "proba" in response_data, "La réponse ne contient pas la clé 'proba'"

def test_dataframe_shape():
    app_train_df = pd.read_csv('application_train.csv')
    assert app_train_df.shape == (307511, 122)

def test_dataframe_type():
    app_train_df = pd.read_csv('application_train.csv')
    assert isinstance(app_train_df, pd.DataFrame), "app_train_df n'est pas un DataFrame."

if __name__ == "__main__":
    pytest.main([__file__])


