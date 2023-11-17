import zipfile 
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import math
from urllib.request import urlopen
import json
import requests
import plotly.graph_objects as go 
import shap
import joblib
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
    
zip_file_path = 'df_test.zip'  
df = 'df_test.csv'  

# Créez un objet ZipFile pour ouvrir le fichier ZIP en mode lecture
with zipfile.ZipFile(zip_file_path, 'r') as zipf:
    with zipf.open(df) as file_in_zip:
        data = pd.read_csv(file_in_zip) 
                
data_test = pd.read_csv('application_test.csv')
        
zip_file_path2 = 'application_train.zip'  
df2 = 'application_train.csv'  

# Créez un objet ZipFile pour ouvrir le fichier ZIP en mode lecture
with zipfile.ZipFile(zip_file_path2, 'r') as zipf:
    with zipf.open(df2) as file_in_zip:
        data_train = pd.read_csv(file_in_zip) 
        
        #description des features
        
description = pd.read_csv('HomeCredit_columns_description.csv', 
                                      usecols=['Row', 'Description'], \
                                  index_col=0, encoding='unicode_escape')

@st.cache_data

def get_client_info(data, id_client):
    client_info = data[data['SK_ID_CURR']==int(id_client)]
    return client_info

headers_request = {"Content-Type": "application/json"}

@st.cache_data
def request_feature_definition():
    url_request = base_url + "get_features_definition"
    response = requests.request( method='POST', headers=headers_request, url=url_request)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return response.json()["feature_definition"]

@st.cache_data
def request_shap_waterfall_chart(client_id, feat_number):
    url_request = base_url + "get_shap_waterfall_chart"
    data_json = {"client_id" : client_id, "feat_number" : feat_number}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    base64_image = response.json()["base64_image"]
    response_json = {
        "base64_image": base64_image,
    }
    return json.dumps(response_json) 

@st.cache_data
def request_shap_waterfall_chart_global(feat_number):
    url_request = base_url + "get_shap_waterfall_chart_global"
    data_json = {"feat_number" : feat_number}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    base64_image = response.json()["base64_image"]
    response_json = {
        "base64_image": base64_image,
    }
    return json.dumps(response_json)


@st.cache_data
def request_comparison_chart(client_id, feat_name):
    url_request = base_url + "get_comparison_graph"
    data_json = {"client_id" : client_id, "feat_name" : feat_name}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return json.dumps({"base64_image": response.json()["base64_image"]})

@st.cache_data()

def plot_distribution(applicationDF,feature, client_feature_val, title):

    if (not (math.isnan(client_feature_val))):
        fig = plt.figure(figsize = (10, 4))

        t0 = applicationDF.loc[applicationDF['TARGET'] == 0]
        t1 = applicationDF.loc[applicationDF['TARGET'] == 1]

        if (feature == "DAYS_BIRTH"):
            sns.kdeplot((t0[feature]/-365).dropna(), label = 'Non défaillant', color='g')
            sns.kdeplot((t1[feature]/-365).dropna(), label = 'Défaillant', color='r')
            plt.axvline(float(client_feature_val/-365),  color="blue", linestyle='--', label = 'Position Client')

        elif (feature == "DAYS_EMPLOYED"):
            sns.kdeplot((t0[feature]/365).dropna(), label = 'Non défaillant', color='g')
            sns.kdeplot((t1[feature]/365).dropna(), label = 'Défaillant', color='r')    
            plt.axvline(float(client_feature_val/365), color="blue", linestyle='--', label = 'Position Client')

        else:    
            sns.kdeplot(t0[feature].dropna(), label = 'Non défaillant', color='g')
            sns.kdeplot(t1[feature].dropna(), label = 'Défaillant', color='r')
            plt.axvline(float(client_feature_val), color="blue",linestyle='--', label = 'Position Client')


        plt.title(title, fontsize='20', fontweight='bold')

        plt.legend()
        plt.show()  
        st.pyplot(fig)
    else:
        st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")
        
@st.cache_data()

def univariate_categorical(applicationDF,feature,client_feature_val,titre,ylog=False,label_rotation=False,
                               horizontal_layout=True):
        if (client_feature_val.iloc[0] != np.nan):
            applicationDF_filtered = applicationDF[applicationDF[feature] != 'XNA']


            temp = applicationDF_filtered[feature].value_counts()
            df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

            categories = applicationDF_filtered[feature].unique()
            categories = list(categories)

            # Calculate the percentage of target=1 per category value
            
            cat_perc = applicationDF_filtered[[feature,'TARGET']].groupby([feature],as_index=False).mean()
            cat_perc["TARGET"] = cat_perc["TARGET"]*100
            cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

            if(horizontal_layout):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))

            # 1. Subplot 1: Count plot of categorical column
            s = sns.countplot(ax=ax1, 
                            x = feature, 
                            data=applicationDF_filtered,
                            hue ="TARGET",
                            order=cat_perc[feature],
                            palette=['g','r'])

            pos1 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])

            # Define common styling
            ax1.set(ylabel = "Nombre de clients")
            ax1.set_title(titre, fontdict={'fontsize' : 15, 'fontweight' : 'bold'})   
            ax1.axvline(int(pos1), color="blue", linestyle='--', label = 'Position Client')
            ax1.legend(['Position Client','Non défaillant','Défaillant' ])

            # If the plot is not readable, use the log scale.
            if ylog:
                ax1.set_yscale('log')
                ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 15,'fontweight' : 'bold'})   
            if(label_rotation):
                s.set_xticklabels(s.get_xticklabels(),rotation=90)

            # 2. Subplot 2: Percentage of defaulters within the categorical column
            s = sns.barplot(ax=ax2, 
                            x = feature, 
                            y='TARGET', 
                            order=cat_perc[feature], 
                            data=cat_perc,
                            palette='Set2')

            pos2 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])

            if(label_rotation):
                s.set_xticklabels(s.get_xticklabels(),rotation=90)
            plt.ylabel('Pourcentage de défaillants [%]', fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            ax2.set_title(titre+" (% Défaillants)",fontdict={'fontsize' : 15, 'fontweight' : 'bold'})
            #ax2.axvline(int(pos2), color="blue", linestyle='--', label = 'Position Client')
            ax2.legend()
            plt.show()
            st.pyplot(fig)
        else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")
            
            
#Chargement des données    
ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
relevant_features = [col for col in data if col not in ignore_features]

#Chargement du modèle
lgbm = pickle.load(open('Modele.pickle', 'rb'))

##################################################
#                     IMAGES                     #
##################################################
# Logo
logo =  Image.open('image.png') 


 #######################################
    # SIDEBAR
#######################################

SHAP_GENERAL = "global_feature_importance.png"
st.sidebar.image(logo, width=150,
                 use_column_width='always')

with st.sidebar:
    #st.header(" Prêt à dépenser")

    st.write("## ID client")
    id_list = data["SK_ID_CURR"].values
    id_client = st.selectbox(
            "Sélectionner l'identifiant du client", id_list)

    st.write("## Actions à effectuer")
    show_credit_decision = st.checkbox("Afficher la décision de crédit")
    show_client_details = st.checkbox("Afficher les informations du client")
    show_client_comparison = st.checkbox("Comparer aux autres clients")
    shap_general = st.checkbox("Afficher la feature importance globale")
    if(st.checkbox("Aide description des features")):
        list_features = description.index.to_list()
        list_features = list(dict.fromkeys(list_features))
        feature = st.selectbox('Sélectionner une variable',  sorted(list_features))
        desc = description['Description'].loc[description.index == feature][:1]
        st.markdown('**{}**'.format(desc.iloc[0]))
        
 #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

#Titre principal

html_temp = """
    <h1 style="color: #0000FF; text-align:center; font-size: 40px; font-weight: bold;">Tableau de bord du rapport de crédit</h1>
"""

st.markdown(html_temp, unsafe_allow_html=True)


#Afficher l'ID Client sélectionné
st.write("Identifiant du client (ID_client):", id_client)
if (int(id_client) in id_list):
    client_info = data_test[data_test['SK_ID_CURR']==int(id_client)]

       
      #-------------------------------------------------------
        # Afficher la décision de crédit
        #-------------------------------------------------------
if show_credit_decision:
    base_url = "http://127.0.0.1:5000/credit/" + str(id_client)

    # Charger le modèle et effectuer les calculs nécessaires
    lgbm = pickle.load(open('Modele.pickle', 'rb'))
    classe_reelle = data[data['SK_ID_CURR'] == id_client]
    ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in classe_reelle.columns if col not in ignore_features]
    classe_reelle = classe_reelle[relevant_features]

    proba = lgbm.predict_proba(classe_reelle)
    prediction = lgbm.predict(classe_reelle)
    pred_proba = {
        'prediction': int(prediction),
        'proba': float(proba[0][0])
    }

    classe_predite = pred_proba['prediction']
    if classe_predite == 1:
        etat = 'Client à risque'
    else:
        etat = 'Client peu risqué'
    proba = 1 - pred_proba['proba']

    # Affichage de la prédiction
    prediction = pred_proba['proba']
    classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
    chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba * 100)) + '%** de risque de défaut (classe réelle : ' + str(classe_reelle) + ')'

    # Calcul du score du client
    client_score = round(proba * 100, 2)

    # Affichage de la décision
    left_column, right_column = st.columns((1, 2))
    left_column.markdown('Risque de défaut: **{}%**'.format(str(client_score)))

    if classe_predite == 1:
        left_column.markdown('Décision: <span style="color:red">**{}**</span>'.format(etat), unsafe_allow_html=True)
    else:
        left_column.markdown('Décision: <span style="color:green">**{}**</span>'.format(etat), unsafe_allow_html=True)


personal_info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "POSSESSION VEHICULE",
            'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'NAME_HOUSING_TYPE':'TYPE DE LOGEMENT', 
            'OCCUPATION_TYPE': "EMPLOI",
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "EXT_SOURCE_1",
            'EXT_SOURCE_2': "EXT_SOURCE_2",
            'EXT_SOURCE_3': "EXT_SOURCE_3",

        }

default_list= ["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","REVENUS","MONTANT CREDIT"]
numerical_features = [ 'DAYS_BIRTH' ,'CNT_CHILDREN','DAYS_EMPLOYED','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1', 
                         'EXT_SOURCE_2 ' , 'EXT_SOURCE_3' ]

rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

if (show_client_details):
        st.header('‍ Informations relatives au client')

        with st.spinner('Chargement des informations relatives au client...'):
            personal_info_df = client_info[list(personal_info_cols.keys())]
               
            personal_info_df.rename(columns=personal_info_cols, inplace=True)

            personal_info_df["AGE"] = int(round(personal_info_df["AGE"]/365*(-1)))
            personal_info_df["NB ANNEES EMPLOI"] =             int(round(personal_info_df["NB ANNEES EMPLOI"]/365*(-1)))


            filtered = st.multiselect("Choisir les informations à afficher",options=list(personal_info_df.columns),
                                      default=list(default_list))
            df_info = personal_info_df[filtered] 
            df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
            df_info = df_info.set_index('SK_ID_CURR')

            st.table(df_info.astype(str).T)
            show_all_info = st.checkbox("Afficher toutes les informations (dataframe brute)")
            if (show_all_info):
                st.dataframe(client_info)

#-------------------------------------------------------
        # Comparer le client sélectionné à d'autres clients
        #-------------------------------------------------------
#

if (show_client_comparison):
        st.header('‍ Comparaison aux autres clients')
      
        with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
            var = st.selectbox("Sélectionner une variable",list(personal_info_cols.values()))
            feature = list(personal_info_cols.keys())[list(personal_info_cols.values()).index(var)]
            
            if (feature in numerical_features):
                
                plot_distribution(data_train, feature, client_info[feature], var)  
                
            elif (feature in rotate_label):
                
                univariate_categorical(data_train, feature,client_info[feature], var, False, True)
            elif (feature in horizontal_layout):
               
                
                univariate_categorical(data_train, feature,client_info[feature], var, False, True, True)
                
            else:
                
                univariate_categorical(data_train, feature, client_info[feature], var)
                                    
       
            
#-------------------------------------------------------
        # Afficher la feature importance globale
        #-------------------------------------------------------
if (shap_general):
        # original_title = '<p style="font-size: 20px;text-align: center;"> <u>Top 10 des variables qui contribuent à la prédiction du modèle </u> </p>'
        # st.markdown(original_title, unsafe_allow_html=True)
        # feature_imp = pd.DataFrame(sorted(zip(lgbm.booster_.feature_importance(importance_type='gain'), data.columns)), columns=['Value','Feature'])

        # fig, ax = plt.subplots(figsize=(10, 5))
        # sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(10))
        # st.pyplot(fig)  

        st.write('## <p style="font-size:20px;">Interprétation globale: impact sur la prédiction</p>', unsafe_allow_html=True)
        shap.initjs()   

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(lgbm)
        shap_values = shap.TreeExplainer(lgbm.booster_).shap_values(data.iloc[:10000, 1:])
        shap.summary_plot(shap_values, features=data.drop(labels="SK_ID_CURR", axis=1).columns, plot_type ="bar", max_display=10, color_bar=False, plot_size=(10, 10))            
        st.pyplot(fig)  

        st.write('## <p style="font-size:20px;">Interprétation locale: impact sur la prédiction de la classe "1" du modèle</p>', unsafe_allow_html=True)                            
        shap.initjs()   

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(lgbm)
        shap_values = shap.TreeExplainer(lgbm.booster_).shap_values(data.iloc[:10000, 1:])
        shap.summary_plot(shap_values[1], features=data.drop(labels="SK_ID_CURR", axis=1).columns, plot_type ="bar", max_display=10, color_bar=False, plot_size=(10, 10))            
        st.pyplot(fig)  


