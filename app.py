
#Importacion de librerias necesarias
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import joblib 
import base64
import matplotlib.cm as cm

#Funcion para resaltar valor mas alto
def resaltar_maximos(row):
    # Obtener los índices de los valores máximos en la fila
    indices_maximos = row[row == row.max()].index
    # Crear una lista de estilos para resaltar los valores máximos
    styles = ['background-color: yellow' if col in indices_maximos else '' for col in row.index]
    return styles

##Funcion predecir resultado   
def predict_result(pel1, pel2):
    
    if pel1 == pel2:
        st.text('Introduce dos peleadores diferentes')
   
    else:
        stats_prediction = pd.read_csv('./df_prediccion_230207.csv')
        cols = stats_prediction.columns[1:]

        stats1 = np.array(stats_prediction[stats_prediction['Name'] == pel1].iloc[:,1:])
        stats2 = np.array(stats_prediction[stats_prediction['Name'] == pel2].iloc[:,1:])

        pred_stats = stats1 - stats2
        df = pd.DataFrame(pred_stats, columns=cols)
        
        scaler = joblib.load('scaler.joblib')
        df.iloc[:, :-5] = scaler.transform(df.iloc[:, :-5])
        
        model = joblib.load('modelo_reg_pred.joblib')
        result = model.predict_proba(df)

        if result[0][0]>result[0][1]:
            st.text(f'El peleador {pel1} ganará con una probabilidad de {round(result[0][0]*100, 2)}%')
            
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f'<div style="width: 300px; height: 100px; background-color: green; display: flex; align-items: center; color: white; font-weight: bold;justify-content: center;">{round(result[0][0]*100, 2)}%</div>', unsafe_allow_html=True)

            with col2:    
                st.markdown(f'<div style="width: 300px; height: 100px; background-color: red; display: flex; align-items: center; color: white; font-weight: bold; justify-content: center;">{round(result[0][1]*100,2)}%</div>', unsafe_allow_html=True)
   

        elif result[0][0]<result[0][1]:
            st.text(f'El peleador {pel2} ganará con una probabilidad de {round(result[0][1]*100, 2)}%')
            
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f'<div style="width: 300px; height: 100px; background-color: red; display: flex; align-items: center; color: white; font-weight: bold;justify-content: center;">{round(result[0][0]*100, 2)}%</div>', unsafe_allow_html=True)

            with col2:    
                st.markdown(f'<div style="width: 300px; height: 100px; background-color: green; display: flex; align-items: center; color: white; font-weight: bold;justify-content: center;">{round(result[0][1]*100, 2)}%</div>', unsafe_allow_html=True)
   
        else:
            st.text('No hay datos suficientes para predecir el resultado')
            
        return

        
##Titulo global del app
col_ini1, col_ini2 = st.columns(2)

with col_ini1:
    col_ini1.empty()
    st.title('Money Fighter')

with col_ini2:    
    st.image('./logo.png', width= 100)


#Menu del streamlit
menu = ['Home', 'Predictions', 'Next event Predictions', 'Top 10 Stats']

choice = st.sidebar.selectbox('Menu', menu)

#carga de los datos
fighter_stats = pd.read_csv('./df_prediccion_230207.csv')
peleadores = fighter_stats['Name'].unique()
next_event = pd.read_csv('./next_event.csv')
txt_descripcion = './description.txt'


if choice == 'Home':

    with open(txt_descripcion, "r") as archivo:
        desc = archivo.read()
        
    st.markdown(desc)

elif choice == 'Predictions':
    st.subheader('Predictions')
    
    col1, col2 = st.columns(2)

    with col1:
        pel1 = st.selectbox('Fighter 1:', peleadores) 
        
    with col2:    
        pel2 = st.selectbox('Fighter 2:', peleadores) 
        
        
    table1 = fighter_stats[fighter_stats['Name'] == pel1].iloc[:,1:]
    table1.rename({table1.index[0]: pel1}, inplace = True)


    table2 = fighter_stats[fighter_stats['Name'] == pel2].iloc[:,1:]
    table2.rename({table2.index[0]: pel2}, inplace = True)

    # Establecer el estilo de la tabla
    st.markdown("""
        <style>
        .dataframe td {
            font-weight: bold;
            color: white;
            background-color: #4c78a8;
            text-align: center;
        }
        .dataframe th {
            font-weight: bold;
            padding: 8px;
            color: white;
            background-color: #4c78a8;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    
    if pel1 != pel2:
        table = pd.concat([table1, table2], axis=0)
        st.dataframe(table.style.set_properties(**{'text-align': 'center'}).background_gradient(cmap='Blues'))



    predict = st.button('Predict result')
    
    if predict:
        #Al hacer click en el boton se ejecuta esta parte del codigo
        predict_result(pel1, pel2)
       
        
                            
elif choice == 'Next event Predictions':
    st.subheader('Next event Predictions')
    st.dataframe(next_event)
    
else:
    st.subheader('Top 10 Stats')
    st.markdown('#### En esta pagina podras visualizar el top 10 peleadores segun cada una de sus estadisticas')
    st.markdown('* Selecciona la estadistica que deseas visualizar')
    stat = st.selectbox('Estadística:', fighter_stats.columns[2:-5])
    
    top_10 = fighter_stats[['Name', stat]].sort_values(by=stat, ascending=False).head(10)
    fig, ax = plt.subplots(figsize= (5,5))
    sns.barplot(data = top_10, x = stat, y = 'Name', orient='h')
    st.pyplot(fig)
    

