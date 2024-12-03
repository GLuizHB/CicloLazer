import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Carregar os dados
@st.cache_data
def carregar_dados(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)
        return df
    except FileNotFoundError:
        st.error("Erro: Arquivo CSV não encontrado. Verifique o caminho dos arquivos.")
        return None

# Definir o caminho para os arquivos CSV
caminho_csv = 'datasets/df_rides.csv'
df = carregar_dados(caminho_csv)

if df is not None:
    # Certificar-se de que as colunas de data estão no formato correto
    df['ride_date'] = pd.to_datetime(df['ride_date'])
    df['time_start'] = pd.to_datetime(df['time_start'])

    # Criar a coluna 'time_start_15min' arredondando para o intervalo de 15 minutos
    df['time_start_15min'] = df['time_start'].dt.floor('15T')

    # Criar variáveis adicionais para o modelo
    df['hour_of_day'] = df['time_start'].dt.hour
    df['day_of_week'] = df['time_start'].dt.weekday
    df['user_age'] = (pd.to_datetime('today') - pd.to_datetime(df['user_birthdate'])).dt.days // 365

    # Seleção de variáveis para o modelo
    df_rides_model = df[['user_gender', 'user_age', 'station_start', 'hour_of_day', 'day_of_week']]
    df_rides_model = pd.get_dummies(df_rides_model, drop_first=True)

    # Definir o alvo (bike_count) e normalizar as variáveis
    bike_counts = df.groupby('time_start_15min').size().reset_index(name='bike_count')
    valid_indices = df_rides_model.index.intersection(bike_counts.index)
    X = df_rides_model.loc[valid_indices]
    y = bike_counts.loc[valid_indices, 'bike_count']

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Divisão dos dados para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Criar e treinar o modelo KNN
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Previsões
    y_pred = knn.predict(X_test)

    # Avaliação de métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    accuracy = accuracy_score(np.round(y_test), np.round(y_pred))

    # Cálculo da curva ROC (apenas para demonstração, normalmente usada em classificação)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Exibir gráficos no Streamlit
    st.title('Análise do Modelo KNN')

    # Curva ROC
    st.subheader('Curva ROC')
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_title('Curva ROC')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    # Gráfico de Dispersão (scatter plot)
    st.subheader('Gráfico de Dispersão (Real vs. Previsto)')
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='green', alpha=0.6)
    ax.set_title('Real vs. Previsto')
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Previsto')
    st.pyplot(fig)

    # Métricas de Avaliação
    st.subheader('Métricas de Avaliação do Modelo')
    st.write(f'R²: {r2:.2f}')
    st.write(f'RMSE (Root Mean Squared Error): {rmse:.2f}')
    st.write(f'Acurácia: {accuracy:.2f}')

    # Visualizando o erro quadrático médio (MSE)
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')

else:
    st.error("Erro ao carregar os dados.")
