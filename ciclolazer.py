import streamlit as st
import pandas as pd
import pydeck as pdk
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Função para carregar os dados com cache
@st.cache_data
def carregar_dados(caminho_csv, caminho_estacoes):
    try:
        df = pd.read_csv(caminho_csv)
        df_stations = pd.read_csv(caminho_estacoes)
        return df, df_stations
    except FileNotFoundError:
        st.error("Erro: Arquivo CSV não encontrado. Verifique o caminho dos arquivos.")
        return None, None

# Definir o caminho relativo para os arquivos CSV
caminho_csv = os.path.join('datasets', 'df_rides.csv')
caminho_estacoes = os.path.join('datasets', 'df_stations.csv')

# Carregar os DataFrames
df, df_stations = carregar_dados(caminho_csv, caminho_estacoes)

if df is not None and df_stations is not None:
    # Layout e CSS personalizados
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #446f57;
        }
        .resultado-previsao {
            font-size: 24px;
            color: #ffcc00;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Exibe o logo na barra lateral
    imagemlogo = os.path.abspath('ciclolazer.jpg')
    st.sidebar.image(imagemlogo, width=300)

    # Configurar colunas de data e hora
    df['ride_date'] = pd.to_datetime(df['ride_date'])
    df['time_start'] = pd.to_datetime(df['time_start'])
    df['time_start_15min'] = df['time_start'].dt.floor('15T')

    # Criar recursos adicionais
    df['hour_of_day'] = df['time_start'].dt.hour
    df['day_of_week'] = df['time_start'].dt.weekday
    df['user_age'] = (pd.to_datetime('today') - pd.to_datetime(df['user_birthdate'])).dt.days // 365

    # Contar o número de viagens por estação
    station_counts = df['station_start'].value_counts().reset_index()
    station_counts.columns = ['station', 'count']

    # Juntar as informações com as coordenadas das estações
    merged_data = pd.merge(station_counts, df_stations, on='station')

    # Adicionar barra lateral para seleção de estação
    stations_selected = st.sidebar.multiselect(
        "Selecione as estações para visualizar no mapa:",
        options=merged_data['station'].unique(),
        default=merged_data['station'].iloc[:1]  # Padrão: primeira estação
    )

    # Filtrar dados com base na seleção
    if stations_selected:
        filtered_data = merged_data[merged_data['station'].isin(stations_selected)]
    else:
        # Se não houver estação selecionada, mostra todas as estações
        filtered_data = merged_data

    # Previsão de bicicletas disponíveis
    st.sidebar.subheader("")
    selected_station = st.sidebar.selectbox("Escolha uma estação para previsão:", stations_selected if stations_selected else merged_data['station'])
    hour_of_day = st.sidebar.slider("Escolha o horário:", min_value=0, max_value=23, value=12)
    day_of_week = st.sidebar.selectbox(
        "Escolha o dia da semana:",
        ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    )

    # Convertendo o dia da semana para número
    day_of_week_dict = {'Segunda': 0, 'Terça': 1, 'Quarta': 2, 'Quinta': 3, 'Sexta': 4, 'Sábado': 5, 'Domingo': 6}
    day_of_week_num = day_of_week_dict[day_of_week]

    # Criar o modelo KNN
    df_rides_model = df[['user_gender', 'user_age', 'station_start', 'hour_of_day', 'day_of_week']]
    df_rides_model = pd.get_dummies(df_rides_model, drop_first=True)

    # Calcular bike_count
    bike_counts = df.groupby('time_start_15min').size().reset_index(name='bike_count')
    valid_indices = df_rides_model.index.intersection(bike_counts.index)
    X = df_rides_model.loc[valid_indices]
    y = bike_counts.loc[valid_indices, 'bike_count']

    # Normalizar e treinar o modelo
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Previsão
    input_data = pd.DataFrame({
        'user_gender': [0],  # Assumindo '1' para feminino
        'user_age': [25],  # Idade simulada
        'station_start': [selected_station],
        'hour_of_day': [hour_of_day],
        'day_of_week': [day_of_week_num]
    })
    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    predicted_bikes = knn.predict(input_data_scaled)
    previsao_bicicletas = int(predicted_bikes[0] // 490)

    # Exibir o mapa com PyDeck
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=filtered_data['lat'].mean(),
                longitude=filtered_data['lon'].mean(),
                zoom=12,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=filtered_data,
                    get_position="[lon, lat]",
                    get_color="[200, 30, 0, 160]",
                    get_radius=200,
                    pickable=True,
                    tooltip=True
                ),
            ],
        ), use_container_width=True  # Ajusta a largura do mapa para ocupar todo o container
    )

    # Exibir o resultado da previsão abaixo do mapa
    st.markdown(
        f'<div class="resultado-previsao">Previsão: {previsao_bicicletas} bicicletas disponíveis.</div>',
        unsafe_allow_html=True
    )
else:
    st.error("Os dados não foram carregados corretamente.")
