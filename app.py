import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsão de Chuva - Epagri", layout="wide")

st.title("🌦️ Sistema de Análise e Previsão de Chuva")
st.markdown("""
Este sistema utiliza dados meteorológicos históricos para prever a ocorrência de chuva no dia seguinte
utilizando um algoritmo de Machine Learning (**Random Forest**).
""")

# --- CONSTANTES E FUNÇÕES (Adaptadas do seu Notebook) ---
COL_TEMP_INST = "Temperatura do Ar Instantânea (°C)"
COL_TEMP_MIN = "Temperatura Mínima (°C)"
COL_TEMP_MAX = "Temperatura Máxima (°C)"
COL_VENTO_VEL = "Velocidade Média do Vento (m/s)"
COL_PRESSAO = "Pressão Atmosférica (mB)"
COL_UMIDADE = "Umidade Relativa do Ar Média (%)"
COL_PRECIP = "Precipitação (mm)"

@st.cache_data
def carregar_dados(uploaded_file):
    """Carrega e processa os dados a partir de um arquivo enviado pelo usuário."""
    if uploaded_file is None:
        return None

    try:
        # Ler CSV (pulando a primeira linha de metadados da Epagri)
        df = pd.read_csv(uploaded_file, skiprows=1)
        df = df.drop(columns=["Unnamed: 11"], errors='ignore') # Tratamento de erro se coluna não existir

        # Renomear colunas
        novas_colunas = ["Código", "Data", COL_TEMP_INST, COL_TEMP_MIN, COL_TEMP_MAX,
                         COL_VENTO_VEL, "Dir Vento", "Vento Max", COL_PRESSAO, COL_UMIDADE, COL_PRECIP]

        # Ajustar se o número de colunas bater (segurança)
        if len(df.columns) == len(novas_colunas):
            df.columns = novas_colunas

        # Tratamento de Datas
        df["Data"] = pd.to_datetime(df["Data"], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df = df.dropna(subset=["Data"]).set_index("Data")

        # Converter numéricos e tratar 9999
        cols_numericas = [COL_TEMP_MAX, COL_TEMP_MIN, COL_UMIDADE, COL_PRESSAO, COL_VENTO_VEL, COL_PRECIP]
        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                limit = 1825 if col == COL_PRECIP else 9999
                df[col] = df[col].apply(lambda x: x if x < limit else np.nan)

        # Interpolação e Agregação Diária
        df = df.interpolate(method='linear').ffill().bfill()

        df_diario = df.resample('D').agg({
            COL_TEMP_MIN: 'min',
            COL_TEMP_MAX: 'max',
            COL_VENTO_VEL: 'mean',
            COL_PRESSAO: 'mean',
            COL_UMIDADE: 'mean',
            COL_PRECIP: 'sum'
        }).dropna()

        return df_diario

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None

def preparar_features(df):
    df_model = df.copy()
    # Target
    df_model["Choveu_Hoje"] = (df_model[COL_PRECIP] > 1.0).astype(int)
    df_model["Target"] = df_model["Choveu_Hoje"].shift(-1) # Prever amanhã

    # Features & Lags
    df_model["TempMax_Ontem"] = df_model[COL_TEMP_MAX].shift(1)
    df_model["Umid_Ontem"] = df_model[COL_UMIDADE].shift(1)
    df_model["Pressao_Ontem"] = df_model[COL_PRESSAO].shift(1)

    features = [COL_TEMP_MAX, COL_TEMP_MIN, COL_UMIDADE, COL_PRECIP, COL_PRESSAO, COL_VENTO_VEL,
                "TempMax_Ontem", "Umid_Ontem", "Pressao_Ontem"]

    df_model = df_model.dropna(subset=features + ["Target"])
    return df_model[features], df_model["Target"], features

@st.cache_resource
def treinar_modelo(X, y):
    # Divisão temporal (sem shuffle)
    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]

    model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

# --- INTERFACE PRINCIPAL ---

st.sidebar.header("1. Carregar Dados")
arquivo = st.sidebar.file_uploader("Faça upload do CSV da Epagri", type="csv")

if arquivo:
    df = carregar_dados(arquivo)

    if df is not None:
        st.success("Dados carregados com sucesso!")

        # Abas para organizar a visualização
        tab1, tab2 = st.tabs(["📊 Análise Exploratória", "🔮 Previsão Interativa"])

        # --- ABA 1: EDA ---
        with tab1:
            st.subheader("Visualização dos Dados Históricos")
            st.dataframe(df.tail())

            col1, col2 = st.columns(2)
            with col1:
                fig_temp = px.line(df, y=[COL_TEMP_MAX, COL_TEMP_MIN], title="Temperaturas Máxima e Mínima", color_discrete_sequence=['#d9373c', '#4ba0ec'])
                st.plotly_chart(fig_temp, use_container_width=True)

                fig_umid = px.line(df, y=COL_UMIDADE, title="Umidade Relativa", color_discrete_sequence=['#51A2FF'])
                st.plotly_chart(fig_umid, use_container_width=True)

            with col2:
                fig_prec = px.bar(df, y=COL_PRECIP, title="Precipitação Diária (Chuva)", color_discrete_sequence=['#636efa'])
                st.plotly_chart(fig_prec, use_container_width=True)

                fig_press = px.line(df, y=COL_PRESSAO, title="Pressão Atmosférica", color_discrete_sequence=['#FF8904'])
                st.plotly_chart(fig_press, use_container_width=True)

        # --- ABA 2: PREVISÃO ---
        with tab2:
            st.subheader("Simulador de Previsão de Chuva (Para Amanhã)")

            # Treinar modelo em tempo real
            X, y, feature_names = preparar_features(df)
            modelo, cols = treinar_modelo(X, y)

            st.info(f"Modelo treinado com {len(X)} registros históricos.")

            # Formuário de entrada
            col_input1, col_input2, col_input3 = st.columns(3)

            # Valores padrão baseados na média do dataset
            mean_vals = X.mean()

            with col_input1:
                st.markdown("#### 📅 Dados de Hoje")
                t_max = st.number_input("Temp. Máxima (°C)", value=float(mean_vals[COL_TEMP_MAX]))
                t_min = st.number_input("Temp. Mínima (°C)", value=float(mean_vals[COL_TEMP_MIN]))
                umid = st.slider("Umidade Média (%)", 0.0, 100.0, float(mean_vals[COL_UMIDADE]))
                precip = st.number_input("Choveu quanto hoje? (mm)", value=0.0)

            with col_input2:
                st.markdown("#### 💨 Atmosfera Hoje")
                press = st.number_input("Pressão Atm (mB)", value=float(mean_vals[COL_PRESSAO]))
                vento = st.number_input("Vel. Vento (m/s)", value=float(mean_vals[COL_VENTO_VEL]))

            with col_input3:
                st.markdown("#### ⏮️ Dados de Ontem (Lag)")
                t_max_ontem = st.number_input("Temp. Máx. Ontem", value=float(mean_vals["TempMax_Ontem"]))
                umid_ontem = st.slider("Umidade Ontem (%)", 0.0, 100.0, float(mean_vals["Umid_Ontem"]))
                press_ontem = st.number_input("Pressão Ontem", value=float(mean_vals["Pressao_Ontem"]))

            if st.button("Prever Tempo para Amanhã"):
                # Montar array na ordem correta das features
                entrada = pd.DataFrame([[
                    t_max, t_min, umid, precip, press, vento,
                    t_max_ontem, umid_ontem, press_ontem
                ]], columns=cols)

                # Predição
                predicao = modelo.predict(entrada)[0]
                proba = modelo.predict_proba(entrada)[0]

                st.divider()
                if predicao == 1:
                    st.error(f"🌧️ ALERTA: Alta probabilidade de CHUVA amanhã! ({proba[1]*100:.1f}%)")
                else:
                    st.success(f"☀️ Probabilidade de Tempo BOM amanhã. (Chance de chuva: {proba[1]*100:.1f}%)")

else:
    st.info("Por favor, faça o upload do arquivo CSV (dados_meteorologicos_....csv) na barra lateral para iniciar.")
