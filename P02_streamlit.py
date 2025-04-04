import streamlit as st
import os
import glob
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

# Asegurar que NLTK use la carpeta local si se sube a Streamlit Cloud
nltk.data.path.append('./nltk_data')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Configuración Inicial ---
st.set_page_config(page_title="Análisis de Letras de Canciones", layout="wide")
DATA_DIR = "Data songs"  # Carpeta donde están las subcarpetas "80's", "Rock", etc.

# --- Funciones auxiliares ---
def cargar_archivos(carpeta):
    ruta = os.path.join(DATA_DIR, carpeta, "*.txt")
    archivos = glob.glob(ruta)
    nombres = [os.path.basename(archivo) for archivo in archivos]
    return dict(zip(nombres, archivos))

# Selección de idioma
idioma = st.selectbox("Selecciona el idioma del análisis", ["español", "inglés"])
stop_words = set(stopwords.words('spanish' if idioma == "español" else 'english'))

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = nltk.word_tokenize(texto)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def obtener_ngramas(tokens, n):
    return list(ngrams(tokens, n))

def mostrar_wordcloud(tokens):
    texto = ' '.join(tokens)
    wc = WordCloud(width=800, height=400, background_color='white').generate(texto)
    st.image(wc.to_array())

def mostrar_distribucion(tokens):
    frec = Counter(tokens)
    comunes = frec.most_common(20)
    palabras, cantidades = zip(*comunes)
    plt.figure(figsize=(10,5))
    plt.bar(palabras, cantidades)
    plt.xticks(rotation=45)
    st.pyplot(plt)

# --- Interfaz Streamlit ---
st.title("🎵 Análisis de Letras de Canciones")

# Selección de carpeta
carpetas = [nombre for nombre in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, nombre))]
carpeta_seleccionada = st.selectbox("Selecciona una carpeta (género/época)", carpetas)

# Cargar archivos de esa carpeta
archivos = cargar_archivos(carpeta_seleccionada)
canciones_seleccionadas = st.multiselect("Selecciona una o más canciones", list(archivos.keys()))

if canciones_seleccionadas:
    for nombre in canciones_seleccionadas:
        st.header(f"📄 {nombre}")
        with open(archivos[nombre], 'r', encoding='utf-8') as f:
            texto = f.read()
        
        tokens = limpiar_texto(texto)

        # --- Métricas ---
        total_palabras = len(tokens)
        promedio = total_palabras / max(1, len(nltk.sent_tokenize(texto)))
        palabras_unicas = set(tokens)
        frec = Counter(tokens)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Palabras", total_palabras)
        col2.metric("Promedio por oración", f"{promedio:.2f}")
        col3.metric("Palabras Únicas", len(palabras_unicas))

        # --- Palabras Comunes ---
        st.subheader("🔝 Palabras más comunes")
        comunes = frec.most_common(10)
        for palabra, freq in comunes:
            st.write(f"{palabra}: {freq}")

        # --- WordCloud ---
        st.subheader("☁️ WordCloud")
        mostrar_wordcloud(tokens)

        # --- Distribución ---
        st.subheader("📊 Distribución de Vocabulario")
        mostrar_distribucion(tokens)

        # --- N-gramas ---
        st.subheader("📎 N-Gramas")
        for n in [2, 3, 4]:
            ngramas = obtener_ngramas(tokens, n)
            frec_ng = Counter(ngramas).most_common(5)
            st.markdown(f"**Top {n}-gramas:**")
            for ng, freq in frec_ng:
                st.write(f"{' '.join(ng)}: {freq}")
