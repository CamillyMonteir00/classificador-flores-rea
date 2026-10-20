
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Classificador de Flores",
    page_icon="🌸",
    layout="centered"
)

# Categorias na mesma ordem usada durante o treinamento
classes = [
    "Bellis perennis",
    "Chrysanthemum zawadzkii",
    "Dahlia coccinea",
    "Helianthus annuus",
    "Hibiscus syriacus",
    "Iris pseudacorus",
    "Lavandula stoechas",
    "Lilium lancifolium",
    "Magnolia grandiflora",
    "Tulipa sylvestris"
]

# Carregar o modelo sem treiná-lo novamente
@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model(
        "classificador_flores.keras"
    )

st.title("🌸 Classificador de Flores")
st.write(
    "Envie uma foto e descubra qual das 10 categorias "
    "de flores o modelo considera mais provável."
)

st.info(
    "Projeto educacional de Machine Learning. "
    "As previsões podem estar incorretas."
)

try:
    modelo = carregar_modelo()

    arquivo = st.file_uploader(
        "Escolha uma imagem de flor",
        type=["jpg", "jpeg", "png"]
    )

    if arquivo is not None:
        imagem = Image.open(arquivo).convert("RGB")

        st.image(
            imagem,
            caption="Imagem enviada",
            use_container_width=True
        )

        if st.button("🔎 Identificar flor"):
            imagem_redimensionada = imagem.resize((128, 128))

            entrada = np.array(
                imagem_redimensionada,
                dtype=np.float32
            )
            entrada = np.expand_dims(entrada, axis=0)

            previsoes = modelo.predict(entrada, verbose=0)[0]
            indices = np.argsort(previsoes)[::-1][:3]

            maior = float(previsoes[indices[0]])
            segunda = float(previsoes[indices[1]])

            st.subheader("Principais previsões")

            for indice in indices:
                st.write(
                    f"**{classes[indice]}** — "
                    f"{previsoes[indice] * 100:.2f}%"
                )
                st.progress(float(previsoes[indice]))

            st.subheader("Resultado da análise")

            if maior < 0.60:
                st.warning(
                    "🤔 Não tenho confiança suficiente para "
                    "identificar esta flor. Ela pode não pertencer "
                    "às categorias que aprendi."
                )

            elif maior - segunda < 0.15:
                st.warning(
                    "⚠️ Estou em dúvida entre espécies diferentes. "
                    "A previsão não é conclusiva."
                )

            else:
                st.success(
                    f"🌸 Minha previsão é: "
                    f"{classes[indices[0]]}"
                )
                st.caption(
                    "A confiança do modelo não garante que "
                    "a identificação esteja correta."
                )

except Exception as erro:
    st.error(
        "Não foi possível carregar ou executar o modelo. "
        "Verifique se o arquivo classificador_flores.keras "
        "está no repositório."
    )
    st.caption(f"Detalhe técnico: {erro}")

