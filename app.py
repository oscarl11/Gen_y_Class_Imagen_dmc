import streamlit as st
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from diffusers import StableDiffusionPipeline
from io import BytesIO

# Configuración del modelo de clasificación de imagen
MODEL_NAME = "microsoft/resnet-50"
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

# Función para cargar y clasificar la imagen
def classify_image(image):
    # Asegurarse de que la imagen esté en formato RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocesar la imagen
    inputs = extractor(images=image, return_tensors="pt")
    # Clasificar la imagen
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    return predicted_class

# Configuración del pipeline de generación de imágenes usando 'diffusers'
device = "cpu"  # Aseguramos que la generación de imagen use el CPU
generator = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
generator.to(device)

# Función para generar imagen a partir de texto
def generate_image(prompt):
    image = generator(prompt).images[0]
    return image

# Título y descripción de la app
st.title("Generación y Clasificación de Imágenes")
st.write("Selecciona una de las opciones a continuación:")

# Opciones para el usuario
option = st.selectbox(
    '¿Qué te gustaría hacer?',
    ['Generar Imagen a partir de Texto', 'Clasificar Imagen']
)

# Sección para la generación de imagen
if option == 'Generar Imagen a partir de Texto':
    st.subheader("Generación de Imagen")

    # Entrada de texto para el prompt
    prompt = st.text_input("Escribe una descripción para generar una imagen:")

    # Botón para generar la imagen
    if st.button("Generar Imagen"):
        if prompt:
            with st.spinner("Generando imagen..."):
                generated_image = generate_image(prompt)
            st.image(generated_image, caption="Imagen generada", use_column_width=True)
        else:
            st.warning("Por favor, ingresa un texto para generar la imagen.")

# Sección para la clasificación de imagen
elif option == 'Clasificar Imagen':
    st.subheader("Clasificación de Imagen")

    # Subir imagen
    uploaded_image = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

    # Mostrar imagen cargada
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Botón para ejecutar la clasificación de la imagen
        if st.button("Clasificar Imagen"):
            if image is not None:
                # Mostrar un spinner mientras se clasifica la imagen
                with st.spinner("Clasificando..."):
                    classification = classify_image(image)
                st.write(f"**Clasificación:** {classification}")
            else:
                st.warning("Por favor, sube una imagen para clasificar.")

