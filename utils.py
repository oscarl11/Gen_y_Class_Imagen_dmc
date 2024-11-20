import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline

# Carga y cachea el modelo de generación de imágenes
@st.cache_resource
def load_image_generation_model():
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype="float32"
    )
    model.to("cuda")  
    return model

# Carga y cachea el modelo de clasificación de imágenes
@st.cache_resource
def load_image_classification_model():
    classifier = pipeline("image-classification", model="microsoft/resnet-50")
    return classifier
