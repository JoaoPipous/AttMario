import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def convert_to_grayscale_average(image):
    return ImageOps.grayscale(image)

def convert_to_grayscale_perceptual(image):
    img_array = np.array(image)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return Image.fromarray(gray)

def convert_to_grayscale_gamma_corrected(image):
    img_array = np.array(image) / 255.0
    corrected = np.power(img_array, 2.2)
    gray = corrected @ [0.299, 0.587, 0.114]
    final_image = (gray**(1/2.2) * 255).astype(np.uint8)
    return Image.fromarray(final_image)

def apply_binarization(image, threshold):
    return image.point(lambda p: 255 if p > threshold else 0)

def apply_interval_threshold(image, low, high):
    return image.point(lambda p: 255 if low < p < high else 0)

st.title("Image Processing Application")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Original Image", use_column_width=True)
    
    gray_avg = convert_to_grayscale_average(img)
    gray_perceptual = convert_to_grayscale_perceptual(img)
    gray_gamma_corrected = convert_to_grayscale_gamma_corrected(img)
    
    st.write("### Grayscale Conversions")
    col1, col2, col3 = st.columns(3)
    col1.image(gray_avg, caption="Average Grayscale", use_column_width=True)
    col2.image(gray_perceptual, caption="Perceptual Grayscale", use_column_width=True)
    col3.image(gray_gamma_corrected, caption="Gamma Corrected Grayscale", use_column_width=True)
    
    st.write("### Histogram (Average Grayscale)")
    hist_data = np.array(gray_avg).flatten()
    fig, ax = plt.subplots()
    ax.hist(hist_data, bins=256, range=(0, 256), color="gray")
    st.pyplot(fig)
    
    st.write("### Additional Image Processing")
    operation = st.selectbox("Choose an operation", ("Binarization", "Interval Threshold"))
    processing_mode = st.selectbox("Apply operation to", ("Original Color", "Average Grayscale", "Perceptual Grayscale", "Gamma Corrected Grayscale"))
    
    if processing_mode == "Original Color":
        target_image = img
    elif processing_mode == "Average Grayscale":
        target_image = gray_avg
    elif processing_mode == "Perceptual Grayscale":
        target_image = gray_perceptual
    else:
        target_image = gray_gamma_corrected
    
    if operation == "Binarization":
        threshold = st.slider("Set threshold", 0, 255, 128)
        processed_image = apply_binarization(target_image.convert("L") if processing_mode == "Original Color" else target_image, threshold)
        st.image(processed_image, caption="Binarized Image", use_column_width=True)
    elif operation == "Interval Threshold":
        low = st.slider("Set lower threshold", 0, 255, 50)
        high = st.slider("Set upper threshold", 0, 255, 200)
        processed_image = apply_interval_threshold(target_image.convert("L") if processing_mode == "Original Color" else target_image, low, high)
        st.image(processed_image, caption="Interval Threshold Image", use_column_width=True)
