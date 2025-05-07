import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

# === Load model ===
@st.cache_resource
def load_cnn_model():
    model = load_model('cnn_.keras')  # Ganti nama file model jika perlu
    return model

# === Label kelas ===
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# === Fungsi preprocessing gambar ===
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     
    image = cv2.resize(image, (128, 128))               
    image = image / 255.0                               
    image = np.expand_dims(image, axis=-1)              
    image = np.expand_dims(image, axis=0)               
    return image

# === Fungsi utama untuk dijalankan dari app.py ===
def run():
    model = load_cnn_model()

    st.title("Klasifikasi Ekspresi Wajah")
    
    try:
        image = Image.open('kumpul.jpeg')
        st.image(image, caption='Heart Attack')
    except:
        st.warning("Gambar 'kumpul.jpeg' tidak ditemukan")

    st.write("Tekan tombol 'Browse files' untuk upload gambar wajah untuk memprediksi ekspresi emosinya.")

    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image_rgb, caption="Gambar yang diupload", use_column_width=True)

        input_image = preprocess_image(image_rgb)
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction)
        predicted_label = labels[predicted_class]
        confidence = prediction[0][predicted_class]

        st.markdown(f"### Prediksi: `{predicted_label.upper()}`")
        st.write(f"Hasil dari prediksi wajahmu yaitu {predicted_label.upper()}")

        try:
            result_image = Image.open('yolo.jpeg')
            st.image(result_image)
        except:
            st.warning("Gambar 'yolo.jpeg' tidak ditemukan")
if __name__ == "__main__":
    run()