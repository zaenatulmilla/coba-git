import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Klasifikasi Katarak CNN",
    page_icon="👁️",
    layout="centered"
)

# ===============================
# Download & Load Model
# ===============================
MODEL_PATH = "model_fix.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1ehai4eFuIAeetBTquJGsMubJisDPNXCi"  # ganti dengan ID Google Drive model kamu
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
# Judul & Deskripsi
# ===============================
st.title("👁️ Website Klasifikasi Mata Katarak Berbasis CNN")
st.write("""
Selamat datang di **Website Prediksi Katarak**.  
Website ini dirancang untuk membantu mengunggah gambar mata dan mengklasifikasikannya 
apakah **Normal** atau **Katarak** menggunakan model **Convolutional Neural Network (CNN)**.
""")

st.divider()

# ===============================
# Galeri Contoh
# ===============================
st.subheader("🖼️ Contoh Gambar Mata")
st.write("Berikut contoh citra mata **Normal** dan **Katarak**:")

def load_gallery_images(folder):
    paths = []
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(folder, file))
    return sorted(paths)

# pastikan kamu punya folder "gallery/normal" dan "gallery/katarak"
normal_imgs = load_gallery_images("gallery/normal")
katarak_imgs = load_gallery_images("gallery/katarak")

cols = st.columns(2)

with cols[0]:
    st.markdown("### 🟢 Normal")
    for img_path in normal_imgs:
        img = Image.open(img_path)
        img = img.resize((250, 250))
        st.image(img, use_container_width=False, caption="Normal")

with cols[1]:
    st.markdown("### 🔴 Katarak")
    for img_path in katarak_imgs:
        img = Image.open(img_path)
        img = img.resize((250, 250))
        st.image(img, use_container_width=False, caption="Katarak")

st.divider()

# ===============================
# Cara Menggunakan
# ===============================
st.subheader("📖 Cara Menggunakan:")
st.write("""
1. Klik tombol **Browse Files** untuk mengunggah citra mata.  
2. Setelah gambar diunggah, sistem akan memproses menggunakan model CNN.  
3. Hasil prediksi berupa status **Normal** atau **Katarak** akan ditampilkan bersama skor kepercayaan.
""")

st.divider()

# ===============================
# Upload & Prediksi
# ===============================
st.subheader("🔬 Uji Coba Klasifikasi:")
uploaded_file = st.file_uploader("📂 Unggah gambar mata (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

    # preprocessing
    img = img.resize((299, 299))  # sesuaikan dengan input model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

    # prediksi
    with st.spinner("🔍 Sedang menganalisis..."):
        pred = model.predict(img_array)[0][0]

    st.divider()

    # hasil prediksi
    st.subheader("📊 Hasil Prediksi:")
    if pred < 0.5:
        st.error(f"🔴 Katarak terdeteksi\n\n**Skor Kepercayaan: {(1 - pred) * 100:.2f}%**")
    else:
        st.success(f"🟢 Mata Normal\n\n**Skor Kepercayaan: {pred * 100:.2f}%**")

    st.caption("⚠️ Catatan: Hasil ini hanya sebagai screening awal, bukan pengganti diagnosa dokter mata.")
