import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import random


# Daftar gambar banner yang tersedia
banner_images = ["banner.jpeg", "banner1.jpeg", "banner2.jpeg"]  # Ganti dengan nama gambar Anda

# Pilih gambar secara acak setiap kali halaman dimuat
selected_image = random.choice(banner_images)


if os.path.exists(selected_image):
    st.write(f"File gambar {selected_image} ditemukan!")
    st.image(selected_image, use_container_width=True)
else:
    st.error(f"File gambar {selected_image} tidak ditemukan! Pastikan gambar ada di direktori yang sama dengan aplikasi.")
    
# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Akuisisi Data dan Analisis", page_icon="📊", layout="wide")
st.markdown(""" 
    <style>
        .block-container { background-color: #9CAF88; }  /* Warna sage */
        .stButton>button { background-color: #4CAF50; color: white; }
        .stSlider>div { background-color: #dff0d8; padding: 5px; border-radius: 10px; }
        .stFileUploader>div { background-color: #dff0f7; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)
# Inisialisasi session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'Beranda'

# Judul Aplikasi dan Gambar Banner
st.image(selected_image, use_container_width=True)
st.title("📊 SPORTGEN")

# Sidebar dengan tombol untuk navigasi
st.sidebar.title("SPORTGEN")
if st.sidebar.button("Beranda"):
    st.session_state['page'] = 'Beranda'
if st.sidebar.button("Analisis Data"):
    st.session_state['page'] = 'Analisis Data'

# Halaman Beranda
if st.session_state['page'] == 'Beranda':
    st.header("Selamat Datang di Aplikasi SPORTGEN!")
    st.write(""" 
        Aplikasi ini memungkinkan Anda untuk mengunggah dataset CSV, melakukan preprocessing data, 
        serta melakukan klasterisasi menggunakan algoritma KMeans dengan berbagai kombinasi fitur.
        """)
    st.write("### Panduan Penggunaan: ")
    st.write("1. Pilih 'Analisis Data' untuk mengunggah dataset.")
    st.write("2. Standarisasi data dan pilih jumlah kluster untuk melakukan klasterisasi.")
    st.write("3. Visualisasikan hasil klasterisasi.")

# Halaman Analisis Data
elif st.session_state['page'] == 'Analisis Data':
    st.header("1️⃣ Unggah Dataset")
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset yang diunggah:")
        st.write(data)

        # Memilih hanya data numerik dan menghapus baris dengan nilai NaN
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        numeric_data = numeric_data.dropna()  # Menghapus baris yang mengandung NaN

        if numeric_data.empty:
            st.warning("Data yang diunggah tidak mengandung kolom numerik yang valid atau semua nilai adalah NaN.")
        else:
            # Standarisasi data numerik
            st.header("2️⃣ Preprocess Data")
            if st.button("Standarisasi Data"):
                try:
                    scaler = StandardScaler()
                    # Periksa apakah numeric_data bukan kosong atau tidak valid
                    if numeric_data.shape[0] > 0:
                        data_scaled = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
                        st.session_state['data_scaled'] = data_scaled
                        st.write("### Data setelah standarisasi:")
                        st.write(data_scaled)
                    else:
                        st.warning("Data numerik tidak memiliki baris yang cukup untuk standarisasi.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan standarisasi: {e}")

            # Klasterisasi
            st.header("3️⃣ Klasterisasi KMeans")
            num_clusters = st.slider("Pilih jumlah kluster:", min_value=2, max_value=10, value=3)

            if st.button("Lakukan Klasterisasi"):
                if 'data_scaled' in st.session_state and not st.session_state['data_scaled'].empty:
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    st.session_state['data_scaled']['cluster'] = kmeans.fit_predict(st.session_state['data_scaled'])
                    st.session_state['kmeans'] = kmeans
                    st.write("### Data dengan klaster:")
                    st.write(st.session_state['data_scaled'])
                else:
                    st.warning("Data harus distandarisasi terlebih dahulu.")

            # Visualisasi
            st.header("4️⃣ Visualisasi Klasterisasi")
            st.subheader("Pilih kombinasi fitur atau gunakan semua fitur dengan PCA:")

            # Opsi fitur untuk scatter plot
            feature_options = ["PTS vs AST", "MP vs TRB", "PTS, STL, BLK (3D)", "PTS vs CS", "Semua Fitur (PCA)"]
            selected_feature = st.selectbox("Pilih Kombinasi Fitur:", feature_options)

            # Visualisasi scatter plot sesuai fitur yang dipilih
            if st.button("Tampilkan Visualisasi"):
                cluster_colors = ['blue', 'red'] + [plt.cm.tab10(i) for i in range(2, num_clusters)]

                # PTS vs AST
                if selected_feature == "PTS vs AST":
                    st.markdown("### **Visualisasi: PTS vs Assists (AST)**")
                    st.markdown(""" 
                    - Menampilkan hubungan antara **Points (PTS)** dan **Assists (AST)** pemain.
                    - Klaster membantu mengelompokkan pemain berdasarkan kontribusi skor dan assist mereka.
                    """)

                    plt.figure(figsize=(10, 6))
                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        plt.scatter(cluster_data['PTS'], cluster_data['AST'], label=f'Cluster {i+1}', color=cluster_colors[i % len(cluster_colors)], s=50)
                    plt.xlabel("Points (PTS)")
                    plt.ylabel("Assists (AST)")
                    plt.title("Pemain berdasarkan kontribusi skor dan assist")
                    plt.legend()
                    st.pyplot(plt)

                    # Analisis Hasil Visualisasi
                    st.markdown("### **Hasil Analisis Klasterisasi: PTS vs AST**")
                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        avg_pts = cluster_data['PTS'].mean()
                        avg_ast = cluster_data['AST'].mean()

                        st.markdown(f"#### **Cluster {i+1}:**")
                        if avg_pts > cluster_data['PTS'].median() and avg_ast > cluster_data['AST'].median():
                            st.write(f"Kelompok ini berisi pemain yang sangat produktif dalam mencetak poin dan memberikan assist.")
                        elif avg_pts > cluster_data['PTS'].median():
                            st.write(f"Kelompok ini berisi pemain dengan kontribusi besar dalam mencetak poin.")
                        elif avg_ast > cluster_data['AST'].median():
                            st.write(f"Kelompok ini berisi pemain dengan kemampuan memberikan assist yang tinggi.")
                        else:
                            st.write(f"Kelompok ini berisi pemain dengan kontribusi skor dan assist yang relatif rendah.")

                # MP vs TRB
                elif selected_feature == "MP vs TRB":
                    st.markdown("### **Visualisasi: MP vs Total Rebounds (TRB)**")
                    st.markdown(""" 
                    - Menampilkan hubungan antara **Minutes Played (MP)** dan **Total Rebounds (TRB)** pemain.
                    - Klaster membantu mengelompokkan pemain berdasarkan kontribusi waktu bermain dan rebound mereka.
                    """)

                    plt.figure(figsize=(10, 6))
                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        plt.scatter(cluster_data['MP'], cluster_data['TRB'], label=f'Cluster {i+1}', color=cluster_colors[i % len(cluster_colors)], s=50)
                    plt.xlabel("Minutes Played (MP)")
                    plt.ylabel("Total Rebounds (TRB)")
                    plt.title("Pemain berdasarkan waktu bermain dan rebound")
                    plt.legend()
                    st.pyplot(plt)
                    # Analisis Hasil Visualisasi
                    st.markdown("### **Hasil Analisis Klasterisasi: MP vs TRB**")
                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        avg_mp = cluster_data['MP'].mean()
                        avg_trb = cluster_data['TRB'].mean()

                        st.markdown(f"#### **Cluster {i+1}:**")
                        if avg_mp > cluster_data['MP'].median() and avg_trb > cluster_data['TRB'].median():
                            st.write(f"Kelompok ini berisi pemain dengan kontribusi tinggi dalam rebound dan waktu bermain yang panjang.")
                        elif avg_mp > cluster_data['MP'].median():
                            st.write(f"Kelompok ini berisi pemain dengan waktu bermain yang tinggi, namun kontribusi reboundnya lebih rendah.")
                        elif avg_trb > cluster_data['TRB'].median():
                            st.write(f"Kelompok ini didominasi oleh pemain yang sangat aktif dalam rebound.")
                        else:
                            st.write(f"Kelompok ini berisi pemain dengan kontribusi rebound dan waktu bermain yang relatif rendah.")

                # PTS, STL, BLK (3D)
                elif selected_feature == "PTS, STL, BLK (3D)":
                    st.markdown("### **Visualisasi: PTS, Steals (STL), Blocks (BLK) dalam 3D**")
                    st.markdown(""" 
                    - Menampilkan hubungan antara **Points (PTS)**, **Steals (STL)**, dan **Blocks (BLK)** dalam ruang tiga dimensi.
                    - Klaster membantu mengelompokkan pemain berdasarkan kontribusi dalam poin, steals, dan blocks mereka.
                    """)

                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        ax.scatter(cluster_data['PTS'], cluster_data['STL'], cluster_data['BLK'], label=f'Cluster {i+1}', color=cluster_colors[i % len(cluster_colors)], s=50)

                    ax.set_xlabel("Points (PTS)")
                    ax.set_ylabel("Steals (STL)")
                    ax.set_zlabel("Blocks (BLK)")
                    ax.set_title("Visualisasi 3D: PTS, STL, dan BLK")
                    plt.legend()
                    st.pyplot(fig)
                    # Analisis Hasil Visualisasi
                    st.markdown("### **Hasil Analisis Klasterisasi: PTS, STL, BLK**")
                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        avg_pts = cluster_data['PTS'].mean()
                        avg_stl = cluster_data['STL'].mean()
                        avg_blk = cluster_data['BLK'].mean()

                        st.markdown(f"#### **Cluster {i+1}:**")
                        if avg_pts > cluster_data['PTS'].median() and avg_stl > cluster_data['STL'].median() and avg_blk > cluster_data['BLK'].median():
                            st.write(f"Kelompok ini berisi pemain yang sangat produktif, baik dalam mencetak poin, steals, dan blocks.")
                        elif avg_pts > cluster_data['PTS'].median() and avg_stl > cluster_data['STL'].median():
                            st.write(f"Kelompok ini terdiri dari pemain yang sangat baik dalam menyerang dan mencuri bola.")
                        elif avg_pts > cluster_data['PTS'].median() and avg_blk > cluster_data['BLK'].median():
                            st.write(f"Kelompok ini lebih dominan dalam menyerang dan memblok tembakan lawan.")
                        elif avg_stl > cluster_data['STL'].median() and avg_blk > cluster_data['BLK'].median():
                            st.write(f"Kelompok ini berisi pemain dengan kontribusi tinggi dalam mencuri bola dan memblok tembakan.")
                        else:
                            st.write(f"Kelompok ini memiliki kontribusi yang lebih seimbang, namun dengan skor dan statistik defensif yang lebih rendah.")

                # Semua Fitur (PCA)
                elif selected_feature == "Semua Fitur (PCA)":
                    st.markdown("### **Visualisasi: Semua Fitur dengan PCA**")
                    st.markdown(""" 
                    - Dimensi data direndahkan menggunakan PCA untuk memvisualisasikan hubungan antara semua fitur.
                    - Klaster ditampilkan berdasarkan hasil PCA.
                    """)

                    pca = PCA(n_components=3)
                    pca_result = pca.fit_transform(st.session_state['data_scaled'].drop(columns=['cluster']))

                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        ax.scatter(pca_result[st.session_state['data_scaled']['cluster'] == i, 0],
                                   pca_result[st.session_state['data_scaled']['cluster'] == i, 1],
                                   pca_result[st.session_state['data_scaled']['cluster'] == i, 2],
                                   label=f'Cluster {i+1}', color=cluster_colors[i % len(cluster_colors)], s=50)

                    ax.set_xlabel("PCA Komponen 1")
                    ax.set_ylabel("PCA Komponen 2")
                    ax.set_zlabel("PCA Komponen 3")
                    ax.set_title("Visualisasi PCA - Klasterisasi")
                    plt.legend()
                    st.pyplot(fig)
                    # Analisis untuk Semua Fitur (PCA)
                    st.markdown("### **Hasil Analisis Klasterisasi: Semua Fitur dengan PCA**")
                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
    
    # Menghitung rata-rata untuk fitur pada tiap klaster
                        pca_mean = cluster_data.drop(columns=['cluster']).mean()

                        st.markdown(f"#### **Cluster {i+1}:**")
                        avg_pca = pca_mean.mean()

    # Menghitung apakah rata-rata pca lebih besar dari median
                        if avg_pca > pca_mean.median():
                            st.write(f"Klaster ini berisi fitur dengan kontribusi besar pada komponen utama PCA.")
                        else:
                            st.write(f"Klaster ini mencakup fitur yang lebih tersebar di antara berbagai komponen utama PCA.")

    # Menampilkan rata-rata komponen utama PCA untuk tiap klaster
                        st.markdown(f"**Rata-rata komponen utama PCA untuk Cluster {i+1}:**")
                        st.write(f"Komponen 1: {pca_mean[0]} | Komponen 2: {pca_mean[1]} | Komponen 3: {pca_mean[2]}")

    # Analisis berdasarkan nilai rata-rata komponen
                        if pca_mean[0] > pca_mean[1] and pca_mean[0] > pca_mean[2]:
                            st.write("Komponen utama 1 memiliki kontribusi paling dominan dalam klaster ini.")
                        elif pca_mean[1] > pca_mean[0] and pca_mean[1] > pca_mean[2]:
                            st.write("Komponen utama 2 memiliki kontribusi paling dominan dalam klaster ini.")
                        elif pca_mean[2] > pca_mean[0] and pca_mean[2] > pca_mean[1]:
                            st.write("Komponen utama 3 memiliki kontribusi paling dominan dalam klaster ini.")
                        else:
                            st.write("Fitur-fitur dalam klaster ini memiliki kontribusi yang relatif merata di antara komponen utama PCA.")

                elif selected_feature == "PTS vs CS":
                    st.markdown("### **Visualisasi: PTS vs Steals (CS)**")
                    st.markdown(""" 
                     - Menampilkan hubungan antara **Points (PTS)** dan **Steals (CS)** pemain.
                     - Klaster membantu mengelompokkan pemain berdasarkan kontribusi skor dan jumlah steal mereka.
                 """)

                    plt.figure(figsize=(10, 6))
                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        plt.scatter(cluster_data['PTS'], cluster_data['STL'], label=f'Cluster {i+1}', color=cluster_colors[i % len(cluster_colors)], s=50)
                    plt.xlabel("Points (PTS)")
                    plt.ylabel("Steals (CS)")
                    plt.title("Pemain berdasarkan kontribusi skor dan steals")
                    plt.legend()
                    st.pyplot(plt)

                     # Analisis Hasil Visualisasi
                    st.markdown("### **Hasil Analisis Klasterisasi: PTS vs CS**")
                    for i in range(num_clusters):
                        cluster_data = st.session_state['data_scaled'][st.session_state['data_scaled']['cluster'] == i]
                        avg_pts = cluster_data['PTS'].mean()
                        avg_stl = cluster_data['STL'].mean()

                        st.markdown(f"#### **Cluster {i+1}:**")
                        if avg_pts > cluster_data['PTS'].median() and avg_stl > cluster_data['STL'].median():
                            st.write(f"Kelompok ini cenderung berisi pemain dengan kemampuan mencuri bola tinggi dan kontribusi skor yang besar.")
                        elif avg_pts > cluster_data['PTS'].median():
                            st.write(f"Kelompok ini berisi pemain dengan kemampuan menyerang tinggi dan skor besar.")
                        elif avg_stl > cluster_data['STL'].median():
                            st.write(f"Kelompok ini didominasi oleh pemain dengan kemampuan steals tinggi.")
                        else:
                            st.write(f"Kelompok ini mencakup pemain dengan kontribusi skor dan steals yang relatif rendah.")
