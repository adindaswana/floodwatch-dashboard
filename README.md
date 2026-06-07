# FloodWatch Dashboard

FloodWatch Dashboard merupakan platform interaktif untuk pemantauan, analisis, dan prediksi risiko banjir yang mengintegrasikan proses ETL (Extract, Transform, Load), machine learning, dan visualisasi data dalam satu sistem berbasis Streamlit.

## Live Demo

Dashboard dapat diakses melalui:

https://floodwatch-dashboard-eny4agircntmnkappwoywv.streamlit.app/

---

## Deskripsi Proyek

FloodWatch Dashboard dikembangkan untuk membantu proses analisis risiko banjir berdasarkan data cuaca dan data kejadian bencana. Sistem ini menggabungkan pengolahan data, pemodelan machine learning, dan visualisasi interaktif sehingga pengguna dapat memperoleh informasi yang lebih mudah dipahami dan mendukung pengambilan keputusan berbasis data.

Proyek ini merupakan implementasi konsep Data Science dan Machine Learning pada bidang mitigasi bencana dan analisis lingkungan.

---

## Tujuan Proyek

- Mengintegrasikan data cuaca dan data kejadian banjir ke dalam satu sistem analitik.
- Melakukan analisis eksploratif terhadap faktor-faktor yang memengaruhi risiko banjir.
- Mengembangkan model machine learning untuk klasifikasi risiko banjir.
- Menyediakan dashboard interaktif untuk visualisasi dan interpretasi data.
- Menampilkan sistem deteksi dini berdasarkan kondisi cuaca dan karakteristik wilayah.

---

## Fitur Utama

### Dashboard Interaktif

- Ringkasan statistik data cuaca dan kejadian banjir.
- Visualisasi tren kejadian banjir.
- Analisis distribusi tingkat risiko banjir.
- Heatmap korelasi antar variabel.
- Visualisasi data interaktif menggunakan Plotly.

### Machine Learning

- Prediksi risiko banjir menggunakan algoritma Random Forest.
- Evaluasi performa model klasifikasi.
- Analisis feature importance.
- Integrasi hasil prediksi ke dalam dashboard.

### Sistem Deteksi Dini

- Identifikasi kondisi cuaca berisiko tinggi.
- Klasifikasi tingkat risiko banjir.
- Penyajian peringatan dini berbasis data.

---

## Teknologi yang Digunakan

### Bahasa Pemrograman

- Python

### Framework dan Library

- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Matplotlib
- Seaborn
- SQLAlchemy
- PyMySQL
- OpenPyXL

### Machine Learning

- Random Forest Classifier

### Pengolahan Data dan ETL

- SQLAlchemy
- Pandas
- PyMySQL

---

## Struktur Proyek

```text
floodwatch-dashboard/
│
├── dashboard/
│   ├── app.py
│   ├── assets/
│   └── output/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── etl/
│   ├── extract.py
│   ├── transform.py
│   └── load.py
│
├── ml/
│   ├── random_forest.py
│   ├── evaluate.py
│   ├── xgboost_model.py
│   └── lstm_model.py
│
├── models/
├── sql/
├── requirements.txt
└── README.md
```


