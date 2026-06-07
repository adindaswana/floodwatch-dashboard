# Dokumentasi Lengkap Dashboard FloodWatch Indonesia

## 1. Fungsi Utama Dashboard
Dashboard **FloodWatch Indonesia** adalah aplikasi berbasis web interaktif (dibangun menggunakan Streamlit) yang dirancang untuk memantau, menganalisis, dan memprediksi risiko bencana banjir di berbagai wilayah di Indonesia. Secara lengkap, fungsi dari dashboard ini adalah:
- **Pemantauan Historis**: Melihat tren dan pola kejadian banjir dari tahun ke tahun berdasarkan data laporan historis bencana dari Badan Nasional Penanggulangan Bencana (BNPB).
- **Pemetaan Risiko**: Mengidentifikasi provinsi dan wilayah geografis yang paling rawan dan paling sering terdampak banjir.
- **Korelasi Lingkungan & Iklim**: Menganalisis pengaruh berbagai faktor lingkungan (seperti curah hujan, elevasi daratan, tingkat kelembapan tanah, dan indeks kerapatan vegetasi/NDVI) terhadap kemungkinan terjadinya banjir.
- **Evaluasi Machine Learning**: Menampilkan tingkat keberhasilan dan hasil evaluasi dari model prediksi algoritma (Machine Learning) yang telah dilatih untuk memprediksi potensi banjir.
- **Sistem Peringatan Dini (Early Warning)**: Memberikan informasi peringatan kewaspadaan secara otomatis berdasarkan indikator cuaca agregat untuk mendukung langkah mitigasi bencana yang proaktif.

---

## 2. Navigasi dan Kontrol Utama (Sidebar)
Pada sisi kiri layar (Sidebar), terdapat panel navigasi dan kontrol yang berfungsi memfilter data secara global di seluruh tab dashboard:
- **Rentang Tahun (Slider)**: Memungkinkan pengguna mengatur periode waktu kejadian (misal: 2020 hingga 2024). Semua grafik akan beradaptasi dengan periode tahun yang dipilih.
- **Provinsi (Dropdown Multiselect)**: Pengguna dapat memilih satu atau beberapa provinsi tertentu untuk memfokuskan analisis hanya pada lokasi geografis tersebut.
- **Indikator Peringatan (Warning Badge)**: Sebuah indikator visual interaktif (warna merah/kuning/hijau) yang selalu tampil di sidebar. Menunjukkan status peringatan ancaman banjir makro berdasarkan rata-rata tingkat curah hujan saat ini dari filter yang aktif.
- **Informasi Dataset**: Keterangan tertulis mengenai sumber-sumber data yang digunakan dalam dashboard (BNPB DIBI, Data Cuaca satelit, dan ekstraksi fitur lingkungan).

---

## 3. Ringkasan Kinerja Utama (KPI Metrics)
Di bagian paling atas halaman (sebelum masuk ke area tab), terdapat barisan 5 kartu metrik (Key Performance Indicators) yang merangkum angka-angka vital:
1. **Total Kejadian**: Total keseluruhan jumlah insiden banjir yang tercatat pada data historis sesuai filter wilayah.
2. **Korban Meninggal**: Angka akumulasi korban jiwa yang ditimbulkan dari seluruh kejadian banjir.
3. **Rumah Terendam**: Jumlah bangunan/rumah warga yang terendam.
4. **Provinsi Terdampak**: Jumlah sebaran provinsi (dari total 34 provinsi) yang masuk ke dalam data kejadian.
5. **Prediksi Status Kritis**: Prediksi *real-time* jumlah daerah/kabupaten yang masuk dalam zona bahaya (Kritis) dari model algoritma.

---

## 4. Rincian Fitur dan Visualisasi per Tab

Visualisasi data dalam dashboard ini dikategorikan ke dalam 4 halaman/tab utama:

### 📊 Tab 1: Overview & Tren
Fokus pada gambaran statistik makro dan deret waktu dari kejadian banjir.
- **Bar Chart (Top 15 Provinsi Rawan Banjir)**  
  Grafik batang horizontal yang menampilkan ranking 15 provinsi dengan frekuensi kejadian banjir paling banyak. Menggunakan spektrum warna untuk mempertegas frekuensi.
- **Insight Cards (Highlight 3 Provinsi Teratas)**  
  Tiga kartu ringkasan visual yang menyoroti angka pasti dari tiga provinsi paling parah terdampak.
- **Dual-Axis Chart (Tren Banjir Tahunan)**  
  Grafik gabungan yang sangat informatif: menampilkan "Jumlah Kejadian Banjir" menggunakan diagram batang (Bar), sekaligus menampilkan tren "Jumlah Korban Meninggal" menggunakan diagram garis (Line chart) pada rentang waktu tahunan yang sama.

### 🌍 Tab 2: Lingkungan & Cuaca
Fokus pada korelasi antara variabel alam/iklim dengan kejadian banjir.
- **Pie/Donut Chart (Distribusi Risiko Banjir)**  
  Menunjukkan proporsi (persentase) wilayah yang dipetakan ke dalam 4 zona risiko: Rendah (Hijau), Sedang (Kuning), Tinggi (Oranye), dan Kritis (Merah).
- **Box Plot (Curah Hujan: Banjir vs Tidak Banjir)**  
  Grafik statistika (Box plot) yang sangat berguna untuk melihat sebaran dan *outliers* tingkat curah hujan saat hari terjadi banjir dibandingkan dengan hari normal tanpa banjir.
- **Scatter Plot Interaktif (Hubungan Elevasi & Curah Hujan)**  
  Grafik titik koordinat (plot sebar) dari ribuan sampel data yang membandingkan ketinggian tanah (Elevasi) dengan Curah Hujan. Titik diberi warna berbeda jika wilayah tersebut mengalami banjir atau tidak.
- **Time-Series Subplots (Data Cuaca Harian)**  
  Dua grafik yang sinkron secara sumbu X (tanggal). Grafik atas menampilkan deret batang untuk Curah Hujan Harian, dan grafik bawah menampilkan fluktuasi garis untuk Suhu dan Kelembapan Udara harian.

### 🤖 Tab 3: Machine Learning
Fokus untuk mengevaluasi kualitas dan pola pengambilan keputusan dari kecerdasan buatan (algoritma prediksi).
- **Cards Metrik Performa Evaluasi Model**: Menampilkan skor teknis kehebatan model berupa **Akurasi Model** (persentase tebakan total yang benar), **Precision** (ketepatan tebakan kelas positif), dan **Recall** (sensitivitas pendeteksian target banjir).
- **Bar Chart (Feature Importance)**  
  Grafik yang merangking seberapa besar tingkat pengaruh setiap variabel/fitur (seperti curah hujan, elevasi, suhu) terhadap keputusan model dalam menentukan prediksi banjir.
- **Heatmap (Confusion Matrix)**  
  Matriks evaluasi berbentuk peta panas yang merinci jumlah tebakan benar (True Positive, True Negative) dan tebakan meleset (False Positive, False Negative) dari model klasifikasi.

### 🚨 Tab 4: Data & Peringatan
Tab operasional yang bersifat analitik detil dan memfasilitasi peringatan dini (*early warning*).
- **Fitur 1: Sistem Peringatan Dini Otomatis**
  - **Gauge Chart (Speedometer)**: Indikator jarum jam yang menampilkan visualisasi rata-rata curah hujan wilayah beserta zona amannya.
  - **Status Risk Metrics**: 4 angka jumlah rekapitulasi daerah pada level Kritis, Tinggi, Sedang, dan Rendah.
  - **Tabel Peringatan Darurat**: Menampilkan daftar sorotan khusus wilayah/kabupaten yang hanya berstatus "Kritis" dan "Tinggi", lengkap dengan data pemicunya (curah hujan maksimum, indeks risiko, dll).
- **Fitur 2: Modul Ekspor Data Laporan**
  - Menyediakan fungsionalitas pengunduhan (Download). Terdapat 3 tombol interaktif untuk menyimpan data dalam format *Spreadsheet* (`.csv`), yaitu: Data Banjir BNPB, Data Fitur ML, dan Data Cuaca mentah.
- **Fitur 3: Tabel Investigasi Detail (Interactive Dataframe)**
  - **Kontrol Filter Khusus Tabel**: Filter *dropdown* untuk menyaring baris tabel berdasarkan nama Kabupaten, jenis Musim (Kemarau/Hujan), level Risiko, dan kejadian Banjir.
  - **Ringkasan Metrik Dinamis**: Angka rata-rata (Total Data, Rata-rata Hujan, Jumlah Kabupaten terdampak) yang beradaptasi dengan filter tabel di atasnya.
  - **Dataframe Wilayah**: Tabel mendetail baris per baris yang interaktif, menampilkan rekam kejadian setiap titik dengan kolom seperti Elevasi, NDVI, Suhu, Status Banjir, dsb.
