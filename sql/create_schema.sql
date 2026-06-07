-- ==============================================================================
-- create_schema.sql v2
-- FIX: Ganti TINYINT(1) → BOOLEAN (sesuai MySQL 8.0.17+)
-- Jalankan file ini di MySQL Workbench sebelum ETL pipeline
-- ==============================================================================

CREATE DATABASE IF NOT EXISTS db_banjir_v2
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE db_banjir_v2;

-- ==============================================================================
-- DIM 1: dim_waktu
-- ==============================================================================
CREATE TABLE IF NOT EXISTS dim_waktu (
    id_waktu    INT AUTO_INCREMENT PRIMARY KEY,
    tanggal     DATE        NOT NULL UNIQUE,
    hari        TINYINT,
    bulan       TINYINT,
    tahun       SMALLINT,
    kuartal     TINYINT,
    nama_bulan  VARCHAR(20),
    nama_hari   VARCHAR(15),
    musim       VARCHAR(20)  COMMENT 'Hujan/Kemarau/Peralihan',
    semester    TINYINT,
    minggu_ke   TINYINT,
    is_libur    BOOLEAN      DEFAULT FALSE,
    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==============================================================================
-- DIM 2: dim_lokasi
-- ==============================================================================
CREATE TABLE IF NOT EXISTS dim_lokasi (
    id_lokasi   INT AUTO_INCREMENT PRIMARY KEY,
    provinsi    VARCHAR(120) NOT NULL,
    kabupaten   VARCHAR(150),
    kecamatan   VARCHAR(150),
    latitude    DECIMAL(10,7),
    longitude   DECIMAL(10,7),
    pulau       VARCHAR(60),
    zona_risiko VARCHAR(30)  DEFAULT 'Sedang',
    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_lokasi (provinsi, kabupaten, kecamatan)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==============================================================================
-- DIM 3: dim_cuaca
-- (sumber: master_cuaca_training_banjir_copy.csv — agregasi harian)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS dim_cuaca (
    id_cuaca           INT AUTO_INCREMENT PRIMARY KEY,
    tanggal_cuaca      DATE        NOT NULL,
    id_lokasi          INT,
    suhu_celcius       FLOAT,
    kelembapan_persen  FLOAT,
    curah_hujan_mm     FLOAT,
    kecepatan_angin    FLOAT,
    kondisi_cuaca      VARCHAR(50),
    created_at         TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_lokasi) REFERENCES dim_lokasi(id_lokasi),
    UNIQUE KEY uq_cuaca (tanggal_cuaca, id_lokasi)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==============================================================================
-- DIM 4: dim_lingkungan
-- (sumber: data_banjir_combine_final.csv — fitur geospasial & lingkungan)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS dim_lingkungan (
    id_lingkungan   INT AUTO_INCREMENT PRIMARY KEY,
    id_lokasi       INT,
    elevation       FLOAT        COMMENT 'Ketinggian wilayah (meter)',
    slope           FLOAT        COMMENT 'Kemiringan lereng (derajat)',
    ndvi            FLOAT        COMMENT 'Normalized Difference Vegetation Index (-1 s/d 1)',
    soil_moisture   FLOAT        COMMENT 'Kelembapan tanah (%)',
    landcover_class VARCHAR(60)  COMMENT 'Kelas tutupan lahan',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_lokasi) REFERENCES dim_lokasi(id_lokasi),
    UNIQUE KEY uq_lingkungan (id_lokasi)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==============================================================================
-- FACT: fact_banjir
-- ==============================================================================
CREATE TABLE IF NOT EXISTS fact_banjir (
    id_fakta            INT AUTO_INCREMENT PRIMARY KEY,
    id_waktu            INT          NOT NULL,
    id_lokasi           INT          NOT NULL,
    id_cuaca            INT,
    id_lingkungan       INT,

    -- Cuaca harian (dari dim_cuaca / master_cuaca)
    curah_hujan_mm      FLOAT,
    avg_rainfall        FLOAT        COMMENT 'Avg rainfall dari combine dataset',
    max_rainfall        FLOAT        COMMENT 'Max rainfall dari combine dataset',
    suhu_celcius        FLOAT,
    kelembapan_persen   FLOAT,
    kecepatan_angin     FLOAT,
    kondisi_cuaca       VARCHAR(50),

    -- Fitur lag (curah hujan)
    hujan_lag1          FLOAT        COMMENT 'Curah hujan H-1',
    hujan_lag3          FLOAT        COMMENT 'Curah hujan H-3',
    hujan_lag7          FLOAT        COMMENT 'Curah hujan H-7',
    rolling_avg_3       FLOAT        COMMENT 'Rata-rata curah hujan 3 hari',
    rolling_avg_7       FLOAT        COMMENT 'Rata-rata curah hujan 7 hari',
    rolling_avg_14      FLOAT        COMMENT 'Rata-rata curah hujan 14 hari',

    -- Fitur lingkungan
    elevation           FLOAT,
    slope               FLOAT,
    ndvi                FLOAT,
    soil_moisture       FLOAT,
    landcover_class     VARCHAR(60),

    -- Dampak banjir (dari Data_Bencana.xlsx)
    terjadi_banjir      BOOLEAN      DEFAULT FALSE,
    korban_meninggal    INT          DEFAULT 0,
    korban_hilang       INT          DEFAULT 0,
    korban_terluka      INT          DEFAULT 0,
    rumah_rusak         INT          DEFAULT 0,
    rumah_terendam      INT          DEFAULT 0,
    fasum_rusak         INT          DEFAULT 0,

    -- Label & Prediksi ML
    status_risiko       ENUM('rendah','sedang','tinggi','kritis') DEFAULT 'rendah',
    label_banjir        BOOLEAN      DEFAULT FALSE  COMMENT 'Ground truth label (0/1)',
    prediksi_rf         ENUM('rendah','sedang','tinggi','kritis'),
    prediksi_lstm       FLOAT,
    skor_anomali        FLOAT,
    is_anomali          BOOLEAN      DEFAULT FALSE,

    sumber_data         VARCHAR(60)  DEFAULT 'BNPB+Combine+Cuaca',
    created_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (id_waktu)      REFERENCES dim_waktu(id_waktu),
    FOREIGN KEY (id_lokasi)     REFERENCES dim_lokasi(id_lokasi),
    FOREIGN KEY (id_cuaca)      REFERENCES dim_cuaca(id_cuaca),
    FOREIGN KEY (id_lingkungan) REFERENCES dim_lingkungan(id_lingkungan),

    INDEX idx_waktu     (id_waktu),
    INDEX idx_lokasi    (id_lokasi),
    INDEX idx_risiko    (status_risiko),
    INDEX idx_banjir    (terjadi_banjir),
    INDEX idx_label     (label_banjir)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==============================================================================
-- STAGING TABLES
-- ==============================================================================
CREATE TABLE IF NOT EXISTS stg_bnpb (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    tanggal         DATE,
    kejadian        VARCHAR(50),
    kabupaten       VARCHAR(150),
    provinsi        VARCHAR(120),
    meninggal       INT          DEFAULT 0,
    hilang          INT          DEFAULT 0,
    terluka         INT          DEFAULT 0,
    rumah_rusak     INT          DEFAULT 0,
    rumah_terendam  INT          DEFAULT 0,
    fasum_rusak     INT          DEFAULT 0,
    loaded_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS stg_combine (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    kabupaten       VARCHAR(150),
    kecamatan       VARCHAR(150),
    avg_rainfall    FLOAT,
    max_rainfall    FLOAT,
    avg_temperature FLOAT,
    elevation       FLOAT,
    landcover_class VARCHAR(60),
    ndvi            FLOAT,
    slope           FLOAT,
    soil_moisture   FLOAT,
    tahun           INT,
    bulan           INT,
    label_banjir    BOOLEAN,
    lat             FLOAT,
    lon             FLOAT,
    loaded_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS stg_cuaca (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    provinsi            VARCHAR(120),
    kota_kabupaten      VARCHAR(150),
    kecamatan           VARCHAR(150),
    kelurahan_desa      VARCHAR(150),
    latitude            FLOAT,
    longitude           FLOAT,
    waktu_lokal         DATETIME,
    suhu_celcius        FLOAT,
    kelembapan_persen   FLOAT,
    curah_hujan_mm      FLOAT,
    kondisi_cuaca       VARCHAR(60),
    kecepatan_angin_kmh FLOAT,
    label_banjir        BOOLEAN,
    loaded_at           TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==============================================================================
-- VIEW ANALITIK
-- ==============================================================================
CREATE OR REPLACE VIEW v_analisis_banjir AS
SELECT
    dw.tahun,
    dw.bulan,
    dw.nama_bulan,
    dw.musim,
    dl.provinsi,
    dl.kabupaten,
    dl.kecamatan,
    dl.pulau,
    fb.curah_hujan_mm,
    fb.avg_rainfall,
    fb.max_rainfall,
    fb.rolling_avg_7,
    fb.elevation,
    fb.ndvi,
    fb.slope,
    fb.soil_moisture,
    fb.landcover_class,
    fb.terjadi_banjir,
    fb.label_banjir,
    fb.status_risiko,
    fb.korban_meninggal,
    fb.korban_hilang,
    fb.korban_terluka,
    fb.rumah_rusak,
    fb.rumah_terendam,
    fb.skor_anomali,
    fb.is_anomali,
    dc.kondisi_cuaca,
    dc.kelembapan_persen,
    dc.kecepatan_angin
FROM fact_banjir fb
JOIN dim_waktu      dw ON fb.id_waktu  = dw.id_waktu
JOIN dim_lokasi     dl ON fb.id_lokasi = dl.id_lokasi
LEFT JOIN dim_cuaca dc ON fb.id_cuaca  = dc.id_cuaca;

-- ==============================================================================
-- QUERY ANALITIK REFERENSI
-- ==============================================================================
-- Q1: Provinsi dengan kejadian banjir tertinggi
-- SELECT provinsi, COUNT(*) AS total, SUM(korban_meninggal) AS korban
-- FROM v_analisis_banjir WHERE terjadi_banjir = TRUE
-- GROUP BY provinsi ORDER BY total DESC LIMIT 10;

-- Q2: Rata-rata curah hujan saat terjadi vs tidak terjadi banjir
-- SELECT label_banjir, ROUND(AVG(avg_rainfall),2) AS avg_hujan,
--        ROUND(AVG(elevation),2) AS avg_elevasi
-- FROM v_analisis_banjir GROUP BY label_banjir;

-- Q3: Distribusi risiko per musim
-- SELECT musim, status_risiko, COUNT(*) AS total
-- FROM v_analisis_banjir GROUP BY musim, status_risiko ORDER BY musim;
