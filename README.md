# COMP8035041 Big Data Analytics: Lakehouse Pipeline, Machine Learning, Graph Analytics, dan Dashboard

Platform Big Data multimodal untuk penilaian harga dan kondisi mobil bekas UK, dari dataset gabungan listing dan anotasi gambar kerusakan kendaraan. Pipeline-nya mencakup lakehouse tiga layer, prediksi harga dengan scikit-learn dan Spark MLlib, graph analytics dengan GraphFrames, serta dashboard untuk interpretasi bisnis.

Satyasiddhi Josh (2702750571), Magister Teknik Informatika, BINUS Graduate Program.

## Isi repository

| Folder atau file                                             | Isi                                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `src/build_dataset.py`                                       | Penggabungan dataset listing dan gambar (lihat `datasetReadme.md`)             |
| `src/s01_ingest_raw.py` sampai `src/s09_export_dashboard.py` | Sembilan tahap pipeline                                                        |
| `src/config.py`, `src/common_ml.py`                          | Konfigurasi path, SparkSession, dan fungsi bersama                             |
| `run_pipeline.py`                                            | Orkestrator lokal, berhenti bila satu tahap gagal                              |
| `dags/used_car_pipeline_dag.py`                              | DAG Airflow dengan urutan tahap yang sama (untuk Linux, WSL, atau Docker)      |
| `data/processed/`                                            | Dataset gabungan (CSV, Parquet, metadata build)                                |
| `data/raw/`                                                  | Raw Zone hasil tahap s01: salinan CSV dan `_manifest.json`                     |
| `lakehouse/curated/`                                         | Curated Zone, tabel Delta Lake dipartisi per brand, plus tabel quarantine      |
| `lakehouse/serving/`                                         | Serving Layer, tabel agregat Delta Lake                                        |
| `lakehouse/ml/`                                              | Dataset ML (Parquet) dan model GBT Spark MLlib                                 |
| `outputs/`                                                   | Laporan kualitas data, metrik, hasil tuning, prediksi, hasil graph, dan figure |
| `dashboard/data/`                                            | CSV hasil tahap s09, sumber data dashboard                                     |
| `dashboard/Dashboard_Final_Project.pbix`                     | Dashboard Power BI (data sudah tersimpan di dalam file)                        |
| `dashboard/app.py`                                           | Versi web dashboard (Streamlit) dengan data dan halaman yang sama              |
| `dashboard/PANDUAN_POWER_BI.md`                              | Langkah membangun dashboard Power BI dari CSV hasil s09                        |

Folder `data/`, `lakehouse/`, dan `outputs/` adalah hasil eksekusi pipeline yang sebenarnya, jadi bisa dicek tanpa menjalankan ulang.

## Dataset

Dataset asli tidak disertakan karena ukurannya besar. Unduh dari Kaggle:

- Listing mobil bekas UK (Desai, 2020): `adityadesai13/used-car-dataset-ford-and-mercedes`, letakkan di folder `Uk selling cars/`
- Gambar dan anotasi kerusakan (Humans in the Loop, 2023): `humansintheloop/car-parts-and-car-damages`, letakkan di folder `car damages images/`

Dataset listing tidak menyatakan lisensinya, sehingga data di repository ini hanya untuk keperluan akademik.

## Menjalankan pipeline (Windows)

Prasyarat:

1. Python 3.11 dan Java 17.
2. `winutils.exe` dan `hadoop.dll` untuk Hadoop 3 di `C:\hadoop\bin`. Set `HADOOP_HOME=C:\hadoop`, tambahkan `C:\hadoop\bin` ke Path, lalu salin `hadoop.dll` ke `C:\Windows\System32`.
3. Path folder project tidak mengandung spasi.
4. Koneksi internet saat pertama kali jalan, karena package Delta Lake dan GraphFrames diunduh dari Maven.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\check_env.py
python run_pipeline.py
```

Opsi lain:

```powershell
python run_pipeline.py --from s05   # mulai dari tahap tertentu
python run_pipeline.py --only s09   # satu tahap saja
```

## Dashboard

- **Versi web:** `pip install streamlit plotly`, lalu `streamlit run dashboard/app.py`.

## Hasil utama

| Bagian          | Hasil                                                                   |
| --------------- | ----------------------------------------------------------------------- |
| Quality gate    | 301 dari 97.712 baris ditandai (0,31%), ambang henti 5%                 |
| Model terbaik   | Gradient Boosting scikit-learn: RMSE 2.037,9 GBP, MAPE 6,77%, R² 0,9579 |
| Spark MLlib     | GBT: RMSE 2.451,7 GBP, MAPE 8,53%                                       |
| Graph analytics | Graph asosiasi berbasis lift, Louvain 4 komunitas, modularity 0,371     |
