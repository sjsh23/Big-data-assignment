# Assignment II COMP8035041: Lakehouse Pipeline, Machine Learning, dan Graph Analytics

Prediksi harga mobil bekas UK dari dataset gabungan listing dan anotasi gambar kerusakan.

## Struktur

```
car damages images/                       dataset gambar Kaggle (Humans in the Loop, 2023)
Uk selling cars/                          CSV per brand Kaggle (Desai, 2020)
data/processed/used_cars_with_images.csv  hasil penggabungan Assignment I, input pipeline
data/raw/                                 Raw Zone (diisi s01)
lakehouse/curated/                        Curated Zone, Delta Lake (s02)
lakehouse/serving/                        Serving Layer (s03)
lakehouse/ml/                             dataset ML dan model Spark (s04, s06)
outputs/                                  metrik, tabel, gambar, log
src/                                      script penggabungan dataset dan script tiap tahap pipeline
dags/                                     DAG Airflow (untuk WSL, Docker, atau Linux)
run_pipeline.py                           orkestrator lokal
```

Dataset asli tidak disertakan di repository ini. Unduh dari Kaggle:
`adityadesai13/used-car-dataset-ford-and-mercedes` dan `humansintheloop/car-parts-and-car-damages`.

## Menjalankan

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\check_env.py
python run_pipeline.py
```

Detail setup Windows (Java 17, winutils) ada di runbook.
