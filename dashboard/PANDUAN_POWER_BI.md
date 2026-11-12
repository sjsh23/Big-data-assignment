# Panduan Dashboard Power BI (Final Project)

Panduan ini membangun dashboard dari data hasil pipeline. Bagian yang otomatis dan yang manual:

| Bagian | Cara |
|---|---|
| Menyiapkan data | Otomatis, `s09_export_dashboard.py` menulis CSV ke `dashboard/data/` |
| Menghubungkan Power BI ke CSV | Manual sekali saja (Get Data), setelah itu cukup Refresh |
| Relasi tabel, measure DAX, visual | Manual di Power BI Desktop, ikuti langkah 4 sampai 6 |
| Memperbarui angka | Jalankan ulang s09, lalu klik Refresh di Power BI |

Power BI tidak bisa dibuat otomatis dari script dengan cara yang stabil. File `.pbix` adalah format biner, jadi visualnya tetap disusun di Power BI Desktop. Yang otomatis adalah aliran datanya: Serving Layer dan hasil model ke CSV, lalu ke Power BI lewat Refresh.

## 1. Jalankan tahap s09

Dari folder project, dengan venv aktif:

```powershell
python run_pipeline.py --only s09
```

Tahap ini membaca tabel Delta di Curated Zone dan Serving Layer dengan Spark, lalu menggabungkannya dengan hasil s05 sampai s08 di `outputs/`. Waktu jalannya sekitar satu menit. Di akhir akan muncul baris "Cek cepat: MAPE GB scikit-learn 6.77%, 78.5% prediksi test set selisihnya maksimal 10%". Kalau angkanya berbeda, berhenti dulu dan cek ulang.

File yang dihasilkan di `dashboard/data/`:

| File | Isi | Peran di model |
|---|---|---|
| `dim_brand.csv` | 9 brand | Dimensi, penghubung slicer brand |
| `dim_listing.csv` | 97.712 listing Curated Zone, termasuk yang ditandai | Dimensi utama (atribut, status, rentang harga dan umur) |
| `fact_prediction.csv` | 19.959 prediksi test set, scikit-learn GB dan Spark GBT | Fakta akurasi model |
| `serving_price_by_brand_year.csv` | Harga dan mileage rata-rata per brand per tahun | Fakta agregat dari Serving Layer |
| `serving_damage_distribution.csv` | Jumlah listing dan gambar per jenis kerusakan | Fakta agregat dari Serving Layer |
| `model_comparison.csv` | Metrik test set 7 model | Tabel lepas |
| `feature_importance.csv` | Importance per fitur, dua metode | Tabel lepas |
| `graph_pairs.csv` | 196 pasangan label dengan jumlah gambar dan lift | Tabel lepas |
| `graph_nodes.csv` | 29 simpul dengan PageRank dan komunitas Louvain | Tabel lepas |
| `kpi_summary.csv` | Ringkasan quality gate dan struktur graph | Tabel lepas |

## 2. Atur locale sebelum import (wajib)

CSV memakai titik sebagai pemisah desimal (contoh `6.775`). Kalau Windows atau Power BI memakai locale Indonesia, titik dibaca sebagai pemisah ribuan, sehingga `6.775` berubah jadi 6775 tanpa pesan error.

Di Power BI Desktop: File > Options and settings > Options > **Current File > Regional Settings** > Locale for import = **English (United States)**. Lakukan ini sebelum import data pertama.

## 3. Import data

1. Home > Get Data > Text/CSV, pilih `dashboard/data/dim_listing.csv`, lalu klik **Transform Data** (bukan Load).
2. Ulangi untuk sembilan file lainnya lewat New Source > Text/CSV di jendela Power Query.
3. Cek tipe data di Power Query:
   - `car_id`, `price`, `mileage`, `year`, `age`, `images`, `jumlah`, `*_order`, `within_10pct_*`, `in_assoc_graph`, `is_best`, `community`: Whole Number
   - `ape_pct_*`, `abs_error_*`, `error_*`, `lift`, `importance`, `mpg`, `engineSize`, `RMSE`, `MAE`, `MAPE_pct`, `R2`, `pagerank_*`, `value`: Decimal Number
   - Kolom teks lainnya: Text
4. Close & Apply.

Kalau folder project dipindah, ubah path lewat Transform data > Data source settings > Change Source.

## 4. Model data (Model view)

Matikan dulu deteksi otomatis yang sering membuat relasi salah: File > Options > Current File > Data Load, hilangkan centang **Autodetect new relationships after data is loaded** dan **Auto date/time**. Hapus relasi lain yang terlanjur dibuat otomatis.

Buat relasi berikut. Ini star schema yang sama dengan model dimensional di subbab 4.2.3:

| Dari | Ke | Kardinalitas | Arah filter |
|---|---|---|---|
| `dim_brand[brand]` | `dim_listing[brand]` | 1 ke banyak | Single |
| `dim_brand[brand]` | `serving_price_by_brand_year[brand]` | 1 ke banyak | Single |
| `dim_listing[car_id]` | `fact_prediction[car_id]` | 1 ke 1 | Both (Power BI selalu memakai Both untuk relasi 1 ke 1) |

Atur urutan kategori: buka Table view (ikon tabel di sidebar kiri), pilih tabel `dim_listing` di panel Data, klik header kolom `price_band`, lalu di tab ribbon **Column tools** pilih Sort by column > `price_band_order`. Lakukan hal yang sama untuk `age_band` dengan `age_band_order`. Cara lain dari Model view: klik kolom `price_band` di kartu tabel, lalu di panel Properties buka Advanced > Sort by column.

Sembunyikan kolom bantu `price_band_order` dan `age_band_order` (klik kanan > Hide in report view) supaya tidak muncul saat menyusun visual.

Screenshot Model view ini untuk subbab 7.1.

## 5. Measure DAX

Buat tabel kosong untuk menampung measure: Home > Enter data, beri nama `_Measures`, lalu Load. Tambahkan measure berikut lewat New measure.

**Pasar (dim_listing dan Serving Layer)**

```DAX
Jumlah Listing = COUNTROWS ( dim_listing )
Listing Valid = CALCULATE ( [Jumlah Listing], dim_listing[status] = "Valid" )
Listing Ditandai = CALCULATE ( [Jumlah Listing], dim_listing[status] = "Ditandai" )
Persen Ditandai = DIVIDE ( [Listing Ditandai], [Jumlah Listing] )
Harga Median = CALCULATE ( MEDIAN ( dim_listing[price] ), dim_listing[status] = "Valid" )
Mileage Median = CALCULATE ( MEDIAN ( dim_listing[mileage] ), dim_listing[status] = "Valid" )
Harga Rata Serving =
    DIVIDE (
        SUMX ( serving_price_by_brand_year, serving_price_by_brand_year[harga_rata] * serving_price_by_brand_year[jumlah] ),
        SUM ( serving_price_by_brand_year[jumlah] )
    )
```

`Harga Rata Serving` adalah rata-rata tertimbang jumlah listing, supaya tahun dengan sedikit listing tidak menyetir angka saat beberapa tahun digabung.

**Akurasi model (fact_prediction)**

```DAX
Jumlah Prediksi = COUNTROWS ( fact_prediction )
MAPE GB = AVERAGE ( fact_prediction[ape_pct_sklearn] ) / 100
MAE GB = AVERAGE ( fact_prediction[abs_error_sklearn] )
RMSE GB = SQRT ( AVERAGEX ( fact_prediction, fact_prediction[error_sklearn] ^ 2 ) )
Prediksi Dalam 10 Persen = AVERAGE ( fact_prediction[within_10pct_sklearn] )
MAPE Spark GBT = AVERAGE ( fact_prediction[ape_pct_spark] ) / 100
```

Format `MAPE GB`, `MAPE Spark GBT`, `Persen Ditandai`, dan `Prediksi Dalam 10 Persen` sebagai Percentage dengan 1 desimal.

**Graph**

```DAX
Lift = AVERAGE ( graph_pairs[lift] )
Gambar Pasangan = SUM ( graph_pairs[images] )
```

Cek angka sebelum lanjut: tanpa filter apa pun, `MAPE GB` harus 6,8% (6,77%), `RMSE GB` sekitar 2.038, dan `Listing Valid` 97.411.

## 6. Halaman dashboard

Gunakan satu tema sederhana untuk semua halaman (View > Themes, misalnya tema bawaan berwarna biru), satu font, dan judul halaman berbentuk kalimat temuan, bukan sekadar nama topik. Letakkan slicer di posisi yang sama di setiap halaman.

### Halaman 1: Pasar

| Visual | Field | Pengaturan |
|---|---|---|
| Card (4 buah) | `Listing Valid`, `Harga Median`, `Mileage Median`, jumlah brand (`COUNTROWS(dim_brand)` sebagai measure) | |
| Line chart | X: `serving_price_by_brand_year[age]`, Y: `Harga Rata Serving`, Legend: `brand` | Filter visual: `jumlah` lebih besar atau sama dengan 20, supaya tahun dengan sedikit listing tidak jadi titik ekstrem. Sumbu X diurutkan naik |
| Bar chart | Y: `dim_listing[brand]`, X: `Harga Median` | Urutkan dari terbesar |
| Slicer | `dim_brand[brand]`, `dim_listing[fuelType]`, `dim_listing[transmission]` | Slicer fuelType dan transmission hanya memengaruhi visual dari dim_listing, karena Serving Layer hanya diagregasi per brand dan tahun |

### Halaman 2: Akurasi estimasi harga (insight Machine Learning)

| Visual | Field | Pengaturan |
|---|---|---|
| Card (4 buah) | `MAPE GB`, `Prediksi Dalam 10 Persen`, `MAE GB`, `Jumlah Prediksi` | |
| Clustered column | X: `dim_listing[price_band]`, Y: `MAPE GB` | Tooltip: `Jumlah Prediksi`, `MAE GB` |
| Bar chart | Y: `dim_listing[brand]`, X: `MAPE GB` | Urutkan dari terkecil |
| Scatter chart | X: `fact_prediction[actual_price]`, Y: `fact_prediction[pred_sklearn_gb]`, Values: `car_id` | Set X dan Y ke "Don't summarize". Power BI menampilkan sampel titik bila jumlahnya sangat banyak, itu normal |
| Slicer | `dim_brand[brand]`, `dim_listing[age_band]` | |

### Halaman 3: Faktor harga dan perbandingan model (insight Machine Learning)

| Visual | Field | Pengaturan |
|---|---|---|
| Bar chart | Y: `feature_importance[feature]`, X: `importance` | Filter visual: `method` = Permutation importance (scikit-learn GB), `kelompok` = Atribut kendaraan |
| Card atau tabel kecil | `feature_importance[kelompok]`, Sum of `importance` | Filter `method` yang sama. Menunjukkan total 29 label gambar mendekati nol |
| Clustered column | X: `model_comparison[Family]`, Y: `RMSE`, Legend: `Library` | Filter visual: `Family` bukan Ablation |
| Table | `model_label`, `RMSE`, `MAPE_pct`, `R2`, `fit_seconds` | Conditional formatting baris dengan `is_best` = 1 |

### Halaman 4: Pola kerusakan (insight Graph Analytics)

| Visual | Field | Pengaturan |
|---|---|---|
| Matrix (heatmap) | Rows: `graph_pairs[from_label]`, Columns: `to_label`, Values: `Lift` | Filter visual: `pair_type` = Damage-Part. Conditional formatting > Background color > Gradient dengan nilai tengah 1, warna netral di 1, warna berbeda di bawah dan di atas 1 |
| Bar chart | Y: `pair_label`, X: `Lift` | Filter visual: `in_assoc_graph` = 1, lalu Top N 10 berdasarkan `Lift` |
| Table | `graph_nodes[community_label]`, `name`, `type`, `jumlah_gambar` | Urutkan per komunitas. Ganti nama "Komunitas 1" dan seterusnya sesuai interpretasimu di subbab 6.2 kalau mau |
| Card (2 buah) | `kpi_summary[value]` dengan filter `metric` = "Graph frekuensi: modularity Louvain" dan "Graph asosiasi: modularity Louvain" | Format 3 desimal |

Power BI tidak punya visual network graph bawaan. Kalau ingin menampilkan graph-nya, sisipkan gambar `outputs/fig_graph_network.png` sebagai Image di halaman ini, dan sebutkan di laporan bahwa itu output s08.

### Halaman 5 (opsional): Kualitas data

| Visual | Field | Pengaturan |
|---|---|---|
| Card | `Persen Ditandai`, `Listing Ditandai` | |
| Bar chart | Y: `dim_listing[data_issue]`, X: `Jumlah Listing` | Filter visual: `status` = Ditandai |
| Bar chart | Y: `dim_listing[brand]`, X: `Persen Ditandai` | |

### Yang sebaiknya tidak dibuat

Jangan membuat visual harga rata-rata per jenis kerusakan atau per part. Pemasangan gambar ke listing dilakukan secara random, jadi visual itu tidak menunjukkan hubungan apa pun, tapi pembaca non-IT bisa menafsirkannya sebagai hubungan sungguhan.

## 7. Simpan dan publish

1. Simpan sebagai `dashboard/Dashboard_Final_Project.pbix`.
2. Home > Publish > pilih My workspace (login dengan akun BINUS).
3. Untuk link yang bisa dibuka dosen, coba File > Embed report > **Publish to web (public)**. Fitur ini bisa dimatikan oleh admin kampus. Kalau muncul pesan bahwa fitur tidak tersedia, alternatifnya:
   - Share report dari Power BI Service ke email dosen. Ini biasanya butuh lisensi Pro.
   - Kumpulkan file `.pbix`, lalu tunjukkan interaktivitasnya (slicer, tooltip, drill) di video presentasi.
4. Coba link-nya di jendela incognito sebelum dicantumkan di laporan.

Setiap kali pipeline dijalankan ulang: jalankan `python run_pipeline.py --only s09`, buka `.pbix`, lalu klik Refresh. Kalau sudah publish, upload ulang lewat Publish.

## 8. Bahan untuk Bab 7 laporan

**7.1 Proses Membangun Dashboard**, isinya:

- Alur data: Serving Layer dan hasil model, diekspor s09 ke CSV, lalu masuk Power Query dan model data Power BI. Sebutkan kenapa lewat CSV: Power BI Desktop tidak punya connector bawaan untuk tabel Delta di folder lokal.
- Pengaturan locale import dan alasannya.
- Model data star schema (screenshot Model view) dan kaitannya dengan subbab 4.2.3.
- Daftar measure utama dan definisinya.
- Rancangan tiap halaman dan pertanyaan bisnis yang dijawabnya.
- Cara publish dan link dashboard (atau alasan bila hanya `.pbix`).

**7.2 Interpretasi Bisnis**, untuk setiap insight (minimal 3, wajib ada dari ML dan graph):

- Screenshot halaman atau visual yang relevan.
- Apa yang terlihat, dengan angkanya.
- Arti bisnisnya dan keputusan yang bisa diambil oleh pemangku kepentingan di Tabel 3.
- Batasannya, misalnya pemasangan gambar sintetis atau asosiasi yang lemah.

Tutup 7.2 dengan satu paragraf yang mengaitkan semua insight kembali ke masalah di Bab 1.
