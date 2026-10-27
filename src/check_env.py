"""Cek environment sebelum menjalankan pipeline: Java, Spark, Delta Lake, dan tulis-baca file."""
import platform
import shutil
import subprocess
import sys

from config import OUT, USE_DELTA, get_spark, read_table, write_table

print("Python      :", sys.version.split()[0], platform.system())
java = shutil.which("java")
print("Java        :", java or "TIDAK DITEMUKAN")
if java:
    v = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    print("              ", v)

spark = get_spark("check_env")
print("Spark       :", spark.version)
print("Delta Lake  :", "aktif" if USE_DELTA else "nonaktif (USE_DELTA=0), memakai Parquet")
df = spark.range(5).withColumnRenamed("id", "x")
write_table(df, OUT / "_env_check")
print("Tulis-baca  :", read_table(spark, OUT / "_env_check").count(), "baris (harus 5)")
spark.stop()
print("Environment siap.")
