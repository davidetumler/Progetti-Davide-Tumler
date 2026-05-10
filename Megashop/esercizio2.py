from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# creo la sessione Spark
spark = SparkSession.builder \
    .appName("Esercizio 2 Megashop") \
    .getOrCreate()

# leggo i file parquet
transazioni = spark.read.parquet("dataset/transazioni_parquet/*.parquet")
prodotti = spark.read.parquet("dataset/anagrafiche/products.parquet")
regioni = spark.read.parquet("dataset/anagrafiche/regions.parquet")

# faccio la join con i prodotti
df = transazioni.join(
    prodotti,
    on="product_id",
    how="left"
)

# faccio la join con le regioni
df = df.join(
    regioni,
    on="region_id",
    how="left"
)

# tengo solo le colonne richieste
df_finale = df.select(
    col("transaction_id"),
    col("region_name"),
    col("category"),
    col("amount"),
    col("year")
)

# controllo il risultato
print("DataFrame finale")
df_finale.show(10)

# salvo il risultato in parquet, diviso per anno
df_finale.write \
    .mode("overwrite") \
    .partitionBy("year") \
    .parquet("data_local/processed_sales")

print("File salvato nella cartella data_local/processed_sales")

spark.stop()