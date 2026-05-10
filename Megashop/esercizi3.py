from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# creo la SparkSession
spark = SparkSession.builder \
    .appName("Esercizio 3") \
    .getOrCreate()

# leggo il file creato nell'esercizio 2
df = spark.read.parquet(
    "data_local/processed_sales"
)

# calcolo il fatturato totale per categoria
fatturato = df.groupBy("category") \
    .sum("amount")

# cambio nome alla colonna
fatturato = fatturato.withColumnRenamed(
    "sum(amount)",
    "totale"
)

# trasformo in pandas
pdf = fatturato.toPandas()

print(pdf)

# creo il grafico
plt.figure(figsize=(8, 5))

plt.bar(
    pdf["category"],
    pdf["totale"]
)

plt.title("Fatturato per categoria")

plt.xlabel("Categoria")

plt.ylabel("Totale vendite")

plt.xticks(rotation=45)

plt.tight_layout()

# salvo il grafico
plt.savefig("fatturato_per_categoria.png")

# mostro il grafico
plt.show()

spark.stop()
