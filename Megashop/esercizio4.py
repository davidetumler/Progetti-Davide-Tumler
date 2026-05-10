from pyspark.sql import SparkSession
from pyspark.sql.types import *

# creo la SparkSession
spark = SparkSession.builder \
    .appName("Streaming") \
    .getOrCreate()

# definisco lo schema
schema = StructType([

    StructField("transaction_id", IntegerType()),
    StructField("product_id", IntegerType()),
    StructField("region_id", IntegerType()),
    StructField("amount", IntegerType()),
    StructField("year", IntegerType())

])

# leggo i file in streaming
stream_df = spark.readStream \
    .schema(schema) \
    .json("streaming_data")

# conto le transazioni per regione
conteggio = stream_df.groupBy(
    "region_id"
).count()

# stampo il risultato in console
query = conteggio.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()