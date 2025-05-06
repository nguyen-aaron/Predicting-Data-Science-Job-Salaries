from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create Spark session
spark = SparkSession.builder.appName("DataScienceSalaryPrediction").getOrCreate()

df = spark.read.csv("datascience_salaries.csv", header=True, inferSchema=True)

df.printSchema()

df.show(10)

print("Done")