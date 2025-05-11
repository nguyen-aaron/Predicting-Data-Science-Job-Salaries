from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create Spark session
spark = SparkSession.builder.appName("DataScienceSalaryPrediction").getOrCreate()

# Load the dataset
df = spark.read.csv("datascience_salaries.csv", header=True, inferSchema=True)

# Clean the dataset
df = df.dropna()  # Drop rows with null values
df = df.dropDuplicates()  # Drop duplicate rows

# Convert to Pandas DataFrame
df_pd = df.toPandas()

print("Metadata of the dataset:")
print("Number of entries:\n", df.count())
print("Features:")
df.printSchema()
print("Metadata:\n")
df.describe().show()



df = df.select(
    col("job_title"),
    col("job_type"),
    col("experience_level"),
    col("location"),
    col("salary_currency"),
    col("salary").cast("double")
).filter(col("salary_currency") == "USD")

#Violin/Box Plot

# Get the top 10 locations with the most job postings
top_locations = (
    df.groupBy("location")
    .count()
    .orderBy(col("count").desc())
    .limit(10)
    .rdd.flatMap(lambda x: [x[0]])
    .collect()
)

# Filter the DataFrame to include only the top 10 locations
df_filtered = df.filter(col("location").isin(top_locations))

# Convert to Pandas for Seaborn plot
pdf = df_filtered.select("location", "salary").toPandas()

# Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=pdf, x="location", y="salary")
plt.title("Salary Distribution by Most Popular Job Locations")
plt.xlabel("Location") 
plt.ylabel("Salary") 
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Convert to Pandas (if using PySpark)
pdf = df.select("salary").dropna().toPandas()

plt.figure(figsize=(10, 6))
sns.histplot(pdf["salary"], bins=30, kde=True)  # Add kde=True for a smooth curve
plt.title("Distribution of All Salaries")
plt.xlabel("Salary (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
