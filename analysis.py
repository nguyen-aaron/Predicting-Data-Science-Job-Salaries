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
plt.title("Salary Distribution by Job Location (Top 10)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Heatmap

# experience_level_mapping = {'Entry': 1, 'Mid': 2, 'Senior': 3}
# df_pd['Experience_Level_Num'] = df_pd['experience_level'].map(experience_level_mapping)

# # Calculate the correlation between Experience Level and Salary
# correlation_matrix = df_pd[['Experience_Level_Num', 'salary']].corr()

# # Plotting the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap: Experience Level vs Salary')
# plt.show()
