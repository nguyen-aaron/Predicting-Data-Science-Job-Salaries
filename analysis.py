from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
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

df = df.select(
    col("job_title"),
    col("job_type"),
    col("experience_level"),
    col("location"),
    col("salary_currency"),
    col("salary").cast("double")
).filter(col("salary_currency") == "USD")

print("Metadata of the dataset:")
print("Number of entries:\n", df.count())
print("Features:")
df.printSchema()
print("Metadata:\n")
df.describe().show()

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



# Histogram of salary distribution
pdf = df.select("salary").dropna().toPandas()

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(pdf["salary"], bins=30, kde=True)
plt.title("Salary Distribution")
plt.xlabel("Salary (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Select relevant features and filter USD
df = df.select(
    col("job_title"),
    col("experience_level"),
    col("location"),
    col("salary_currency"),
    col("salary").cast("double")
).filter(col("salary_currency") == "USD")


# Shirin Alapati

# Step 1: Create indexers for categorical features
indexers = [
    StringIndexer(inputCol="job_title", outputCol="job_title_index"),
    StringIndexer(inputCol="experience_level", outputCol="experience_level_index"),
    StringIndexer(inputCol="location", outputCol="location_index")
]

# Step 2: Assemble indexed features into a single vector
assembler = VectorAssembler(
    inputCols=["job_title_index", "experience_level_index", "location_index"],
    outputCol="features"
)

# Step 3: Define the Random Forest model
rf = RandomForestRegressor(featuresCol="features", labelCol="salary", maxBins=512)

# Step 4: Create a pipeline combining all stages
pipeline = Pipeline(stages=indexers + [assembler, rf])

# Step 2: Fit the pipeline model
model = pipeline.fit(df)

# Step 3: Extract and visualize feature importances
rf_model = model.stages[-1]
importances = rf_model.featureImportances.toArray()
features = ["job_title", "experience_level", "location"]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Spark MLlib Feature Importance for Salary Prediction")
plt.tight_layout()
plt.show()

# Nontrivial Plot 2: Hexbin Plot (Experience vs. Salary)

# Step 1: Convert to Pandas
df_hex = df.select("experience_level", "salary").dropna().toPandas()

# Step 2: Map experience level to numeric values
experience_map = {
    "Entry": 1,
    "Mid": 2,
    "Senior": 3,
    "Executive": 4
}
df_hex["experience_level_num"] = df_hex["experience_level"].map(experience_map)


# Step 3: Hexbin plot
plt.figure(figsize=(10, 6))
plt.hexbin(
    df_hex["experience_level_num"],
    df_hex["salary"],
    gridsize=15,
    cmap="viridis",
    mincnt=1
)
plt.colorbar(label='Job Count in Bin')
plt.xticks([1, 2, 3, 4], ["Entry", "Mid", "Senior", "Executive"])
plt.xlabel("Experience Level")
plt.ylabel("Salary (USD)")
plt.title("Hexbin Plot: Experience Level vs. Salary")
plt.tight_layout()
plt.show()

