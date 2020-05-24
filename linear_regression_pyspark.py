from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Customers').getOrCreate()
from pyspark.ml.regression import LinearRegression
dataset=spark.read.csv("gs://kundan-test12/ecommerce.csv",inferSchema=True,header=True)
dataset
dataset.show()
dataset.printSchema()
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=["Avg Session Length","Time on App","Time on Website","Length of Membership"],outputCol="Independent Features")
output=featureassembler.transform(dataset)
output.show()
output.select("Independent Features").show()
output.columns
finalized_data=output.select("Independent Features","Yearly Amount Spent")
finalized_data.show()
train_data,test_data=finalized_data.randomSplit([0.75,0.25])

regressor=LinearRegression(featuresCol='Independent Features', labelCol='Yearly Amount Spent')
regressor=regressor.fit(train_data)

regressor.coefficients
regressor.intercept
pred_results=regressor.evaluate(test_data)
pred_results.predictions.show(40)

