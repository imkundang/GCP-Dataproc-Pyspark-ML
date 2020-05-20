# Spark Session, Pipeline, Functions, and Metrics
from __future__ import print_function
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
#from pyspark.mllib.evaluation import MulticlassMetrics,MulticlassClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.types import IntegerType
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql.session import SparkSession
# Keras / Deep Learning
#import keras
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
#from keras import optimizers, regularizers
#from keras.optimizers import Adam

# Elephas for Deep Learning on Spark
#from elephas.ml_model import ElephasEstimator

# Spark Session

sc = SparkContext()
spark = SparkSession(sc)


# Load Data to Spark Dataframe
df = spark.read.format('csv').load('gs://kundan-test12/bank_kaggle.csv',header=True,
                    inferSchema=True)

# View Schema
df.printSchema()

df.show(5)

# Drop Unnessary Features (Day and Month)
df = df.drop('month','id')

labelIndexer = StringIndexer(inputCol="default", outputCol="indexedLabel").fit(df).transform(df)
print(labelIndexer.printSchema())




# Create a view so that Spark SQL queries can be run against the data.
labelIndexer.createOrReplaceTempView("df")

# As a precaution, run a query in Spark SQL to ensure no NULL values exist.
sql_query = """
SELECT *
from df
where default is not null
and age is not null
and duration is not null
and pdays is not null
and previous is not null
and campaign is not null
and balance is not null
and day is not null
"""
clean_data = spark.sql(sql_query)

#labels

def vector_from_inputs(r):
  return (r["indexedLabel"], Vectors.dense((r["age"]),(r["duration"]),(r["pdays"]),(r["previous"]),(r["campaign"]),(r["balance"]),(r["day"])))



# Create an input DataFrame for Spark ML using the above function.
training_data = clean_data.rdd.map(vector_from_inputs).toDF(["label", "features"])
#training_data = clean_data
training_data.cache()

#split the data into train and test
#labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(training_data).transform(training_data)
print(training_data.printSchema())

(trainingData, testData) = training_data.randomSplit([.7,.3])
print(trainingData.show(5))
print(testData.show(5))


rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
pipeline =Pipeline(stages=[rf])

model= pipeline.fit(trainingData)

predictions = model.transform(testData)
predictions.select("prediction", "label", "features", "probability", "rawPrediction").show(5)


evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="accuracy")

accuracy = evaluator.evaluate(predictions)

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector.
bucket = "kundan-test12"
spark.conf.set('temporaryGcsBucket', bucket)


# Saving the data to BigQuery
#temp_gcs_bucket= SPARK._jsc.hadoopConfiguration().get('fs.gs.system.bucket')

predictions.write.format('bigquery') \
  .option('table', 'bqtesting.rf_results').save()



print("Test error = %g" % (1.0 - accuracy))
