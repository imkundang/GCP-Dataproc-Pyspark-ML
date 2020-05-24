from pyspark.sql import SparkSession
import pandas as pd
spark = SparkSession.builder.appName('pipeline').getOrCreate()
spark.conf.set("spark.debug.maxToStringFields", 10000)
df = spark.read.format('csv').load('gs://kundan-test12/bank_kaggle.csv',header=True,
                    inferSchema=True)
df.printSchema()
columns_to_drop = ['id', 'month']
df = df.drop('id')
df = df.drop('month')
df.printSchema()
#removing response attribute by subsetting column
cols = ['age', 'job', 'marital', 'education',  'balance', 'housing', 'loan', 'contact', 'day', 'duration', 'campaign', 'pdays', 
'previous', 'poutcome', 'deposit', 'default']
df1=df.select(*cols)
print("df1 schema:",df1.printSchema())
#https://www.youtube.com/watch?v=CHBp62khOlo
#https://www.youtube.com/watch?v=kQ1ZJAh62B0

#create categorical and numerical variable without label

categorical_cols = [item[0] for item in df1.dtypes if  item[1].startswith("string")][:-1]
print(categorical_cols)
numerical_cols = [item[0] for item in df1.dtypes if  item[1].startswith("int") | item[1].startswith("double") | item[1].startswith('float')]
print(numerical_cols)

print("number of categorical column is:",str(len(categorical_cols)))
print("number of numerical column is:",str(len(numerical_cols)))


from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler

stages =[]
for categoricalCol in categorical_cols:
    print(categoricalCol)
    stringIndexer = StringIndexer(inputCol= categoricalCol,outputCol=categoricalCol + 'index')
    print(stringIndexer.getOutputCol())
    OHencoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "_catVec"]) 
    print(OHencoder)
    #print(OHencoder.getOutputCol())
    stages +=   [stringIndexer, OHencoder]
assemblerInputs = [c + "_catVec" for c in categorical_cols] + numerical_cols
Vectorassembler = VectorAssembler(inputCols=assemblerInputs,outputCol="features")
stages += [Vectorassembler]
print(stages)

#using pyspark ml pipeline apply all the stages into it

from pyspark.ml import Pipeline
colsub = df1.columns
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df1)
print("pipeline model is:",pipelineModel)
df1 = pipelineModel.transform(df1)
selectedCols = ['features'] + colsub
df1 = df1.select(selectedCols)
print("df1 schema after pipeline processing:",df1.printSchema())
Col_drop= ['age', 'job', 'marital', 'education',  'balance', 'housing', 'loan', 'contact', 'day', 'duration', 'campaign', 'pdays', 
'previous', 'poutcome', 'deposit']
df1 = df1.drop(*Col_drop)
df2 = StringIndexer(inputCol="default", outputCol="indexedLabel").fit(df1).transform(df1)

print("df2 after dropping dupliate cols:",df1.printSchema())
df2.show(3,truncate=False)
df3=df2.drop('default')

#df_final=pd.DataFrame(df3.take(5),columns=df3.columns)

#random forest classifier pipeline using train test split
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql.functions import rand



(trainingData, testData) = df3.randomSplit([.7,.3])
print(trainingData.show(5))
print(testData.show(5))


rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)
pipeline =Pipeline(stages=[rf])

model= pipeline.fit(trainingData)

predictions = model.transform(testData)
pred=predictions.select("prediction", "indexedLabel", "features", "probability", "rawPrediction").show(5)


evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",predictionCol="prediction",metricName="accuracy")

accuracy = evaluator.evaluate(predictions)

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector.
bucket = "kundan-test12"
spark.conf.set('temporaryGcsBucket', bucket)


# Saving the data to BigQuery
#temp_gcs_bucket= SPARK._jsc.hadoopConfiguration().get('fs.gs.system.bucket')

predictions.write.format('bigquery') \
  .option('table', 'bqtesting.rf_results5').save()



print("Test error = %g" % (1.0 - accuracy))






