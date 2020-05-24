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
'''cols = ['age', 'job', 'marital', 'education',  'balance', 'housing', 'loan', 'contact', 'day', 'duration', 'campaign', 'pdays', 
'previous', 'poutcome', 'deposit', 'default']'''
#df1=df.toDF(*cols)
#https://www.youtube.com/watch?v=CHBp62khOlo
#https://www.youtube.com/watch?v=kQ1ZJAh62B0

#create categorical and numerical variable without label

df1 = df
df1.printSchema()
categorical_cols = [item[0] for item in df1.dtypes if  item[1].startswith("string")]
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

cols = df1.columns
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df1)
print("pipeline model is:",pipelineModel)
df = pipelineModel.transform(df1)
selectedCols = ['features'] + cols
df = df.select(selectedCols)
df.printSchema()


pd.DataFrame(df.take(5),columns=df.columns)






