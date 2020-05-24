from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Customers').getOrCreate()
df=spark.read.csv("gs://kundan-test12/creditcard.csv",inferSchema=True,header=True)
df.printSchema()
len(df.columns)
df.select(["V"+str(x) for x in range(1,5)]).show(5)
#df_pandas=df.toPandas()
#df_pandas.head()

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
df.columns[1:-1]
from pyspark.sql.functions import col

for col_name in df.columns[1:-1]+["class"]:
    df=df.withColumn(col_name, col(col_name).cast("float"))

df=df.withColumnRenamed("class","label")
df.columns
vectorAssembler =VectorAssembler(inputCols=df.columns[1:-1],outputCol='features')
df_tr=vectorAssembler.transform(df)
df_tr =df_tr.select(['features','label'])
df_tr.show(3,truncate=False)
lr=LogisticRegression(maxIter=10,featuresCol='features',labelCol='label')
model=lr.fit(df_tr)
print(model.summary.areaUnderROC)
paramgrid=ParamGridBuilder().addGrid(lr.regParam, [.1,.01]).addGrid(lr.fitIntercept, [False,True])\
    .addGrid(lr.elasticNetParam, [0.0,.5,1.0]).build()

crossval= CrossValidator(estimator=lr,estimatorParamMaps=paramgrid,evaluator=BinaryClassificationEvaluator(),
numFolds=2)
cvmodel=crossval.fit(df_tr)
cvmodel.avgMetrics