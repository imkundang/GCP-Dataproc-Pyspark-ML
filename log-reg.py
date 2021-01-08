from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml import Pipeline
spark = SparkSession.builder.appName('pipeline').getOrCreate()
spark.conf.set("spark.debug.maxToStringFields", 1000000)
bucket = "kundan-test12"
spark.conf.set('temporaryGcsBucket', bucket)

"Read file from google cloud storage to dataframe...include it in common functions"

def read_as_dataframe(filepath):
  filepath='gs://kundan-test12/bank_kaggle.csv'
  df = spark.read.format('csv').load(filepath,header=True,
                      inferSchema=True)

  return df
def drop_col(df):
  #df = read_as_dataframe(filepath)
  #df.printSchema()
  "include it under references file"
  col_drop=['id','contact','day','month','default'] #can put into refeences file
  df=df.drop(*col_drop)
  #df.printSchema()
  #cols=df.columns
  return df

#separate categorica and numeric columns
def cat_num_cols(df):
  df=read_as_dataframe(filepath)
  df = drop_col(df)
  cols=df.columns

  cat_cols=[x[0] for x in df.dtypes if x[1].startswith('string')][:-1]
  print('cat columns')
  print(cat_cols)
  num_cols=[x[0] for x in df.dtypes if x[1].startswith('int') | x[1].startswith("double") | x[1].startswith('float')]
  print(num_cols)
  print("number of categorical column is:",str(len(cat_cols)))
  print("number of numerical column is:",str(len(num_cols)))

  return cols, num_cols, cat_cols


#from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
#include it in common functions
def vector_assembler(cat_cols,num_cols):

  cols,num_cols,cat_cols=cat_num_cols(df)

  stages =[]
  for categoricalCol in cat_cols:
      print(categoricalCol)
      stringIndexer = StringIndexer(inputCol= categoricalCol,outputCol=categoricalCol + 'index')
      print(stringIndexer)
      print(stringIndexer.getOutputCol())
      encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "_catVec"]) 
      #print(OHencoder)
      #print(OHencoder.getOutputCol())
      stages +=   [stringIndexer, encoder]
  label_indexer=StringIndexer(inputCol='deposit',outputCol='label')
  stages+=[label_indexer]
  assemblerInputs = [c + "_catVec" for c in cat_cols] + num_cols
  Vectorassembler = VectorAssembler(inputCols=assemblerInputs,outputCol="features")
  stages += [Vectorassembler]
  return stages

#from pyspark.ml import Pipeline

def pipeline_model(stages,df):
  #stages=vector_assembler()

  pipeline=Pipeline(stages=stages)
  pipelineModel=pipeline.fit(df)
  df=pipelineModel.transform(df)
  selectedCols=['features', "label"] 
  modeling_cols=['features','label'] + cols
  feature_cols=['features']
  label_col=['label']
  df=df.select(*modeling_cols)
  #df.printSchema()
  #df.show(5,truncate=False)
  

  return df

def train_test_split(df):
  train,test=df.randomSplit([.7,.3],seed=1)
  #print('train data records:' + str(train.count()))
  #print('test data records:' + str(test.count()))

  return train, test

#from pyspark.ml.classification import LogisticRegression
def log_reg_model(train,test):
  train,test = train_test_split(df)
  
  #features,label=pipeline_model(stages)
  lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
  lrModel = lr.fit(train)
  lr_summary=lrModel.summary
  print(lr_summary.accuracy)
  bucket = "kundan-test12"
  spark.conf.set('temporaryGcsBucket', bucket)

  predictions = lrModel.transform(test)
  df_pred=predictions.select('label','features','probability','prediction')
  return df_pred

def load_df_as_table(df):

  bucket = "kundan-test12"
  spark.conf.set('temporaryGcsBucket', bucket)
  dataset_table='bqtesting.log_reg_pred99'
  
  df_pred.write.format('bigquery') \
    .option('table', dataset_table).save()


if __name__=="__main__":
  print('start of ml pipeline')
  print('reading the raw file as a datafram')
  filepath='gs://kundan-test12/bank_kaggle.csv'
  dataframe=read_as_dataframe(filepath)
  dataframe.show(5)
  print('drop irrelevant column')
  df=drop_col(dataframe)
  df.printSchema()
  print('numerical columns and categorical columns')
  cols,num_cols,cat_cols=cat_num_cols(df)
  print('num cols are :', num_cols)
  print('cat cols are:',cat_cols)
  print('cols are:', cols)
  print('one hot enoding and vetor assembling')
  stages=vector_assembler(cat_cols,num_cols)
  print(stages)
  print('transform dataframe')
  df=pipeline_model(stages,df)
  print(df.show(5,truncate=False))
  train,test=train_test_split(df)
  #print(train.shape,test.shape)
  print('test and prediction')
  df_pred=log_reg_model(train,test)
  df_pred.show()
  print('load predicted dataframe to bigquery table')
  load_df_as_table(df_pred)
  print('end of ml pipeline')

  





