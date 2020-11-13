from pyspark import SQLContext, SparkContext
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql. functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.regression import LinearRegression, SparkContext


sc = SparkContext()
sqlContext = SQLContext(sc)

feature_columns = ['txCount','marketcap','exchangeVolume','generatedCoins','fees','activeAddresses','medianTxValue', 'medianFee', 'averageDifficulty', 'paymentCount','blockSize','blockCount']

datetoint = udf(lambda val: int(val.strftime("%s")), IntegerType())

data_frame = sqlContext.read.load("etc.csv", format = "csv", sep = ",", inferSchema= "true",header="true")

data_frame = data_frame.withColumn("time", datetoint("date"))
data_frame = data_frame.drop('date')
assembler = VectorAssembler(inputCols=feature_columns, outputCol="All_features")

pipeline = Pipeline(stages=[assembler])
model = pipeline.fit(data_frame)
df = model.transform(data_frame)

df = df.drop('txCount','marketcap','exchangeVolume','generatedCoins','fees','activeAddresses','medianTxValue', 'medianFee', 'averageDifficulty', 'paymentCount','blockSize','blockCount')
training_data, testing_data = df.randomSplit([0.7, 0.3])

lr = LinearRegression(featuresCol="All_features", labelCol="price")
trainedModel = lr.fit(training_data)

evaluation = trainedModel.evaluate(testing_data)
print("Mean Absolute Error:{}".format(evaluation.meanAbsoluteError))
print("Root Mean Square Error:{}".format(evaluation.rootMeanSquaredError))
print("Co-efficient of Determination: {}".format(evaluation.r2))

predictions = trainedModel.transform(testing_data)
predictions = predictions.orderBy("prediction", ascending=False)
predictions.select(predictions.columns).show()


# prediction made for dataset past 30 june 2019

data_frame = sqlContext.read.load("etc-predictor.csv", format = "csv", sep = ",", inferSchema= "true",header="true")

data_frame = data_frame.withColumn("time", datetoint("date"))
data_frame = data_frame.drop('date')

# assembler = VectorAssembler(inputCols=feature_columns, outputCol="All_features")
df = assembler.transform(data_frame)
df = df.select("All_features", "price")
evaluation = trainedModel.evaluate(df)

print("Mean Absolute Error:{}".format(evaluation.meanAbsoluteError))
print("Root Mean Square Error:{}".format(evaluation.rootMeanSquaredError))
print("Co-efficient of Determination: {}".format(evaluation.r2))

predictions = trainedModel.transform(df)
predictions = predictions.orderBy("prediction", ascending=False)
predictions.select(predictions.columns).show()
