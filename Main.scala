import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{LinearRegression, GeneralizedLinearRegression}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object Main {
  def main(args: Array[String]): Unit = {
    //Create Spark session
    val spark = SparkSession.builder().master("local[*]").appName("Airline").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    //Read csv files to data frame
    val df = spark.read.option("header", "true").option("InferSchema","true").csv("D:/tmp/data/20*.csv")

    //drop unnecessary columns and remove null values
    val df1 = df.drop("OP_CARRIER_FL_NUM","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME","CANCELLED","CANCELLATION_CODE","DIVERTED","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY","Unnamed: 27")
    val df1null = df1.na.drop("any", Seq("DEP_DELAY","ARR_DELAY"))

   //feature engineering
    //extract date Of Week and week of year from date, Hour from Scheduled departure time
    val df1sel = df1null.filter(col("ARR_DELAY") > -100 ).filter(col("ARR_DELAY") < 250)
    val df1f = df1sel.withColumn("dayOfWeek", dayofweek(col("FL_DATE")).cast("double"))
      .withColumn("HH", expr("floor(CRS_DEP_TIME / 100)").cast("double"))
      .withColumn("Wnum", weekofyear(col("FL_DATE")).cast("double"))
    println(s"Number of records: ${df1sel.count()}")
    println(s"Number of airports: ${df1sel.select("ORIGIN").distinct().count()}")
    println(s"Number of carriers: ${df1sel.select("OP_CARRIER").distinct().count()}")

    //Split data for training and test
    val Array(df2, dfT2) = df1f.randomSplit(Array(0.8,0.2), seed = 555)

    //String Indexer for Origin, Destination and Carrier
    val OriginIx = new StringIndexer().setInputCol("ORIGIN").setOutputCol("OriginIndex").setHandleInvalid("skip").fit(df2)
    val DestIx = new StringIndexer().setInputCol("DEST").setOutputCol("DestIndex").setHandleInvalid("skip").fit(df2)
    val CarrierIx = new StringIndexer().setInputCol("OP_CARRIER").setOutputCol("CarrierIndex").setHandleInvalid("skip")
    val Origin1 = new OneHotEncoder().setInputCol("OriginIndex").setOutputCol("OriginOneHot")
    val Dest1 = new OneHotEncoder().setInputCol("DestIndex").setOutputCol("DestOneHot")
    val Carrier1 = new OneHotEncoder().setInputCol("CarrierIndex").setOutputCol("CarrierOneHot")

    val dayOfW1 = new OneHotEncoder().setInputCol("dayOfWeek").setOutputCol("dayWOneHot")
    val HHW1 = new OneHotEncoder().setInputCol("HH").setOutputCol("HHOneHot")
    val Wnum1 = new OneHotEncoder().setInputCol("Wnum").setOutputCol("WnumOneHot")

    println("Combine Features")
    //Combine features into a feature vector
    val featSpace = Array("DEP_DELAY", "OriginOneHot","DestOneHot", "CarrierOneHot","DISTANCE")
    val featTime = Array("DEP_DELAY", "WnumOneHot", "dayWOneHot","HHOneHot")
    val featCol = Array( "DEP_DELAY", "OriginOneHot","DestOneHot", "CarrierOneHot","DISTANCE", "WnumOneHot", "dayWOneHot","HHOneHot")
    val asrSpace = new VectorAssembler().setInputCols(featSpace).setOutputCol("Features")
    val asrTime = new VectorAssembler().setInputCols(featTime).setOutputCol("Features")
    val assembler = new VectorAssembler().setInputCols(featCol).setOutputCol("Features")

     println("Define model")
    //Define the model
    val lr = new LinearRegression().setLabelCol("ARR_DELAY").setFeaturesCol("Features").setRegParam(0.01)
    val gr = new GeneralizedLinearRegression().setFamily("gaussian").setLabelCol("ARR_DELAY").setFeaturesCol("Features")
    //create pipeline for stages
    val pipeSpace = new Pipeline().setStages(Array(OriginIx, Origin1,  DestIx, Dest1, CarrierIx, Carrier1, asrSpace))
    val pipeTime = new Pipeline().setStages(Array(Wnum1, dayOfW1, HHW1, asrTime))
    val pipeline = new Pipeline().setStages(Array(OriginIx, Origin1,  DestIx, Dest1, CarrierIx, Carrier1, Wnum1, dayOfW1, HHW1, assembler))

    //Training
    println("Training Space")
    val trainSpace = pipeSpace.fit(df2).transform(df2)
    val testSpace = pipeSpace.fit(dfT2).transform(dfT2)
    val lrModelS = lr.fit(trainSpace)
    val grModelS = gr.fit(trainSpace)

    println("Training Time")
    val trainTime = pipeTime.fit(df2).transform(df2)
    val testTime = pipeTime.fit(dfT2).transform(dfT2)
    val lrModelT = lr.fit(trainTime)
    val grModelT = gr.fit(trainTime)

    println("Training Space + Time")
    val trainingdata = pipeline.fit(df2).transform(df2)
    val testdata = pipeline.fit(dfT2).transform(dfT2)
    val lrModel = lr.fit(trainingdata)
    val grModel = gr.fit(trainingdata)

    //prediction
    println("Prediction")
    val PredSlr = lrModelS.transform(testSpace)
    val PredSgr = grModelS.transform(testSpace)
    val PredTlr = lrModelT.transform(testTime)
    val PredTgr = grModelT.transform(testTime)
    val Predlr = lrModel.transform(testdata)
    val Predgr = grModel.transform(testdata)


    println("Evaluation - ARR_DELAY in {-100, +250}")
    val evaluator = new RegressionEvaluator().setLabelCol("ARR_DELAY").setPredictionCol("prediction").setMetricName("rmse")
    val evalMetric = Array("rmse", "mae", "r2")
    for (eval <- evalMetric) {
      evaluator.setMetricName(eval)
      val lrevalueS = evaluator.evaluate(PredSlr)
      println(s"Prediction Space lr : $eval: $lrevalueS")
      val grevalueS = evaluator.evaluate(PredSgr)
      println(s"Prediction Space gr : $eval: $grevalueS")
      val lrevalueT = evaluator.evaluate(PredTlr)
      println(s"Prediction Time lr : $eval: $lrevalueT")
      val grevalueT = evaluator.evaluate(PredTgr)
      println(s"Prediction Time gr : $eval: $grevalueT")
      val lrevalue = evaluator.evaluate(Predlr)
      println(s"Prediction Space + Time lr : $eval: $lrevalue")
      val grevalue = evaluator.evaluate(Predgr)
      println(s"Prediction Space + Time gr : $eval: $grevalue")
    }

    spark.stop()
  }
}