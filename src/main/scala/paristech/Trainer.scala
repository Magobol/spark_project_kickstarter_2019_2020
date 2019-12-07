package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder, TrainValidationSplit}
// import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
// import org.apache.spark.ml.evaluation
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


import org.apache.spark.ml.param.ParamMap

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    import spark.implicits._

//    val df:DataFrame = spark.read.parquet("prepared_trainingset/")
    val df:DataFrame = spark.read.parquet("cleanData.parquet/")
      .filter(!isnull($"text"))
      .filter(!($"country" rlike "DE"))


    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    val cvModel: CountVectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("vect")
      .setMinDF(50)

    val idf = new IDF()
      .setInputCol(cvModel.getOutputCol)
      .setOutputCol("tfidf")

    val indexerCountry = new StringIndexer()
      .setInputCol("country")
      .setOutputCol("country_indexed")

    val indexerCurrency = new StringIndexer()
      .setInputCol("currency")
      .setOutputCol("currency_indexed")

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign","hours_prepa","goal","country_onehot","currency_onehot"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,remover,cvModel,idf,indexerCountry, indexerCurrency,encoder, assembler,lr))


    val Array(train,test) = df.randomSplit(Array[Double](0.9, 0.1),seed = 100)
    val size = (train.count,test.count)


    val model1 = pipeline.fit(train)
    val predictions = model1.transform(test)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val f1 = evaluator.evaluate(predictions)
//    println("Test set accuracy for Model 1 = " + f1)


    val grid = new ParamGridBuilder()
      .addGrid(lr.regParam,Array(10e-8,10e-6,10e-4,10e-2))
      .addGrid(cvModel.minDF,Array(35.0,55.0,75,95))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(grid)
      .setTrainRatio(0.7)

    val gridSearch = trainValidationSplit.fit(df)
    val gridSearchBestModel = gridSearch.bestModel

    val f1best = evaluator.evaluate(gridSearchBestModel.transform(test))


    val bestPipelineModel = gridSearchBestModel.asInstanceOf[PipelineModel]
    val stages = bestPipelineModel.stages
    val cvStage = stages(2).asInstanceOf[CountVectorizerModel]
    val lrStage = stages(8).asInstanceOf[LogisticRegressionModel]



    predictions.groupBy("final_status", "predictions").count.show()
    println("Train dataset size is : " + size._1)
    println("Test dataset size is : " + size._2)
    println("Test set accuracy for Model 1 = " + f1)
    println("Test set accuracy for the best model of the Grid Search is = " + f1best)
    println("Logistic Regression alpha optimal = " + lrStage.getRegParam)
    println("minDF optimal = " + cvStage.getMinDF)
//    println("Params for best model are : " + gridSearchBestModel.getParam(lr.getRegParam))

  }
}
