package paristech

import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.SparkConf


    object Preprocessor {
      def main(args: Array[String]): Unit = {

        // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
        // On vous donne un exemple de setting quand même
        val conf = new SparkConf().setAll(Map(
          "spark.scheduler.mode" -> "FIFO",
          "spark.speculation" -> "false",
          "spark.reducer.maxSizeInFlight" -> "48m",
          "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
          "spark.kryoserializer.buffer.max" -> "1g",
          "spark.shuffle.file.buffer" -> "32k",
          "spark.default.parallelism" -> "12",
          "spark.sql.shuffle.partitions" -> "12"
        ))

        // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
        // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)

        val spark = SparkSession
          .builder
          .config(conf)
          .appName("TP Spark : Preprocessor")
          .getOrCreate()


        /** *****************************************************************************
          *
          * TP 2
          *
          *       - Charger un fichier csv dans un dataFrame
          *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
          *       - Sauver le dataframe au format parquet
          *
          * if problems with unimported modules => sbt plugins update
          *
          * *******************************************************************************/

        import spark.implicits._


        val df: DataFrame = spark
          .read
          .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
          .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
          .csv("data/train_clean.csv")

        println(s"Nombre de lignes : ${df.count}")
        println(s"Nombre de colonnes : ${df.columns.length}")

        val df3: DataFrame = df
          .withColumn("goal", $"goal".cast("Int"))
          .withColumn("deadline", $"deadline".cast("Int"))
          .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
          .withColumn("created_at", $"created_at".cast("Int"))
          .withColumn("launched_at", $"launched_at".cast("Int"))
          .withColumn("backers_count", $"backers_count".cast("Int"))
          .withColumn("final_status", $"final_status".cast("Int"))
          .dropDuplicates("deadline")
          .filter(!isnull($"state_changed_at"))
          .withColumn("country", when($"country" === "False", $"currency").otherwise($"country"))
          .filter(($"disable_communication" === "True") || ($"disable_communication" === "False"))
          .drop("disable_communication")
          .filter($"country" rlike ".{2}")
          .filter($"currency" rlike ".{3}")
          .drop("backers_count", "state_changed_at")
          .withColumn("days_campaign", datediff(from_unixtime($"deadline"), from_unixtime($"launched_at")))
          .withColumn("hours_prepa", (($"launched_at" - $"created_at") / 60).cast("Int"))
          .drop("launched_at", "deadline", "created_at")
          .withColumn("name", lower($"name"))
          .withColumn("desc", lower($"desc"))
          .withColumn("keywords", lower($"keywords"))
          .withColumn("text", concat($"name", lit(" "), $"desc", lit(" "), $"keywords"))
          .withColumn("days_campaign", when(isnull($"days_campaign"), -1).otherwise($"days_campaign"))
          .withColumn("hours_prepa", when(isnull($"hours_prepa"), -1).otherwise($"hours_prepa"))
          .withColumn("goal", when(isnull($"goal"), -1).otherwise($"goal"))
          .withColumn("country", when(isnull($"country"), " ").otherwise($"country"))
          .withColumn("currency", when(isnull($"currency"), " ").otherwise($"currency"))




        df3.write.mode(SaveMode.Overwrite).parquet("/home/jorge/Documents/Git/spark_project_kickstarter_2019_2020/cleanData.parquet")
        df3.show(50)

      }
    }




