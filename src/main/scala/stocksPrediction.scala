import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.time.{ZoneId, ZonedDateTime}

import com.cloudera.sparkts.models.ARIMA
import com.cloudera.sparkts.{DateTimeIndex, DayFrequency, TimeSeriesRDD}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

object stocksPrediction {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("SrocksPrediction")
      .getOrCreate()

    import spark.implicits._

    val DAYS = 5

    val amaznDf = spark
      .read
      .option("header", "true")
      .csv("data/amzn.csv")

    val amazn = amaznDf.select(amaznDf("\uFEFFDate").as("amznDate"), amaznDf("Close").as("closeAmazn"))

    val googDf: DataFrame = spark
      .read
      .option("header", "true")
      .csv("data/goog.csv")

    val goog = googDf.select(googDf("\uFEFFDate").as("googDate"), googDf("Close").as("closeGoog"))

    var yhooDf: DataFrame = spark
      .read
      .option("header", "true")
      .csv("data/yhoo.csv")

    val yhoo = yhooDf.select(yhooDf("\uFEFFDate").as("yhooDate"), yhooDf("Close").as("closeYhoo"))

    val data = amazn
      .join(goog, $"amznDate" === $"googDate").select($"amznDate", $"closeAmazn", $"closeGoog")
      .join(yhoo, $"amznDate" === $"yhooDate").select($"amznDate".as("date"), $"closeAmazn", $"closeGoog", $"closeYhoo")

    def toTimeStamp(str: String): Timestamp = {
      val format = new SimpleDateFormat("dd-MMMM-yy")
      new Timestamp(format.parse(str).getTime)
    }

    val toTime = spark.udf.register("toTime", toTimeStamp _)

    val formattedData = data
      .flatMap {
        row =>
          Array(
            (row.getString(row.fieldIndex("date")), "amzn", row.getString(row.fieldIndex("closeAmazn"))),
            (row.getString(row.fieldIndex("date")), "goog", row.getString(row.fieldIndex("closeGoog"))),
            (row.getString(row.fieldIndex("date")), "yhoo", row.getString(row.fieldIndex("closeYhoo")))
          )
      }.toDF("date", "symbol", "closingPrice")


    val finalDf = formattedData
      .withColumn("timestamp", toTime(formattedData("date")))
      .withColumn("price", formattedData("closingPrice").cast(DoubleType))
      .drop("date", "closingPrice")


    val minDate = finalDf.selectExpr("min(timestamp)").collect()(0).getTimestamp(0)
    val maxDate = finalDf.selectExpr("max(timestamp)").collect()(0).getTimestamp(0)

    val zone = ZoneId.systemDefault()

    val dtIndex = DateTimeIndex.uniformFromInterval(
      ZonedDateTime.of(minDate.toLocalDateTime, zone),
      ZonedDateTime.of(maxDate.toLocalDateTime, zone),
      new DayFrequency(1)
    )

    val tsRdd = TimeSeriesRDD.timeSeriesRDDFromObservations(dtIndex, finalDf, "timestamp", "symbol", "price")


    tsRdd.mapSeries { vector => {
      val newVec = new org.apache.spark.mllib.linalg.DenseVector(vector.toArray.map(x => if (x.equals(Double.NaN)) 0 else x))
      //      val newVecSize = newVec.size
      val arimaModel = ARIMA.fitModel(1, 0, 0, newVec)
      val forecasted = arimaModel.forecast(newVec, DAYS)
      //      val forecastedVecSize = forecasted.size

      new org.apache.spark.mllib.linalg.DenseVector(forecasted.toArray.slice(forecasted.size - (DAYS + 1), forecasted.size - 1))
    }
    }.toJavaRDD().saveAsTextFile("/tmp/stocksPredict")

  }
}
