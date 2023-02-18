import org.apache.spark.sql.functions._
import scala.collection.parallel._
import org.apache.spark.sql.types.IntegerType

def reselect(qtr:String) = {
  val df = spark.read.parquet("s3a://loan-performance-data/full_data/qtr="+qtr)
  .select("PRODUCT_TYPE",
  "CURRENT_LOAN_DELINQUENCY_STATUS",
  "PROPERTY_TYPE",
  "ZERO_BALANCE_CODE",
  "BORROWER_CREDIT_SCORE_AT_ORIGINATION",
  "CURRENT_INTEREST_RATE",
  "FIRST_TIME_HOME_BUYER_INDICATOR",
  "LOAN_AGE",
  "LOAN_IDENTIFIER",
  "LOAN_PURPOSE",
  "METROPOLITAN_STATISTICAL_AREA",
  "MONTHLY_REPORTING",
  "NUMBER_OF_UNITS",
  "OCCUPANCY_TYPE",
  "ORIGINAL_DEBT_TO_INCOME_RATIO",
  "ORIGINAL_LOAN_TERM",
  "ZIP_CODE_SHORT")
  .filter(!$"BORROWER_CREDIT_SCORE_AT_ORIGINATION".isNull)
  .filter(!$"ORIGINAL_DEBT_TO_INCOME_RATIO".isNull)
  .filter($"FIRST_TIME_HOME_BUYER_INDICATOR" !== "U")
  .filter($"OCCUPANCY_TYPE" !== "U")
  .filter($"ORIGINAL_LOAN_TERM" === 360 || $"ORIGINAL_LOAN_TERM" === 180)
  .withColumn("PREPAID",
  when($"ZERO_BALANCE_CODE" === "01" || $"ZERO_BALANCE_CODE" === "16", 1)
  .otherwise(0)
  )
  .withColumn("NUMBER_OF_UNITS", $"NUMBER_OF_UNITS".cast(IntegerType))
  .withColumn("DEFAULT",
  when($"ZERO_BALANCE_CODE" === "01" || $"ZERO_BALANCE_CODE" === "16" || $"ZERO_BALANCE_CODE" === "", 0)
  .otherwise(1)
  )
  .withColumn("CURRENT_LOAN_DELINQUENCY_STATUS",
  when($"CURRENT_LOAN_DELINQUENCY_STATUS" === "X" || $"CURRENT_LOAN_DELINQUENCY_STATUS" === "", -1)
  .otherwise($"CURRENT_LOAN_DELINQUENCY_STATUS".cast(IntegerType))
  )
  .drop("ZERO_BALANCE_CODE")
  .withColumn( "SINGLE",
  when($"PROPERTY_TYPE"==="SF", 1)
  .otherwise(0)
  )
  .withColumn( "CONDO",
  when($"PROPERTY_TYPE"==="CO", 1)
  .otherwise(0)
  )
  .withColumn( "CO_OP",
  when($"PROPERTY_TYPE"==="CP", 1)
  .otherwise(0)
  )
  .withColumn( "MANUFACTURED",
  when($"PROPERTY_TYPE"==="MH", 1)
  .otherwise(0)
  )
  .withColumn( "PUD",
  when($"PROPERTY_TYPE"==="PU", 1)
  .otherwise(0)
  )
  .drop("PROPERTY_TYPE")
  .withColumn( "PRINCIPAL",
  when($"OCCUPANCY_TYPE"==="P", 1)
  .otherwise(0)
  )
  .withColumn( "SECOND",
  when($"OCCUPANCY_TYPE"==="S", 1)
  .otherwise(0)
  )
  .withColumn( "INVESTOR",
  when($"OCCUPANCY_TYPE"==="I", 1)
  .otherwise(0)
  )
  .drop("OCCUPANCY_TYPE")
  .withColumn( "PURCHASE",
  when($"LOAN_PURPOSE"==="P", 1)
  .otherwise(0)
  )
  .withColumn( "CASH-OUT",
  when($"LOAN_PURPOSE"==="C", 1)
  .otherwise(0)
  )
  .withColumn( "NO-CASH-OUT",
  when($"LOAN_PURPOSE"==="R", 1)
  .otherwise(0)
  )
  .withColumn( "REFINANCE",
  when($"LOAN_PURPOSE"==="U", 1)
  .otherwise(0)
  )
  .drop("LOAN_PURPOSE")
  .withColumn("FIXED_RATE", when($"PRODUCT_TYPE"==="FRM", 1).otherwise(0))
  .drop("PRODUCT_TYPE")
  .withColumn("FIRST_TIME", when($"FIRST_TIME_HOME_BUYER_INDICATOR"==="Y", 1).otherwise(0))
  .drop("FIRST_TIME_HOME_BUYER_INDICATOR")
  val unemploy = spark.read.parquet("s3a://loan-performance-data/other_data/msa_unemploy.parquet")
  .select("METROPOLITAN_STATISTICAL_AREA", "DATE", "UNEMPLOYMENT_RATE")
  val zillow = spark.read.parquet("s3a://loan-performance-data/other_data/zillow_data.parquet")
  val mortgage = spark.read.parquet("s3a://loan-performance-data/other_data/mortgage_rates.parquet")
  .drop("__index_level_0__")
  val joined = df.join(unemploy, df("METROPOLITAN_STATISTICAL_AREA") === unemploy("METROPOLITAN_STATISTICAL_AREA") && df("MONTHLY_REPORTING")===unemploy("DATE"), "inner")
  .drop("DATE")
  .join(zillow,  df("ZIP_CODE_SHORT") === zillow("ZIP_CODE_SHORT") && df("MONTHLY_REPORTING")===zillow("MONTHLY_REPORTING_PERIOD"), "inner")
  .drop("MONTHLY_REPORTING_PERIOD")
  .join(mortgage, df("MONTHLY_REPORTING") === mortgage("DATE"), "inner")
  .withColumn("DELTA_INTEREST",
  when($"ORIGINAL_LOAN_TERM" === 360, $"MORTGAGE30US"-$"CURRENT_INTEREST_RATE")
  .otherwise($"MORTGAGE15US"-$"CURRENT_INTEREST_RATE")
  )
  .drop("DATE")
  .drop("MORTGAGE15US")
  .drop("MORTGAGE30US")
  .drop("CURRENT_INTEREST_RATE")
  .drop("METROPOLITAN_STATISTICAL_AREA")
  .drop("ZIP_CODE_SHORT")
  .drop("ORIGINAL_LOAN_TERM")
  .coalesce(6)
  joined.write.parquet("s3a://loan-performance-data/more_features2/qtr="+qtr)
}
val qtrs = spark.read.parquet("s3a://loan-performance-data/full_data").select($"qtr").distinct.collect
val pars = qtrs.par
pars.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(20))
pars.foreach(x=>reselect(x.getString(0)))
