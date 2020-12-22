import pyspark
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DateType, TimestampType, StringType
from pyspark.sql.functions import isnan, when, count, isnull, udf, concat, lit, to_timestamp, date_add, dayofmonth, rand
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor, FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel

import argparse
import os
from os import path, listdir
import numpy as np

##################
# Initial config #
##################

""" Function to set the spark context and sql context """
def init_config():
    #sc = pyspark.SparkContext('local[*]')
    sc = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "15g") \
        .appName('Air delay predictor') \
        .getOrCreate()

    sql_c = SQLContext(sc)

    # Trying to manage the output loggs
    sc.sparkContext.setLogLevel('ERROR')

    print("This application will try to learn to predict the arrival delay from data of fligths")

    return sc, sql_c

####################
# Loading the data #
####################

""" Function to load the data an manage the input arguments"""
def load_data(sql_c):
    # Define the parser options
    parser = argparse.ArgumentParser()
    parser.add_argument("path2data", help=" Path to a folder (between quotes) containing the data as csv files")

    models_group = parser.add_argument_group("Models", "Select a specific models, if none is selected linear regression will be applied")
    models_group.add_argument("--LR", action='store_true', help=" Use Linear Regression algorithm to predict the arrival delay of a flight")
    models_group.add_argument("--GLR", action='store_true', help=" Use Generalized Linear Regression algorithm to predict the arrival delay of a flight")
    models_group.add_argument("--RF", action='store_true', help=" Use Random Forest algorithm to predict the arrival delay of a flight")
    models_group.add_argument("--GBT", action='store_true', help=" Use Gradient Boost Tree algorithm to predict the arrival delay of a flight")
    models_group.add_argument("--FM", action='store_true', help=" Use Factorization Machines Regression algorithm to predict the arrival delay of a flight")

    parser.add_argument("--sample", help="Select just a fraction of the data, input between 0 and 1", type=float)
    args = parser.parse_args()

    # Check the user input about the location of the csv files
    if args.path2data:
        path2data = args.path2data
        if path.isdir(path2data):
            directory_files = listdir(path2data)
            if directory_files:
                if not any(".csv" in fi for fi in directory_files):
                    print("There is not any csv file in the given directory, this app only works with csv files")
                    exit()
                print(f"{len(directory_files)} csv files found")
            else: 
                print("There are not any files in the given directory")
                exit()
        else:
            print("Path does not exist or it is not a path")
            exit()

    # Load the data from the given directory
    initial_df = sql_c.read.csv(path2data+"/*.csv", header = True)

    if args.sample:
        try:
            float(args.sample)
        except ValueError:
            print ("sample is not a number, taking the whole data")
        
        if 0.001 <= args.sample  <= 1:
            print(f"Selecting the {args.sample * 100}% of the data")
            initial_df = initial_df.sample(fraction=args.sample)
        else:
            print ("sample is not between 0 and 1, taking the whole data")

    ml_models = []

    # Store the selected models
    if args.LR:
        ml_models.append("LR")
        print("Linear Regression selected")
    if args.GLR:
        ml_models.append("GLR")
        print("Generalized Linear Regression selected")
    if args.RF:
        ml_models.append("RF")
        print("Random Forest selected")
    if args.GBT:
        ml_models.append("GBT")
        print("Gradient Boost Tree selected")
    if args.FM:
        ml_models.append("FM")
        print("Factorization Machines Regression selected")

    # Default case:
    if not ml_models:
        ml_models.append("LR")
        print("No model selected, linear regression algorithm will be used")

    return initial_df, ml_models

#######################
# Processing the data #
#######################

""" Function to conver the timeformat in to total minutes from 00:00"""
def hourFormatter(value_string):
    # Add 0's in front of some time variables and convert the time in to minutes
    value_string = value_string.zfill(4) 
    value_int = int(value_string[:2])*60 + int(value_string[2:])
    return value_int

""" Function to prepare de dataset --> drop and transform some variables"""
def preprocess_data(initial_df):
    # Drop not allowed columns 
    banned_columns = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
    processed_df = initial_df.drop(*banned_columns)

    # Drop variables Cancelled and CancellationCode. Also drop cancelled rows
    processed_df = processed_df.withColumn("Cancelled", processed_df["Cancelled"].cast('boolean'))
    processed_df = processed_df.filter(processed_df.Cancelled == False)
    processed_df = processed_df.drop("CancellationCode", "Cancelled")

    # Drop variables TaxiOut, FlightNum, TailNum (categorical --> more info in plane-data.csv)
    processed_df = processed_df.drop("TaxiOut", "FlightNum", "TailNum")

    # Drop rows containing  na in DepTime, the rest will be erased when converted to int later 
    processed_df = processed_df.filter(processed_df["DepTime"] != "NA")

    # Fill "NA" values of the distance varible
    #   Split the dataframe so we can get a df with only NA values for variable distance
    df1 = processed_df.filter(processed_df.Distance == "NA").drop("Distance")    # nulls
    df2 = processed_df.filter(processed_df.Distance != "NA")    # not nulls

    #   Make a df with all the combinations of Orig, Dest, and distance
    location_distance_df = df2.select(df2.Origin, df2.Dest, df2.Distance).distinct()

    #   Join the missing values with the df of combinations or, dest and distance
    df1 = df1.join(location_distance_df, [df1.Origin == location_distance_df.Origin, df1.Dest == location_distance_df.Dest], how="left").drop(location_distance_df.Origin).drop(location_distance_df.Dest)

    #   Union the no "NA" part of the dataframe with the joined one
    processed_df = df1.union(df2)

    #   Mark those parts as deleteable
    df1.unpersist()
    df2.unpersist()
    location_distance_df.unpersist()

    #   Fill the rest with the avg
    processed_df = processed_df.withColumn("Distance", processed_df["Distance"].cast('int'))
    imputer = Imputer(strategy='mean', inputCol='Distance', outputCol='Distance')
    processed_df = imputer.fit(processed_df).transform(processed_df)

    # Clean the date and time variables and cast them to timestamps
    formatter = udf(hourFormatter)

    time_cols = ["DepTime", "CRSDepTime", "CRSArrTime"]
    for c in time_cols:
        processed_df = processed_df.withColumn(c, formatter(processed_df[c]))

    # Cast columns types to int
    for c in ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'ArrDelay', 'DepDelay', 'Distance']:
        processed_df = processed_df.withColumn(c, processed_df[c].cast('int'))

    # Drop rows with CRSElapsedTime negative --> does not make sense to have a flight with negative duration
    processed_df = processed_df.filter(processed_df["CRSElapsedTime"] >= 0)
    #processed_df = processed_df.filter(processed_df["Distance"] > 0)

    # Drop all the rows containing null values (if there are any left)
    for c in ["CRSElapsedTime", "ArrDelay", "DepDelay", "Origin", "Dest"]:
        processed_df = processed_df.filter(processed_df[c].isNotNull())

    # Drop categorical variables UniqueCarrier (carriers.csv), Origin (airports.csv) and Dest (airports.csv) 
    processed_df = processed_df.drop("Origin", "Dest")

    processed_df = processed_df.select("Year","Month","DayofMonth","DayOfWeek","DepTime","CRSDepTime","CRSArrTime","CRSElapsedTime","ArrDelay","DepDelay","Distance","UniqueCarrier")

    return processed_df

""" Final function to prepare the dataset --> encode some variables and scale the features"""
def df_transformer(processed_df, useFM):

    # Encode the categorical data (Carrier)
    car_indexer = StringIndexer(inputCol="UniqueCarrier", outputCol="carrier_idx")
    car_encoder = OneHotEncoder(dropLast=False, inputCol="carrier_idx", outputCol="carrier_enc")

    # Assembler to create a dense vector
    assembler = VectorAssembler(inputCols=["Year","Month","DayofMonth","DayOfWeek","DepTime","CRSDepTime","CRSArrTime","CRSElapsedTime","DepDelay","Distance", "carrier_enc"],
                                outputCol='features')

    # Scaler
    scaler = StandardScaler(inputCol='features', outputCol='features_scaled')

    # Pipeline
    pipeline = Pipeline(stages=[car_indexer, car_encoder, assembler, scaler])
    final_df = pipeline.fit(processed_df).transform(processed_df).select("features_scaled", "ArrDelay")
    final_df = final_df.withColumnRenamed("features_scaled","features")
    final_df = final_df.withColumnRenamed("ArrDelay","label")
    (train_df,test_df) = final_df.randomSplit([0.8,0.2]) # Split train / validation after CV
    print("Some info about the label column: ")
    final_df.describe(["label"]).show()

    if useFM:
        
        # Assembler to create a dense vector
        assembler = VectorAssembler(inputCols=["Year","Month","DayofMonth","DayOfWeek","DepTime","CRSDepTime","CRSArrTime","CRSElapsedTime","DepDelay","Distance"],
                                    outputCol='features')

        # Scaler
        scaler = StandardScaler(inputCol='features', outputCol='features_scaled')

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        final_df = pipeline.fit(processed_df).transform(processed_df).select("features_scaled", "ArrDelay")
        final_df = final_df.withColumnRenamed("features_scaled","features")
        final_df = final_df.withColumnRenamed("ArrDelay","label")
        (train_FM_df,test_FM_df) = final_df.randomSplit([0.8,0.2]) # Split train / validation after CV
        print("Some info about the label column: ")
        final_df.describe(["label"]).show()
        return train_df, test_df, train_FM_df, test_FM_df
    
    return train_df, test_df

######################
# Creating the model #
######################
""" Class to add some verbosity to the crossvalidation process"""
class CrossValidatorVerbose(CrossValidator):

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)

        eva = self.getOrDefault(self.evaluator)
        metricName = eva.getMetricName()

        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        h = 1.0 / nFolds

        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        metrics = [0.0] * numModels

        for i in range(nFolds):
            foldNum = i + 1
            #print("Comparing models on fold %d" % foldNum)

            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
            validation = df.filter(condition)
            train = df.filter(~condition)

            for j in range(numModels):
                paramMap = epm[j]
                model = est.fit(train, paramMap)
                # TODO: duplicate evaluator to take extra params from input
                metric = eva.evaluate(model.transform(validation, paramMap))
                metrics[j] += metric

                avgSoFar = metrics[j] / foldNum
                #print("params: %s\t%s: %f\tavg: %f" % (
                #   {param.name: val for (param, val) in paramMap.items()},
                #   metricName, metric, avgSoFar))

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)

        bestParams = epm[bestIndex]
        bestModel = est.fit(dataset, bestParams)
        avgMetrics = [m / nFolds for m in metrics]
        bestAvg = avgMetrics[bestIndex]
        #print("Training the data we obtain:")
        print("Best model:\nparams: %s\t%s: %f" % (
            {param.name: val for (param, val) in bestParams.items()},
            metricName, bestAvg))

        return self._copyValues(CrossValidatorModel(bestModel, avgMetrics))

""" Function to create and train the models selected by the user"""
def create_models(ml_models, train_df, test_df, train_FM_df, test_FM_df):
    # Declare evaluator for crossvalidation
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

    for mo in ml_models:    #, maxMemoryInMB=5000
        if mo == "LR":
            print("---- Linear Regression ----")
            model = LinearRegression(featuresCol="features", labelCol="label")
            paramGrid = ParamGridBuilder().addGrid(model.regParam, [0.1, 0.01]).addGrid(model.elasticNetParam, [0.1, 0.6]).build()

        elif mo == "GLR":
            print("---- Generalized linear Regression ----")
            model = GeneralizedLinearRegression(featuresCol="features", labelCol="label")
            paramGrid = ParamGridBuilder().addGrid(model.regParam, [0.1, 0.01]).build()

        elif mo == "RF":
            print("---- Random Forest ----")
            model = RandomForestRegressor(featuresCol="features", labelCol="label", maxMemoryInMB=5000)
            paramGrid = ParamGridBuilder().addGrid(model.maxDepth, [5, 10]).addGrid(model.numTrees, [10 ,20 ]).build()
            
        elif mo == "GBT":
            print("---- Gradient Boost Tree ----")
            model = GBTRegressor(featuresCol="features", labelCol="label", maxMemoryInMB=5000)
            paramGrid = ParamGridBuilder().addGrid(model.maxDepth, [5, 10]).build()

        elif mo == "FM":
            print("---- Factorization Machines Regression ----")
            model = FMRegressor(featuresCol="features", labelCol="label")
            paramGrid = ParamGridBuilder().addGrid(model.regParam, [0.5, 0.3, 0.1, 0.01]).build()

        else:
            print(f"{mo} no detected as a ml model")

        if mo != "FM":
            # Cross validation
            #cval = CrossValidatorVerbose(estimator=model,estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=4)
            cval = CrossValidator(estimator=model,estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=4)

            cvModel = cval.fit(train_df)

            model_evaluation(cvModel, test_df, "features", "label")
        else:
            # Cross validation
            #cval = CrossValidatorVerbose(estimator=model,estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=4)
            cval = CrossValidator(estimator=model,estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=4)

            cvModel = cval.fit(train_FM_df)

            model_evaluation(cvModel, test_FM_df, "features", "label")
        

########################
# Validating the model #
########################

""" Function to evaluate the results of the models """
def model_evaluation(cvModel, test_df, featuresC, labelC):
    predictions = cvModel.transform(test_df)
    predictions.select(featuresC,labelC,"prediction").show(5)

    evaluator1 = RegressionEvaluator(labelCol=labelC, predictionCol="prediction", metricName="rmse")
    evaluator2 = RegressionEvaluator(labelCol=labelC, predictionCol="prediction", metricName="mse")
    evaluator3 = RegressionEvaluator(labelCol=labelC, predictionCol="prediction", metricName="mae")

    rmse = evaluator1.evaluate(predictions)
    mse = evaluator2.evaluate(predictions)
    mae = evaluator3.evaluate(predictions)

    print(f"Root Mean Squared Error (RMSE) on validation data = %g" % rmse)
    print(f"Mean Squared Error (MSE) on validation data = %g" % mse)
    print(f"Mean Absolute Error (MAE) on validation data = %g" % mae)

########    
# Main #
########

def main():
    sc, sql_c = init_config()
    initial_df, ml_models = load_data(sql_c)
    processed_df = preprocess_data(initial_df)
    if "FM" in ml_models:
        (train_df, test_df, train_FM_df, test_FM_df) = df_transformer(processed_df, useFM=True)
        create_models(ml_models, train_df, test_df, train_FM_df, test_FM_df)
    else:
        train_df, test_df = df_transformer(processed_df, useFM=False)
        create_models(ml_models, train_df, test_df, None, None)


if __name__ == "__main__":
    main()