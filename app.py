# -*- coding: utf-8 -*-
# coding=gbk
import io
import pandas as pd
import numpy as np
import base64
from flask import Response, Flask, render_template, session, request, redirect
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg, max
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import IsotonicRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

features = ["aged_70_older", "gdp_per_capita",
            "life_expectancy", "hospital_beds_per_thousand"]

df = ""
lastContinent = ""
lastBarColumn = ""
lastLocation = ""
lastLineColumn = ""
lastMonth = ""
lastTarget = ""
lastRegressionMethod = ""

dfTable = []
ddsTable = []
title = ""
plotUrl = ""

cnt = "0.0"
mae = 0.0
mse = 0.0
rmse = 0.0
r2 = 0.0
aged_70_older = 0.0
gdp_per_capita = 0.0
life_expectancy = 0.0
hospital_beds_per_thousand = 0.0

app = Flask(__name__)


@app.route("/")
def start():
    return render_template("index.html")


@app.route("/index", methods=["GET", "POST"])
def index():
    global df, lastContinent, lastBarColumn, lastLocation, lastLineColumn, lastMonth, lastTarget, lastRegressionMethod
    sc = SparkSession.builder.getOrCreate()
    spark = SparkSession(sc)
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    df = spark.read.format("csv").option("encoding", "Big5").option("header", "true").option(
        "inferSchema", "true").load("s3://s410985023/owid-covid-data.csv")
    dataFrame = df.toPandas().head(5)
    dds = df.toPandas().describe()
    dfTable = [dataFrame.to_html(classes="data")]
    ddsTable = [dds.to_html(classes="data")]

    if request.method == "POST":
        graph = request.form.get("graph")
        if graph == "Bar graph":
            continent = request.form.get("continent")
            column = request.form.get("barColumn")
            if lastContinent == continent and lastBarColumn == column:
                return render_template("index.html", graph=graph, dfTable=dfTable, ddsTable=ddsTable, title=title, plotUrl=plotUrl, isPress=True)

            lastContinent = continent
            lastBarColumn = column
            covid19Bar(continent, column)
            return render_template("index.html", graph=graph, dfTable=dfTable, ddsTable=ddsTable, title=title, plotUrl=plotUrl, isPress=True)
        elif graph == "Line graph":
            location = request.form.get("location")
            column = request.form.get("lineColumn")
            month = request.form.get("month")
            if lastLocation == location and lastLineColumn == column and lastMonth == month:
                return render_template("index.html", graph=graph, dfTable=dfTable, ddsTable=ddsTable, title=title, plotUrl=plotUrl, isPress=True)

            lastLocation = location
            lastLineColumn = column
            lastMonth = month
            covid19Line(location, column, month)
            return render_template("index.html", graph=graph, dfTable=dfTable, ddsTable=ddsTable, title=title, plotUrl=plotUrl, isPress=True)
        elif graph == "Regression graph":
            target = request.form.get("feature")
            regressionMethod = request.form.get("regressionMethod")
            if lastTarget == target and lastRegressionMethod == regressionMethod:
                return render_template("index.html", graph=graph, dfTable=dfTable, ddsTable=ddsTable, title=title, plotUrl=plotUrl, cnt=cnt, mae=mae, mse=mse,
                                       rmse=rmse, r2=r2, aged_70_older=aged_70_older, gdp_per_capita=gdp_per_capita, life_expectancy=life_expectancy,
                                       hospital_beds_per_thousand=hospital_beds_per_thousand, target=target, regressionMethod=regressionMethod, isPress=True)

            lastTarget = target
            lastRegressionMethod = regressionMethod
            covid19Reg(target, regressionMethod)
            return render_template("index.html", graph=graph, dfTable=dfTable, ddsTable=ddsTable, title=title, plotUrl=plotUrl, cnt=cnt, mae=mae,
                                   mse=mse, rmse=rmse, r2=r2, aged_70_older=aged_70_older, gdp_per_capita=gdp_per_capita, life_expectancy=life_expectancy,
                                   hospital_beds_per_thousand=hospital_beds_per_thousand, target=target, regressionMethod=regressionMethod, isPress=True)

    return render_template("index.html", dfTable=dfTable, ddsTable=ddsTable, isPress=False)


def covid19Bar(continent, column):
    global df, title, plotUrl
    plt.switch_backend("agg")
    img = io.BytesIO()

    dfBar = df
    dfBar = dfBar.filter(dfBar["continent"] == continent).filter(
        dfBar["location"].isNotNull()).filter(dfBar[column].isNotNull())
    dfPlot = dfBar.select("location", column).distinct().sort(
        column, ascending=False)

    x = dfPlot.toPandas()["location"].apply(str).tolist()
    y = dfPlot.toPandas()[column].values.tolist()

    plt.figure(figsize=(10, 25))
    plt.xlabel("loaction")
    plt.xticks(rotation=90)
    plt.ylabel(column)
    plt.title(continent+" "+column)
    plt.bar(x, y)
    plt.savefig(img, format="png")
    img.seek(0)

    title = continent + " " + column + " bar graph"
    plotUrl = base64.b64encode(img.getvalue()).decode()


def covid19Line(location, column, month):
    global df, title, plotUrl, a
    plt.switch_backend("agg")
    img = io.BytesIO()

    dfLine = df
    dfLine = dfLine.filter(dfLine["location"] == location).filter(dfLine["date"].isNotNull(
    )).filter(dfLine["date"].like(month+"%")).filter(dfLine[column].isNotNull())
    dfPlot = dfLine.select("date", column).sort(
        "date", ascending=True)

    x = dfPlot.toPandas()["date"].apply(str).str.slice(0, 10).tolist()
    y = dfPlot.toPandas()[column].values.tolist()

    plt.figure(figsize=(8, 10))
    plt.xlabel("date")
    plt.xticks(rotation=90)
    plt.ylabel(column)
    plt.title(location+" "+column)
    plt.plot(x, y, color='red')
    plt.savefig(img, format="png")
    img.seek(0)

    title = location + " " + column + "(" + month + ")" + " line graph"
    plotUrl = base64.b64encode(img.getvalue()).decode()


def covid19Reg(target, regressionMethod):
    global df, features, title, plotUrl
    global cnt, mae, mse, rmse, r2
    global aged_70_older, gdp_per_capita, life_expectancy, hospital_beds_per_thousand
    plt.switch_backend("agg")
    img = io.BytesIO()

    dfReg = df
    dfReg = dfReg.filter(dfReg[target].isNotNull())
    for feature in features:
        dfReg = dfReg.filter(dfReg[feature].isNotNull())

    dfReg = dfReg.groupBy("iso_code").agg(max(target).alias(target),
                                          max("aged_70_older").alias("aged_70_older"), max(
                                              "gdp_per_capita").alias("gdp_per_capita"),
                                          max("life_expectancy").alias(
                                              "life_expectancy"),
                                          max("hospital_beds_per_thousand").alias("hospital_beds_per_thousand"))
    cnt = dfReg.count()

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    output = assembler.transform(dfReg)
    trainData, testData = output.select(
        target, "features").randomSplit([0.7, 0.3])

    model = LinearRegression(featuresCol="features",
                             labelCol=target, predictionCol="prediction")
    if regressionMethod == "Linear regression":
        model = LinearRegression(
            featuresCol="features", labelCol=target, predictionCol="prediction")
    elif regressionMethod == "Decision tree regression":
        model = DecisionTreeRegressor(
            featuresCol="features", labelCol=target, predictionCol="prediction")
    elif regressionMethod == "Random forest regression":
        model = RandomForestRegressor(
            featuresCol="features", labelCol=target, predictionCol="prediction")
    elif regressionMethod == "Gradient-boosted tree regression":
        model = GBTRegressor(featuresCol="features",
                             labelCol=target, predictionCol="prediction")
    elif regressionMethod == "Isotonic regression":
        model = IsotonicRegression(
            featuresCol="features", labelCol=target, predictionCol="prediction")

    model = model.fit(trainData)
    predictions = model.transform(testData)
    if regressionMethod == "Linear regression":
        coefficients = model.coefficients
        aged_70_older = coefficients[0]
        gdp_per_capita = coefficients[1]
        life_expectancy = coefficients[2]
        hospital_beds_per_thousand = coefficients[3]

    evaluatorMAE = RegressionEvaluator(
        labelCol=target, predictionCol="prediction", metricName="mae")
    mae = evaluatorMAE.evaluate(predictions)
    evaluatorMSE = RegressionEvaluator(
        labelCol=target, predictionCol="prediction", metricName="mse")
    mse = evaluatorMSE.evaluate(predictions)
    evaluatorREMSE = RegressionEvaluator(
        labelCol=target, predictionCol="prediction", metricName="rmse")
    rmse = evaluatorREMSE.evaluate(predictions)
    evaluatorR2 = RegressionEvaluator(
        labelCol=target, predictionCol="prediction", metricName="r2")
    r2 = evaluatorR2.evaluate(predictions)

    x = predictions.select(target).toPandas()[target].values.tolist()
    y = predictions.select("prediction").toPandas()[
        "prediction"].values.tolist()

    plt.xlabel(target)
    plt.ylabel("prediction")
    plt.title(target + " prediction")
    plt.plot(x, x, color="red")
    plt.scatter(x, y)
    plt.savefig(img, format="png")
    img.seek(0)

    title = target + " regression analysis graph" + \
        "(" + regressionMethod + ")"
    plotUrl = base64.b64encode(img.getvalue()).decode()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
