from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)

matches_with_date = matches_clean.select("id", "date")

clf_base = (
    match_features
    .join(matches_with_date, on="id", how="left")
    .withColumn("toss_is_team1", F.when(F.col("toss_winner") == F.col("team1"), 1.0).otherwise(0.0))
    .withColumn("toss_bat", F.when(F.col("toss_decision") == "bat", 1.0).otherwise(0.0))
    .withColumn("label", F.col("team1_win").cast("double"))
)

t1_hist = clf_base.select(
    F.col("id"), F.col("date"), F.col("team1").alias("team"),
    F.when(F.col("winner") == F.col("team1"), 1.0).otherwise(0.0).alias("won")
)
t2_hist = clf_base.select(
    F.col("id"), F.col("date"), F.col("team2").alias("team"),
    F.when(F.col("winner") == F.col("team2"), 1.0).otherwise(0.0).alias("won")
)

win_history = (
    t1_hist.unionByName(t2_hist)
    .withColumn(
        "prior_win_rate",
        F.avg("won").over(
            Window.partitionBy("team").orderBy("date", "id").rowsBetween(Window.unboundedPreceding, -1)
        )
    )
)

match_clf_df = (
    clf_base
    .join(
        win_history.select(F.col("id"), F.col("team").alias("team1"), F.col("prior_win_rate").alias("t1_prior_win_rate")),
        on=["id", "team1"], how="left"
    )
    .join(
        win_history.select(F.col("id"), F.col("team").alias("team2"), F.col("prior_win_rate").alias("t2_prior_win_rate")),
        on=["id", "team2"], how="left"
    )
    .withColumn("t1_prior_win_rate", F.col("t1_prior_win_rate").cast("double"))
    .withColumn("t2_prior_win_rate", F.col("t2_prior_win_rate").cast("double"))
)

imputer = Imputer(
    inputCols=["t1_prior_win_rate", "t2_prior_win_rate"],
    outputCols=["t1_win_rate_imp", "t2_win_rate_imp"],
    strategy="mean"
)

team1_indexer = StringIndexer(inputCol="team1", outputCol="team1_idx", handleInvalid="keep")
team2_indexer = StringIndexer(inputCol="team2", outputCol="team2_idx", handleInvalid="keep")
venue_indexer = StringIndexer(inputCol="venue", outputCol="venue_idx", handleInvalid="keep")
toss_dec_indexer = StringIndexer(inputCol="toss_decision", outputCol="toss_dec_idx", handleInvalid="keep")

encoder = OneHotEncoder(
    inputCols=["team1_idx", "team2_idx", "venue_idx", "toss_dec_idx"],
    outputCols=["team1_vec", "team2_vec", "venue_vec", "toss_dec_vec"]
)

assembler = VectorAssembler(
    inputCols=[
        "toss_is_team1", "toss_bat", "t1_win_rate_imp", "t2_win_rate_imp",
        "team1_vec", "team2_vec", "venue_vec", "toss_dec_vec"
    ],
    outputCol="raw_features",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=False
)



train_clf, test_clf = match_clf_df.randomSplit([0.8, 0.2], seed=42)


auc_eval = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

acc_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)



classifiers = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features", labelCol="label", maxIter=30, regParam=0.05
    ),
    "Decision Tree": DecisionTreeClassifier(
        featuresCol="features", labelCol="label", maxDepth=5, seed=42
    ),
    "Random Forest": RandomForestClassifier(
        featuresCol="features", labelCol="label", numTrees=50, maxDepth=6, seed=42
    )
}

for model_name, clf in classifiers.items():
    pipeline = Pipeline(stages=[
        imputer, team1_indexer, team2_indexer, venue_indexer, toss_dec_indexer,
        encoder, assembler, scaler, clf
    ])
    pipeline_model = pipeline.fit(train_clf)
    predictions = pipeline_model.transform(test_clf)

    print(f"=== {model_name} ===")
    print(f"ROC-AUC : {auc_eval.evaluate(predictions):.4f}")
    print(f"Accuracy: {acc_eval.evaluate(predictions):.4f}")
    print(f"F1-Score: {f1_eval.evaluate(predictions):.4f}")
    predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()



from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression

# Cache datasets
train_clf = train_clf.cache()
test_clf = test_clf.cache()

# Materialize cache
train_clf.count()

# Logistic Regression
lr_tune = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=10
)

# Pipeline
lr_pipeline = Pipeline(stages=[
    imputer,
    team1_indexer,
    team2_indexer,
    venue_indexer,
    toss_dec_indexer,
    encoder,
    assembler,
    scaler,
    lr_tune
])

# Smaller parameter grid
param_grid = (
    ParamGridBuilder()
    .addGrid(lr_tune.regParam, [0.01, 0.1])
    .build()
)

# Faster Cross Validation
cv = CrossValidator(
    estimator=lr_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=auc_eval,
    numFolds=2,
    seed=42,
    parallelism=2
)

# Train model
cv_model = cv.fit(train_clf)

# Predictions
tuned_preds = cv_model.transform(test_clf).cache()

# Evaluation
print("=== Tuned Logistic Regression ===")

print(f"Tuned Test ROC-AUC: {auc_eval.evaluate(tuned_preds):.4f}")

print(f"Tuned Test Accuracy: {acc_eval.evaluate(tuned_preds):.4f}")

print(f"Tuned Test F1-Score: {f1_eval.evaluate(tuned_preds):.4f}")

# Save model
model_path = "/tmp/cricket_match_winner_model"

cv_model.bestModel.write().overwrite().save(model_path)

# Reload model
loaded_pipeline_model = PipelineModel.load(model_path)

# Test loaded model
loaded_pipeline_model.transform(
    test_clf.limit(5)
).select(
    "id",
    "team1",
    "team2",
    "label",
    "probability",
    "prediction"
).show(5, truncate=False)


RELOADING THE PIPELINE MODEL

model_path = "/tmp/cricket_match_winner_model"
cv_model.bestModel.write().overwrite().save(model_path)
loaded_pipeline_model = PipelineModel.load(model_path)
loaded_pipeline_model.transform(test_clf).select("id", "team1", "team2", "label", "probability", "prediction").show(5)

#Predicting player_runs using prior career average runs, prior strike rate, and
#balls_faced from batter_features joined with match dates.

batter_with_context = (
    batter_features
    .join(matches_clean.select(F.col("id").alias("match_id"), "date", "venue"), on="match_id", how="inner")
    .withColumn("player_runs", F.col("player_runs").cast("double"))
    .withColumn("balls_faced", F.col("balls_faced").cast("double"))
    .withColumn("strike_rate", F.col("strike_rate").cast("double"))
)

batter_win = Window.partitionBy("batter").orderBy("date", "match_id").rowsBetween(Window.unboundedPreceding, -1)

reg_df = (
    batter_with_context
    .withColumn("prior_avg_runs", F.avg("player_runs").over(batter_win))
    .withColumn("prior_strike_rate", F.avg("strike_rate").over(batter_win))
    .filter(F.col("balls_faced") >= 5)
)

reg_imputer = Imputer(
    inputCols=["prior_avg_runs", "prior_strike_rate"],
    outputCols=["prior_avg_runs_imp", "prior_strike_rate_imp"],
    strategy="mean"
)

reg_venue_idx = StringIndexer(inputCol="venue", outputCol="venue_idx", handleInvalid="keep")
reg_venue_ohe = OneHotEncoder(inputCols=["venue_idx"], outputCols=["venue_vec"])
reg_assembler = VectorAssembler(
    inputCols=["balls_faced", "prior_avg_runs_imp", "prior_strike_rate_imp", "venue_vec"],
    outputCol="features",
    handleInvalid="keep"
)

lr_regressor = LinearRegression(
    featuresCol="features", labelCol="player_runs", maxIter=30, regParam=0.1
)

train_reg, test_reg = reg_df.randomSplit([0.8, 0.2], seed=42)
reg_pipeline = Pipeline(stages=[reg_imputer, reg_venue_idx, reg_venue_ohe, reg_assembler, lr_regressor])
reg_model = reg_pipeline.fit(train_reg)
reg_predictions = reg_model.transform(test_reg)

rmse_eval = RegressionEvaluator(labelCol="player_runs", predictionCol="prediction", metricName="rmse")
mae_eval = RegressionEvaluator(labelCol="player_runs", predictionCol="prediction", metricName="mae")
r2_eval = RegressionEvaluator(labelCol="player_runs", predictionCol="prediction", metricName="r2")

print("=== Linear Regression: Batter Runs Prediction ===")
print(f"RMSE: {rmse_eval.evaluate(reg_predictions):.2f}")
print(f"MAE : {mae_eval.evaluate(reg_predictions):.2f}")
print(f"R²  : {r2_eval.evaluate(reg_predictions):.4f}")

reg_predictions.select(
    "batter", "balls_faced",
    F.round("prior_avg_runs_imp", 1).alias("prior_avg_runs"),
    F.col("player_runs").alias("actual_runs"),
    F.round("prediction", 1).alias("predicted_runs")
).show(10, truncate=False)
