import os
os.makedirs("outputs", exist_ok=True)
## Dataset Understanding
import pyspark
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

print("Libraries imported successfully")

spark = (SparkSession.builder
         .appName("Cricket_Intelligence_Project")
         .master("local[*]")
         .getOrCreate())

print("Spark Version:", spark.version)
print("Spark Session Created Successfully")

matches = spark.read.csv(
    "data/matches.csv",
    header=True,
    inferSchema=True
)

deliveries = spark.read.csv(
    "data/deliveries.csv",
    header=True,
    inferSchema=True
)

print("Matches Dataset")
matches.show(5)

print("Deliveries Dataset")
deliveries.show(5)

print("Matches Schema")
matches.printSchema()

print("Deliveries Schema")
deliveries.printSchema()

print("Total Matches:", matches.count())
print("Total Deliveries:", deliveries.count())

print("Matches Dataset Rows:", matches.count())
print("Matches Dataset Columns:", len(matches.columns))

print("Deliveries Dataset Rows:", deliveries.count())
print("Deliveries Dataset Columns:", len(deliveries.columns))

print("Matches Columns:")
print(matches.columns)

print("\nDeliveries Columns:")
print(deliveries.columns)

matches.show(10, truncate=False)

deliveries.show(10, truncate=False)

print("Matches Schema:")
matches.printSchema()

print("Deliveries Schema:")
deliveries.printSchema()

print(deliveries.columns)
print("Over values:", deliveries.select("over").distinct().orderBy("over").show(25))

## Apache Spark Implementation

print("Spark Version:", spark.version)
print("Spark Application Name:", spark.sparkContext.appName)
print("Spark Master:", spark.sparkContext.master)

matches_rdd = matches.rdd

print("RDD Created Successfully")
print("Total Records:", matches_rdd.count())

completed_matches_rdd = matches_rdd.filter(
    lambda row: row["winner"] is not None
)

print("Completed Matches:", completed_matches_rdd.count())

seasons_rdd = matches_rdd.map(lambda row: row["season"])

unique_seasons = seasons_rdd.distinct()

print("Available Seasons:")
print(unique_seasons.collect())

first_five_matches = matches_rdd.take(5)

for match in first_five_matches:
    print(match)

print("Matches Partitions:", matches.rdd.getNumPartitions())
print("Deliveries Partitions:", deliveries.rdd.getNumPartitions())

## Data Cleaning and Preprocessing
print("Missing Values in Matches:")

for column in matches.columns:
    missing = matches.filter(col(column).isNull()).count()
    if missing > 0:
        print(column, ":", missing)

print("Missing Values in Deliveries:")

for column in deliveries.columns:
    missing = deliveries.filter(col(column).isNull()).count()
    if missing > 0:
        print(column, ":", missing)

matches_clean = matches.dropDuplicates(["id"])

print("Original Matches:", matches.count())
print("Cleaned Matches:", matches_clean.count())

matches_clean = matches_clean.fillna({
    "city": "Unknown",
    "venue": "Unknown",
    "toss_winner": "Unknown",
    "toss_decision": "Unknown",
    "result": "Unknown"
})

print("Missing Values Handled")
matches_clean.show(5)

deliveries_clean = deliveries.filter(
    col("match_id").isNotNull() &
    col("total_runs").isNotNull() &
    col("batting_team").isNotNull() &
    col("bowling_team").isNotNull()
)

print("Original Deliveries:", deliveries.count())
print("Cleaned Deliveries:", deliveries_clean.count())

print("Cleaned Matches:")
matches_clean.printSchema()

print("Cleaned Deliveries:")
deliveries_clean.printSchema()

print("Matches Count:", matches_clean.count())
print("Deliveries Count:", deliveries_clean.count())

print("Duplicate Deliveries Check")

print("Original:", deliveries.count())
print("After Removing Exact Duplicates:",
      deliveries.dropDuplicates().count())

print("Remaining Missing Values in Matches")

for c in matches_clean.columns:
    print(c, matches_clean.filter(col(c).isNull()).count())

print("Remaining Missing Values in Deliveries")

for c in deliveries_clean.columns:
    print(c, deliveries_clean.filter(col(c).isNull()).count())

## Dataset Integration and Feature Preparation
joined_data = deliveries_clean.join(
    matches_clean,
    deliveries_clean.match_id == matches_clean.id,
    "inner"
)

print("Joined Dataset:")
joined_data.show(5, truncate=False)

print("Joined Rows:", joined_data.count())

joined_data = joined_data.select(
    deliveries_clean.match_id,
    "season",
    "city",
    "venue",
    "team1",
    "team2",
    "winner",
    "batting_team",
    "bowling_team",
    "batter",
    "bowler",
    "batsman_runs",
    "total_runs",
    "is_wicket"
)

joined_data.show(5)

match_features = matches_clean.filter(
    col("winner").isNotNull() &
    col("team1").isNotNull() &
    col("team2").isNotNull()
).select(
    "id",
    "season",
    "city",
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner"
)

match_features = match_features.filter(
    (col("winner") == col("team1")) |
    (col("winner") == col("team2"))
)

match_features = match_features.withColumn(
    "team1_win",
    when(col("winner") == col("team1"), 1).otherwise(0)
)

match_features.show(10)

match_features.show(10)

batter_features = deliveries_clean.groupBy(
    "match_id", "batter"
).agg(
    sum("batsman_runs").alias("player_runs"),
    sum(when(
            (col("extras_type").isNull()) |
            (~col("extras_type").contains("wides")),
            1
        ).otherwise(0)
    ).alias("balls_faced")
)

batter_features = batter_features.withColumn(
    "strike_rate",
    when(
        col("balls_faced") > 0,
        col("player_runs") * 100 / col("balls_faced")
    ).otherwise(0)
)

batter_features.show(10)

bowler_features = deliveries_clean.groupBy(
    "match_id", "bowler"
).agg(
    sum(
        col("total_runs") -
        when(
            col("extras_type").isin("byes", "legbyes", "penalty"),
            col("extra_runs")
        ).otherwise(0)
    ).alias("runs_conceded"),

    sum(
        when(
            col("is_wicket") == 1,
            1
        ).otherwise(0)
    ).alias("wickets")
)

bowler_features.show(10)

team_features = deliveries_clean.groupBy(
    "match_id", "batting_team"
).agg(
    sum("total_runs").alias("team_runs"),
    sum("is_wicket").alias("team_wickets"),
    count("*").alias("deliveries_count")
)

team_features.show(10)

match_features.write.mode("overwrite").option(
    "header", True
).csv("outputs/match_features")

batter_features.write.mode("overwrite").option(
    "header", True
).csv("outputs/batter_features")

bowler_features.write.mode("overwrite").option(
    "header", True
).csv("outputs/bowler_features")

team_features.write.mode("overwrite").option(
    "header", True
).csv("outputs/team_features")

print("All datasets saved successfully")

## Exploratory Data Analysis
season_analysis = matches_clean.groupBy("season").count().orderBy("season")

season_analysis.show(30)

team_wins = matches_clean.filter(
    col("winner").isNotNull()
).groupBy("winner").count().orderBy(
    col("count").desc()
)

team_wins.show(10)

toss_analysis = matches_clean.filter(
    col("winner").isNotNull()
).groupBy("toss_decision").agg(
    count("*").alias("total_matches"),
    sum(
        when(col("toss_winner") == col("winner"), 1).otherwise(0)
    ).alias("toss_winner_match_wins")
)

toss_analysis.show()

venue_analysis = matches_clean.groupBy("venue").count().orderBy(
    col("count").desc()
)

venue_analysis.show(10, truncate=False)

team_runs_analysis = team_features.groupBy(
    "batting_team"
).agg(
    avg("team_runs").alias("average_runs"),
    max("team_runs").alias("highest_runs")
).orderBy(
    col("average_runs").desc()
)

team_runs_analysis.show(15)

phase_analysis = deliveries_clean.withColumn(
    "match_phase",
    when(col("over") <= 5, "Powerplay")
    .when(col("over") <= 15, "Middle Overs")
    .otherwise("Death Overs")
)

phase_summary = phase_analysis.groupBy(
    "match_phase"
).agg(
    sum("total_runs").alias("total_runs"),
    avg("total_runs").alias("average_runs"),
    count("*").alias("total_deliveries")
)

phase_summary.show()

player_runs_analysis = batter_features.groupBy(
    "batter"
).agg(
    sum("player_runs").alias("total_runs"),
    avg("strike_rate").alias("average_strike_rate")
).orderBy(
    col("total_runs").desc()
)

player_runs_analysis.show(10, truncate=False)

bowler_analysis = bowler_features.groupBy(
    "bowler"
).agg(
    sum("wickets").alias("total_wickets"),
    avg("runs_conceded").alias("average_runs_conceded")
).orderBy(
    col("total_wickets").desc()
)

bowler_analysis.show(10, truncate=False)

import matplotlib.pyplot as plt

season_pd = season_analysis.toPandas()

season_pd.plot(
    x="season",
    y="count",
    kind="bar",
    figsize=(12, 5),
    legend=False
)

plt.title("Number of Matches by Season")
plt.xlabel("Season")
plt.ylabel("Number of Matches")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

team_pd = team_wins.limit(10).toPandas()

team_pd.plot(
    x="winner",
    y="count",
    kind="bar",
    figsize=(10, 5),
    legend=False
)

plt.title("Top 10 Teams by Match Wins")
plt.xlabel("Team")
plt.ylabel("Number of Wins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

players_pd = player_runs_analysis.limit(10).toPandas()

players_pd.plot(
    x="batter",
    y="total_runs",
    kind="bar",
    figsize=(10, 5),
    legend=False
)

plt.title("Top 10 Run Scoring Players")
plt.xlabel("Player")
plt.ylabel("Total Runs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

runs_pd = team_runs_analysis.limit(10).toPandas()

runs_pd.plot(
    x="batting_team",
    y="average_runs",
    kind="bar",
    figsize=(10, 5),
    legend=False
)

plt.title("Average Runs by Batting Team")
plt.xlabel("Team")
plt.ylabel("Average Runs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
