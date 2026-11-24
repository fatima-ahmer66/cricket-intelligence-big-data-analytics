# CRICKET INTELLIGENCE: BIG DATA ANALYTICS AND PREDICTIVE MODELING

**Project Report**

**Course:** Big Data  
**Submitted To:** Dr. Muhamma Farooq 
**Submitted By:** Fatima Ahmer & Mahnoor Nadeem     


# Executive Summary

Cricket produces a large amount of data through matches, teams, players, scores, wickets, and ball-by-ball records. Analyzing this information can help understand team performance, individual player performance, and match outcomes.

This project, titled *Cricket Intelligence: Big Data Analytics and Predictive Modeling*, uses Apache Spark, PySpark, and machine learning techniques to analyze cricket data. Two datasets were used: a matches dataset containing match-level information and a deliveries dataset containing ball-by-ball information.

The project involved loading and understanding the datasets, processing them through Apache Spark, checking missing values, removing duplicate match records, cleaning important fields, and combining both datasets through a common match ID.

Exploratory Data Analysis (EDA) was performed to identify patterns in seasons, teams, venues, batting performances, bowling performances, and match phases. The project also applied three classification algorithms: Logistic Regression, Decision Tree, and Random Forest. Linear Regression was used to predict batter runs.

Among the classification models, Random Forest achieved the highest accuracy of 54.46%. The Linear Regression model achieved an RMSE of 8.52, MAE of 6.37, and R² of 0.8377.

The project provided practical experience in using Big Data technologies and machine learning for cricket analytics. It also highlighted limitations in prediction performance and opportunities for future improvements.

# 1. INTRODUCTION AND PROJECT PLANNING

## 1.1 Introduction

Cricket is a sport that generates a large amount of data during every match. This information includes match results, teams, players, runs, wickets, bowling performance, batting performance, venues, and ball-by-ball records.

This data can provide useful information about team performance and individual players. However, analyzing large datasets manually can be difficult and time-consuming.

Big Data technologies help process and analyze large amounts of information more efficiently. Apache Spark is one such technology that supports data processing and analytical operations.

This project uses Apache Spark and PySpark to process cricket datasets and prepare them for analysis. It also uses machine learning techniques to develop models for predicting match winners and player runs.

The main purpose of this project is to convert raw cricket data into meaningful information and demonstrate how data analytics can support cricket-related decisions.

## 1.2 Problem Statement

Cricket datasets contain a large amount of information about matches, teams, players, and individual deliveries. Understanding this information manually can be difficult, especially when working with thousands of match records and ball-by-ball observations.

There is a need to use efficient data processing and analytical techniques to organize this information, identify important patterns, and develop predictive models.

This project aims to use Apache Spark, PySpark, and machine learning to process cricket data, analyze team and player performance, and predict match winners and batter runs.

## 1.3 Project Objectives

The main objectives of this project are:

1. To process cricket datasets using Apache Spark and PySpark.
2. To understand match-level and ball-by-ball cricket information.
3. To identify missing values and duplicate records.
4. To clean and prepare datasets for analysis.
5. To combine match and delivery information using a common match ID.
6. To prepare match, batter, bowler, and team features.
7. To perform Exploratory Data Analysis on cricket data.
8. To develop classification models for predicting match winners.
9. To develop a regression model for predicting batter runs.
10. To evaluate the performance of the developed machine learning models.
11. To identify the limitations of the analysis and suggest future improvements.

## 1.4 Tools and Technologies

The following tools and technologies were used in the project:

**Python:** Used as the main programming language for data processing, analysis, and machine learning.

**Google Colab:** Used as the cloud-based environment for writing and executing the project notebook.

**Apache Spark:** Used as the data processing engine for handling cricket datasets.

**PySpark:** Used to interact with Apache Spark through Python and perform data processing operations.

**PySpark MLlib:** Used to develop and evaluate classification and regression models.

**Pandas:** Used to convert smaller analytical results into a format suitable for displaying and plotting.

**Matplotlib:** Used to create graphs and visualizations for exploratory data analysis.

**GitHub:** Used to organize and store the project notebook, documentation, and related files.

## 1.5 Project Scope

The project focuses on analyzing historical cricket data through Big Data processing and machine learning.

The scope includes:

- Loading match-level and ball-by-ball datasets.
- Understanding dataset structure and columns.
- Creating a Spark session.
- Demonstrating RDD operations.
- Checking missing values and duplicate records.
- Cleaning and preprocessing the datasets.
- Combining the datasets using match ID.
- Preparing match, batter, bowler, and team features.
- Performing Exploratory Data Analysis.
- Predicting match winners using classification models.
- Predicting batter runs using regression.
- Evaluating model performance.

The project uses historical data to identify patterns and develop predictive models. The analysis is limited to the information available in the selected datasets.

## 1.6 Project Planning and Workflow

The project was completed through a series of stages.

**Stage 1: Dataset Collection and Understanding**

The cricket datasets were collected and loaded into Google Colab. Their structure, columns, data types, and record counts were examined.

**Stage 2: Apache Spark Implementation**

A Spark session was created using PySpark. RDD operations were performed to demonstrate basic Spark processing concepts.

**Stage 3: Data Cleaning and Preprocessing**

Missing values and duplicate records were checked. Duplicate match records were removed, selected missing values were handled, and delivery records without essential information were filtered.

**Stage 4: Dataset Integration and Feature Preparation**

The matches and deliveries datasets were combined using the match ID. Separate analytical features were prepared for matches, batters, bowlers, and teams.

**Stage 5: Exploratory Data Analysis**

The data was analyzed to understand season trends, team wins, venue usage, batting performance, bowling performance, and scoring patterns.

**Stage 6: Predictive Analytics**

Classification models were developed for match winner prediction, and a regression model was developed for batter runs prediction.

**Stage 7: Evaluation and Reporting**

The models were evaluated using suitable performance metrics. The findings, limitations, and recommendations were documented.

# 2. DATASET COLLECTION AND UNDERSTANDING

## 2.1 Dataset Source

The project uses two cricket datasets containing match-level and ball-by-ball information.

**Dataset Source:** IPL Complete Dataset 2008–2020 – Kaggle

https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

The datasets used in the notebook are:

1. matches.csv
2. deliveries.csv

The dataset source and the exact version of the downloaded files should be verified before final submission because the records loaded in the notebook include seasons through 2024, while the supplied Kaggle page refers to 2008–2020.

## 2.2 Matches Dataset

The matches dataset contains information about individual cricket matches.

It includes details such as:

- Match ID
- Season
- City
- Date
- Match type
- Player of the match
- Venue
- Team 1
- Team 2
- Toss winner
- Toss decision
- Match winner
- Match result
- Result margin
- Target runs
- Target overs
- Super over information
- Umpires

This dataset is mainly used for understanding match results, team performance, toss decisions, venues, and seasons.

## 2.3 Deliveries Dataset

The deliveries dataset contains ball-by-ball information from cricket matches.

It includes:

- Match ID
- Inning
- Batting team
- Bowling team
- Over
- Ball
- Batter
- Bowler
- Non-striker
- Batsman runs
- Extra runs
- Total runs
- Extras type
- Wicket information
- Dismissed player
- Dismissal type
- Fielder

This dataset is used to analyze individual batting and bowling performances, team scores, wickets, and scoring patterns during different match phases.

## 2.4 Dataset Size

The notebook reported the following approximate dataset sizes:

| Dataset | Number of Records | Number of Columns |
|---|---:|---:|
| Matches | 1,095 | 20 |
| Deliveries | 260,920 | 17 |

The matches dataset provides an overview of individual matches, while the deliveries dataset provides more detailed information about what happened during each delivery.

## 2.5 Common Identifier

Both datasets contain a match identifier.

- The matches dataset uses the column `id`.
- The deliveries dataset uses the column `match_id`.

These columns were used to connect the two datasets. This allowed match information to be combined with ball-by-ball information for further analysis.

# 3. APACHE SPARK IMPLEMENTATION

## 3.1 Introduction to Apache Spark

Apache Spark is a data processing engine used to process and analyze large datasets. It supports operations such as filtering, grouping, mapping, and aggregating records.

In this project, Apache Spark was used through PySpark in Google Colab. It helped process the cricket datasets and prepare them for analysis.

## 3.2 Spark Session Creation

A Spark session was created to start the Apache Spark environment.

The session provided access to Spark DataFrames and other processing features. The Spark version and application information were also checked to confirm that the environment was running.

The project used local mode with available computer cores. This allowed Spark to perform processing using the resources available in the Google Colab environment.

## 3.3 RDD Operations

RDD stands for Resilient Distributed Dataset. It is a basic data structure in Apache Spark that supports distributed data processing.

The project demonstrated the following RDD operations:

**Creating an RDD:** The matches DataFrame was converted into an RDD.

**Filtering:** Matches with missing winner information were filtered out to identify completed matches.

**Mapping:** The season information was extracted from match records.

**Distinct:** Unique seasons were identified from the available data.

**Take:** The first five match records were retrieved for inspection.

**Counting:** The number of records and partitions was examined.

These operations demonstrated how Spark can process records through transformations and actions.

## 3.4 Transformations and Actions

Spark uses transformations and actions to process data.

**Transformations** create new datasets from existing data. Examples used in the project include filtering and mapping.

**Actions** return results or trigger the execution of transformations. Examples include counting records and retrieving a limited number of records.

Spark also uses lazy evaluation, which means that transformations are not executed immediately. Processing starts when an action is called.

## 3.5 Local Processing

The project used Spark in local mode through the available computer cores.

This demonstrated Spark processing concepts within Google Colab. However, the project did not use a multi-node distributed cluster.

# 4. DATA CLEANING AND PREPROCESSING

## 4.1 Introduction

Data cleaning is an important stage of data analysis because raw datasets may contain missing values, duplicate records, and incomplete information.

Before performing analysis, both cricket datasets were checked and prepared using PySpark.

## 4.2 Missing Value Analysis

Missing values were checked in the matches and deliveries datasets.

Each column was examined to identify whether any records contained null values.

This helped identify incomplete information and determine which fields required attention before further processing.

## 4.3 Removing Duplicate Match Records

Duplicate records in the matches dataset were handled using the match ID.

The duplicate match records were removed based on the `id` column. This helped ensure that each match was represented by a unique match record.

## 4.4 Handling Missing Values

Selected missing values in the matches dataset were replaced with the value "Unknown".

The columns included:

- City
- Venue
- Toss winner
- Toss decision
- Result

This allowed the records to remain available for analysis without leaving those selected fields empty.

## 4.5 Cleaning Delivery Records

The deliveries dataset was filtered to remove records that did not contain essential information.

Records were retained when the following fields were available:

- Match ID
- Total runs
- Batting team
- Bowling team

This helped ensure that the delivery records contained the information required for team and player analysis.

## 4.6 Duplicate Delivery Check

The deliveries dataset was also checked for exact duplicate records.

The original number of records was compared with the number remaining after checking exact duplicates.

This helped identify whether duplicate delivery records were present.

## 4.7 Final Preprocessing

After cleaning, the datasets were examined again to check their schemas, record counts, and remaining missing values.

Missing dismissal information was not removed because not every delivery results in a wicket. Keeping these records was important for preserving the complete delivery information.

The cleaned datasets were then used for integration, feature preparation, and exploratory analysis.

# 5. DATASET INTEGRATION AND FEATURE PREPARATION

## 5.1 Dataset Integration

The matches and deliveries datasets were combined using an inner join.

The `match_id` column from the deliveries dataset was matched with the `id` column from the matches dataset.

This allowed the project to connect information about individual deliveries with details about the match in which they occurred.

The combined dataset included match information such as season, city, venue, teams, and winner, along with delivery information such as batter, bowler, runs, and wickets.

## 5.2 Match Features

Match features were prepared using the matches dataset.

The prepared information included:

- Match ID
- Season
- City
- Venue
- Team 1
- Team 2
- Toss winner
- Toss decision
- Match winner

Matches without a valid winner or the required team information were excluded from the classification feature dataset.

A target variable was also prepared to represent whether Team 1 won the match.

## 5.3 Batter Features

Batter features were prepared by grouping delivery records according to match ID and batter.

The following information was calculated:

- Total runs scored by the batter in a match.
- Balls faced.
- Strike rate.

Strike rate was calculated using runs scored and balls faced.

These features helped compare batter performance across matches and were also used in the regression stage.

## 5.4 Bowler Features

Bowler features were prepared by grouping delivery records according to match ID and bowler.

The following information was calculated:

- Runs conceded.
- Wickets taken.

Runs conceded were adjusted to exclude selected extras, including byes, leg byes, and penalty runs.

These features helped summarize bowling performance in individual matches.

## 5.5 Team Features

Team features were prepared by grouping delivery records according to match ID and batting team.

The following information was calculated:

- Total team runs.
- Total wickets.
- Number of deliveries.

These features provided a summary of team batting performance in each match.

## 5.6 Saving Prepared Datasets

The prepared match, batter, bowler, and team feature datasets were saved in CSV format through Spark.

This allowed the analytical datasets to be stored separately and reused in later stages of the project.

# 6. EXPLORATORY DATA ANALYSIS

## 6.1 Introduction

Exploratory Data Analysis (EDA) was performed to understand the cricket datasets and identify useful patterns.

The analysis focused on match seasons, team wins, toss decisions, venues, batting performance, bowling performance, and scoring patterns.

Graphs were also created using Matplotlib to make the results easier to understand.

## 6.2 Matches by Season

The number of matches played in each season was calculated.

The analysis showed that the 2013 season had the highest number of matches, with 76 matches.

The 2007/08 season had 58 matches. Other seasons also showed differences in the number of recorded matches.

This analysis helped understand how match records were distributed across different seasons.

## 6.3 Top Teams by Match Wins

The match winner column was grouped to identify teams with the highest number of recorded wins.

The results were:

| Team | Match Wins |
|---|---:|
| Mumbai Indians | 144 |
| Chennai Super Kings | 138 |
| Kolkata Knight Riders | 131 |
| Royal Challengers Bangalore | 116 |
| Rajasthan Royals | 112 |

Mumbai Indians had the highest number of recorded match wins in the dataset, followed by Chennai Super Kings and Kolkata Knight Riders.

This analysis helped compare the historical performance of different teams.

## 6.4 Toss Decision Analysis

The project examined toss decisions and whether the toss-winning team also won the match.

The results were:

| Toss Decision | Total Matches | Toss-Winning Team Also Won |
|---|---:|---:|
| Field | 704 | 377 |
| Bat | 391 | 177 |

For matches in which the toss-winning team chose to field, that team also won 377 out of 704 matches, approximately 53.6%.

For matches in which the toss-winning team chose to bat, that team also won 177 out of 391 matches, approximately 45.3%.

The results show a higher match-winning proportion for toss-winning teams that chose to field in this dataset.

However, this relationship does not prove that choosing to field directly causes a team to win. Other factors, such as team strength, venue, weather, and match conditions, can also affect match outcomes.

## 6.5 Venue Analysis

The venue analysis was performed to identify the grounds with the highest number of recorded matches.

The top venues included:

| Venue | Matches |
|---|---:|
| Eden Gardens | 77 |
| Wankhede Stadium | 73 |
| M Chinnaswamy Stadium | 65 |
| Feroz Shah Kotla | 60 |
| Rajiv Gandhi International Stadium, Uppal | 49 |

Eden Gardens had the highest number of recorded matches, followed by Wankhede Stadium and M Chinnaswamy Stadium.

This analysis helped identify frequently used cricket venues.

## 6.6 Average Runs by Batting Team

Team batting performance was analyzed using average runs and highest runs scored.

The results showed that Royal Challengers Bangalore had the highest average runs in the displayed team results, at approximately 195.33.

Gujarat Titans followed with approximately 172.38 runs, while Lucknow Super Giants had approximately 170.68 runs.

This analysis provided a comparison of batting performance among teams in the available records.

## 6.7 Match Phase Analysis

The deliveries dataset was divided into three match phases:

- Powerplay
- Middle Overs
- Death Overs

The analysis produced the following results:

| Match Phase | Total Runs | Average Runs per Delivery |
|---|---:|---:|
| Powerplay | 103,217 | 1.26 |
| Middle Overs | 169,127 | 1.28 |
| Death Overs | 75,412 | 1.62 |

The death overs had the highest average runs per delivery, approximately 1.62.

The powerplay had an average of approximately 1.26 runs per delivery, while the middle overs had approximately 1.28.

This suggests that scoring was faster during the death overs in the analyzed dataset.

## 6.8 Top Run-Scoring Batters

The batter analysis was performed by calculating total runs across the available match records.

The top batters were:

| Player | Total Runs |
|---|---:|
| V Kohli | 8,014 |
| S Dhawan | 6,769 |
| RG Sharma | 6,630 |
| DA Warner | 6,567 |
| SK Raina | 5,536 |

Virat Kohli had the highest total runs in the analysis, followed by Shikhar Dhawan and Rohit Sharma.

This analysis helped identify the leading run-scoring batters in the dataset.

## 6.9 Top Wicket-Taking Bowlers

Bowling performance was analyzed by calculating the total wickets taken by each bowler.

The top bowlers were:

| Player | Total Wickets |
|---|---:|
| YS Chahal | 213 |
| DJ Bravo | 207 |
| PP Chawla | 201 |
| SP Narine | 200 |
| R Ashwin | 198 |

Yuzvendra Chahal had the highest number of wickets in the analysis, followed by Dwayne Bravo and Piyush Chawla.

This analysis helped compare the wicket-taking performance of different bowlers.

## 6.10 EDA Summary

The exploratory analysis provided several useful findings:

1. The 2013 season had the highest number of recorded matches.
2. Mumbai Indians had the highest number of match wins.
3. Eden Gardens was the most frequently recorded venue.
4. Royal Challengers Bangalore had the highest average runs in the displayed team results.
5. Death overs had the highest average runs per delivery.
6. Virat Kohli was the leading batter by total runs.
7. Yuzvendra Chahal was the leading bowler by total wickets.
8. Toss-winning teams that chose to field had a higher win proportion than those choosing to bat.

These findings provided an overview of historical cricket performance and supported the later predictive analytics stage.

# 7. PREDICTIVE ANALYTICS AND MACHINE LEARNING

## 7.1 Introduction

Predictive analytics uses historical data to develop models that estimate possible future outcomes.

In this project, machine learning was applied to two prediction tasks:

1. Match winner classification.
2. Batter runs regression.

PySpark MLlib was used to prepare features, train models, and evaluate their performance.

## 7.2 Match Winner Prediction

The first prediction task focused on identifying whether Team 1 would win a match.

The match dataset was prepared using information about the two teams, venue, toss decision, and historical team performance.

The target variable represented whether Team 1 won the match.

Historical team win rates were prepared using previous match results. The aim was to use past team performance as information for the prediction.

Toss-related features were also included. Therefore, the model represents a prediction made with toss information available, rather than a prediction made before the toss.

## 7.3 Feature Preparation for Classification

The classification pipeline used several feature preparation steps.

**Imputation:** Missing historical team win-rate values were handled using mean imputation.

**String Indexing:** Categorical information such as team names, venues, and toss decisions was converted into numerical indexes.

**One-Hot Encoding:** Categorical indexes were converted into numerical vectors.

**Vector Assembly:** Different features were combined into a single feature vector.

**Standard Scaling:** Numerical features were scaled to prepare them for model training.

These steps helped convert the match information into a format suitable for machine learning algorithms.

## 7.4 Train-Test Split

The classification dataset was divided into training and testing sets.

- 80% of the data was used for training.
- 20% of the data was used for testing.

The split used a random seed of 42 to make the division reproducible.

The training dataset was used to develop the models, while the test dataset was used to evaluate their performance on unseen records.

## 7.5 Classification Models

Three classification algorithms were implemented.

### 7.5.1 Logistic Regression

Logistic Regression is a classification algorithm used to estimate the probability of an outcome belonging to a particular class.

In this project, it was used to predict whether Team 1 would win or lose.

The model achieved:

- ROC-AUC: 0.5299
- Accuracy: 0.5117
- F1-Score: 0.5117

The results show that Logistic Regression had limited predictive performance, with accuracy close to 51.17%.

### 7.5.2 Decision Tree Classifier

Decision Tree is a machine learning algorithm that makes predictions using a series of decision rules.

It divides the data into smaller groups based on feature values and uses those decisions to predict the target class.

The model achieved:

- ROC-AUC: 0.4736
- Accuracy: 0.4930
- F1-Score: 0.3716

The results show that the Decision Tree model performed below the other two initial classification models.

### 7.5.3 Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make a prediction.

It uses the results of several trees to produce a final classification.

The model achieved:

- ROC-AUC: 0.5264
- Accuracy: 0.5446
- F1-Score: 0.5444

Random Forest achieved the highest accuracy and F1-score among the three initial classification models.

However, its ROC-AUC remained close to 0.5, which indicates limited ability to distinguish between the two match outcome classes.

## 7.6 Classification Model Comparison

The results of the three classification models are summarized below.

| Model | ROC-AUC | Accuracy | F1-Score |
|---|---:|---:|---:|
| Logistic Regression | 0.5299 | 0.5117 | 0.5117 |
| Decision Tree | 0.4736 | 0.4930 | 0.3716 |
| Random Forest | 0.5264 | 0.5446 | 0.5444 |

Random Forest performed best in terms of accuracy and F1-score.

Logistic Regression achieved the highest ROC-AUC among the initial models, although its result was also close to 0.5.

Overall, the classification results indicate that predicting match winners using the selected features was challenging.

## 7.7 Confusion Matrix Analysis

Confusion matrices were also produced to compare actual and predicted match outcomes.

The confusion matrix results showed the following:

**Logistic Regression**

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 46 | 52 |
| Actual 1 | 52 | 63 |

**Decision Tree**

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 96 | 2 |
| Actual 1 | 106 | 9 |

**Random Forest**

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 49 | 49 |
| Actual 1 | 48 | 67 |

These results show how often the models correctly or incorrectly classified the match outcomes.

Random Forest produced a more balanced set of predictions than the Decision Tree model in this test set.

## 7.8 Logistic Regression Hyperparameter Tuning

Hyperparameter tuning was performed to improve the Logistic Regression model.

Cross-validation was used to compare different model settings and select a suitable configuration.

The tuned Logistic Regression model achieved:

- ROC-AUC: 0.5299
- Accuracy: 0.5211
- F1-Score: 0.5217

The tuned model achieved a small improvement in accuracy and F1-score compared with the original Logistic Regression model.

However, the improvement was limited, and the ROC-AUC remained unchanged.

## 7.9 Saving and Reloading the Classification Model

The best Logistic Regression pipeline model was saved and then loaded again.

The loaded model was used to generate predictions on the test dataset.

This demonstrated how a trained machine learning pipeline can be saved and reused instead of training it again from the beginning.

# 8. PLAYER RUNS PREDICTION USING REGRESSION

## 8.1 Introduction

The second prediction task focused on estimating the number of runs scored by a batter in a match.

Regression is a machine learning approach used to predict numerical values.

In this project, Linear Regression was used to predict batter runs using historical player performance and match-related information.

## 8.2 Regression Feature Preparation

The batter feature dataset was combined with match dates and venue information.

Historical player performance was calculated using previous matches.

The model used the following information:

- Balls faced in the current match.
- Prior average runs.
- Prior average strike rate.
- Venue.

The historical average runs and strike rate were calculated using previous match records for the same batter.

The model predicted the batter's runs for the current match.

It is important to note that balls faced is information from the current match. Therefore, this model is more suitable for estimating runs during an ongoing innings than predicting a batter's runs completely before a match begins.

## 8.3 Linear Regression Model

Linear Regression was selected to estimate the relationship between the input features and batter runs.

The dataset was divided into training and testing sets using an 80/20 random split with a seed of 42.

The model was trained using the training data and evaluated using the test data.

## 8.4 Regression Evaluation Metrics

The following evaluation metrics were used.

**Root Mean Squared Error (RMSE):** Measures the average size of prediction errors while giving greater importance to larger errors.

**Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.

**R-squared (R²):** Measures how much of the variation in the target variable is explained by the model.

## 8.5 Regression Results

The Linear Regression model achieved the following results:

| Evaluation Metric | Result |
|---|---:|
| RMSE | 8.52 |
| MAE | 6.37 |
| R² | 0.8377 |

The MAE of 6.37 means that the model's predictions differed from actual batter runs by approximately 6.37 runs on average in the test data.

The RMSE of 8.52 indicates the size of prediction errors, with larger errors receiving more weight.

The R² value of 0.8377 indicates that the model explained approximately 83.77% of the variation in batter runs within the test data.

Overall, the regression model produced promising results on the selected test data. However, the use of current-match balls faced means that the model should not be interpreted as a fully pre-match prediction system.

## 8.6 Actual and Predicted Runs

The notebook also displayed examples comparing actual and predicted batter runs.

| Batter | Balls Faced | Prior Average Runs | Actual Runs | Predicted Runs |
|---|---:|---:|---:|---:|
| CL White | 10 | 21.7 | 6 | 12.4 |
| P Kumar | 15 | 21.7 | 18 | 19.9 |
| SB Joshi | 6 | 21.7 | 3 | 6.3 |
| IK Pathan | 5 | 21.7 | 3 | 3.7 |
| ML Hayden | 17 | 21.7 | 25 | 21.9 |
| SM Katich | 17 | 21.7 | 21 | 21.9 |
| S Dhawan | 41 | 21.7 | 52 | 58.2 |
| AM Nayar | 14 | 21.7 | 20 | 17.6 |
| S Chanderpaul | 11 | 21.7 | 16 | 13.1 |
| SM Pollock | 12 | 21.7 | 28 | 14.6 |

The comparison shows that some predictions were close to the actual runs, while others had larger differences.

This demonstrates that regression models can estimate player performance, but their predictions are not always exact.

# 9. OVERALL RESULTS AND DISCUSSION

## 9.1 Summary of Main Findings

The project produced useful findings through data processing, exploratory analysis, and predictive modeling.

The main findings were:

1. The matches dataset contained approximately 1,095 records.
2. The deliveries dataset contained approximately 260,920 records.
3. Mumbai Indians had the highest number of recorded match wins, with 144.
4. Virat Kohli had the highest total runs in the batter analysis, with 8,014.
5. Yuzvendra Chahal had the highest number of wickets in the bowler analysis, with 213.
6. Eden Gardens was the most frequently recorded venue, with 77 matches.
7. Death overs had the highest average runs per delivery, at approximately 1.62.
8. Random Forest achieved the highest classification accuracy, at 54.46%.
9. Linear Regression achieved an R² of 0.8377 for batter runs prediction.

## 9.2 Comparison of Classification and Regression

The classification models had relatively low accuracy, with the best model achieving 54.46%.

The regression model produced an R² of 0.8377 and an MAE of 6.37.

These results suggest that the selected features were more useful for estimating batter runs in the tested setting than for predicting match winners.

However, the classification and regression tasks use different target variables and evaluation metrics. Their scores should not be directly compared as if they measured the same thing.

## 9.3 Overall Interpretation

The project demonstrates how cricket data can be processed and analyzed using Apache Spark and PySpark.

The EDA findings helped identify historical patterns in team and player performance. The machine learning models demonstrated how historical information can be used to develop predictive systems.

The classification results also showed that match winner prediction is difficult when using a limited set of features. Match outcomes can depend on many factors that may not be fully represented in the available data.

The regression results were promising, but the model used information from the current innings. Therefore, its results are more appropriate for estimating runs during a match rather than predicting performance entirely before the match.

# 10. PROJECT LIMITATIONS AND CHALLENGES

## 10.1 Project Limitations

The project had several limitations.

**1. Dataset Limitations**

The analysis depended on the information available in the selected cricket datasets. Some important factors that may influence match outcomes were not included.

**2. Classification Performance**

The classification models achieved relatively low accuracy. Their ROC-AUC results were also close to 0.5, showing limited predictive ability.

**3. Limited Features**

Match winner prediction used selected features such as teams, venue, toss information, and historical team win rates. More relevant information may be required to improve prediction performance.

**4. Regression Prediction Timing**

The batter runs model used balls faced from the current match. Therefore, it is not a completely pre-match prediction model.

**5. Random Train-Test Split**

The project used random 80/20 train-test splits. A time-based split could provide a more realistic evaluation by training on earlier seasons and testing on later seasons.

**6. Clustering Not Implemented**

Although player performance features were prepared, a K-Means clustering model was not implemented in the submitted notebook. Therefore, player grouping and cluster evaluation remain outside the completed analysis.

**7. Local Spark Environment**

The project used Spark in local mode with the available computer cores. It did not use a multi-node cluster.

## 10.2 Challenges Faced

Several challenges were encountered during the project.

**Working with Large Datasets:** Handling ball-by-ball cricket records required the use of Spark DataFrames and PySpark operations.

**Data Quality:** Missing values and duplicate records needed to be checked before performing analysis.

**Dataset Integration:** The matches and deliveries datasets used different column names for the match identifier, so the correct columns had to be matched.

**Feature Preparation:** Match, batter, bowler, and team information needed to be grouped and transformed into suitable analytical features.

**Machine Learning Preparation:** Categorical variables needed to be converted into numerical features before model training.

**Model Evaluation:** Different evaluation metrics were required for classification and regression, and the results needed to be interpreted correctly.

These challenges provided practical experience in data processing, feature preparation, and machine learning.

# 11. CONCLUSION AND RECOMMENDATIONS

## 11.1 Conclusion

The project, *Cricket Intelligence: Big Data Analytics and Predictive Modeling*, focused on analyzing cricket data using Apache Spark, PySpark, and machine learning.

Two datasets were used: matches and deliveries. The datasets were loaded, examined, cleaned, and combined using the match ID.

RDD operations were performed to demonstrate basic Apache Spark concepts. Feature datasets were prepared for matches, batters, bowlers, and teams.

Exploratory Data Analysis was conducted to understand season trends, team wins, venues, batting performance, bowling performance, and match phases.

Three classification models were developed to predict match winners: Logistic Regression, Decision Tree, and Random Forest. Random Forest achieved the highest accuracy of 54.46% among the initial classification models.

Linear Regression was used to predict batter runs. It achieved an RMSE of 8.52, MAE of 6.37, and R² of 0.8377.

Overall, the project provided practical experience in using Big Data technologies and machine learning to analyze cricket information. It demonstrated how raw data can be transformed into useful insights and how predictive models can support sports analytics.

## 11.2 Achievement of Project Objectives

The project achieved the following objectives:

1. Cricket datasets were loaded and examined using PySpark.
2. Apache Spark was initialized and used for data processing.
3. RDD operations such as filtering, mapping, distinct, and counting were demonstrated.
4. Missing values and duplicate records were checked.
5. Selected missing values were handled and incomplete delivery records were filtered.
6. Matches and deliveries were combined using the match ID.
7. Match, batter, bowler, and team features were prepared.
8. Exploratory Data Analysis was performed.
9. Logistic Regression, Decision Tree, and Random Forest were used for match winner classification.
10. Logistic Regression hyperparameter tuning was performed.
11. Linear Regression was used for batter runs prediction.
12. Classification and regression models were evaluated using suitable metrics.

## 11.3 Practical Applications

The findings of this project can be useful in several areas of cricket.

**Team Performance Analysis:** Teams can study previous match results and compare their historical performance.

**Player Evaluation:** Coaches and analysts can examine batting and bowling statistics to understand player strengths and weaknesses.

**Match Preparation:** Historical venue and team information can support match preparation and planning.

**Player Performance Monitoring:** Player statistics can help identify consistent performers and areas that need improvement.

**Data-Based Decision Making:** Cricket analysts can use statistical findings and predictive models to support their decisions.

## 11.4 Recommendations

Based on the project findings, the following recommendations are suggested:

1. Include additional information such as player availability, injuries, weather, and pitch conditions where reliable data is available.
2. Use more historical team performance features to improve match winner prediction.
3. Test additional classification algorithms and compare their results.
4. Use time-based train-test splitting to evaluate predictions on future seasons.
5. Improve the batter runs model by using only information available at the intended prediction time.
6. Implement K-Means clustering in future work to group players according to batting and bowling performance.
7. Explore distributed Spark environments when working with larger datasets.
8. Continue improving data quality and feature preparation to support more reliable analysis.

## 11.5 Future Enhancements

The project can be extended in several ways.

**Improved Match Prediction:** Additional pre-match features and more suitable validation methods could be used to improve match winner prediction.

**Advanced Player Prediction:** Player runs could be estimated using historical performance and information available at the beginning of an innings or match.

**Player Clustering:** K-Means clustering could be implemented to group players according to their performance statistics, such as total runs, strike rate, wickets, and economy.

**Additional Data:** More recent cricket data could be added to improve the coverage of the analysis.

**Distributed Processing:** A distributed Spark cluster could be used to demonstrate processing across multiple machines.

These improvements could make the project more useful and provide opportunities for further research.

# 12. REFERENCES

# Dataset Reference

IPL Complete Dataset 2008–2020. Kaggle.

https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

# 13. APPENDICES

## Appendix A: Dataset Information

The project used two main datasets:

| Dataset | Description |
|---|---|
| matches.csv | Match-level information, including teams, seasons, venues, toss decisions, and winners |
| deliveries.csv | Ball-by-ball information, including batters, bowlers, runs, extras, and wickets |

The datasets were connected through the match identifier.

## Appendix B: Main Analytical Outputs

The following analytical outputs were produced in the notebook:

1. Number of matches by season.
2. Top teams by match wins.
3. Toss decision analysis.
4. Most frequently used venues.
5. Average and highest team runs.
6. Runs scored in different match phases.
7. Top run-scoring batters.
8. Top wicket-taking bowlers.

## Appendix C: Classification Model Results

| Model | ROC-AUC | Accuracy | F1-Score |
|---|---:|---:|---:|
| Logistic Regression | 0.5299 | 0.5117 | 0.5117 |
| Decision Tree | 0.4736 | 0.4930 | 0.3716 |
| Random Forest | 0.5264 | 0.5446 | 0.5444 |
| Tuned Logistic Regression | 0.5299 | 0.5211 | 0.5217 |

## Appendix D: Regression Model Results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 8.52 | 6.37 | 0.8377 |

## Appendix E: Project Deliverables

The project repository contains or is intended to contain:

- Complete combined Jupyter Notebook.
- Project documentation in README.md.
- Python package requirements.
- Dataset information and source details.
- Python source files.
- GitHub Actions workflow.
- Final project report.

The complete implementation and analytical outputs are available in the project Jupyter Notebook.

---

**END OF REPORT**
