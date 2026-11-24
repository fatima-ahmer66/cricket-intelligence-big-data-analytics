# cricket-intelligence-big-data-analytics
# **Cricket Intelligence: Big Data Analytics and Predictive Modeling**

**Dataset Source (Kaggle):**\
IPL Complete Dataset 2008–2020: [https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020?utm_source=chatgpt.com)

## **1. Introduction and Project Planning**

### **1.1 Introduction**

Cricket generates a large amount of data through matches, teams, players, scores, wickets, and ball-by-ball records. This data contains useful information that can help understand team performance, player statistics, and match results.

This project uses Big Data technologies and machine learning techniques to analyze cricket data. Apache Spark and PySpark are used to process, clean, and analyze the datasets. The project also applies machine learning models to predict match winners and player runs.

The main purpose is to convert raw cricket data into useful information and understand how data analytics can support cricket-related decisions.

### **1.2 Problem Statement**

Cricket datasets contain a large amount of information about matches and individual player performances. Analyzing this information manually can be time-consuming and difficult.

This project aims to use Apache Spark and machine learning to process cricket data, identify important patterns, analyze team and player performance, and develop predictive models.

### **1.3 Project Objectives**

The main objectives of this project are:

1. To process cricket datasets using Apache Spark and PySpark.
2. To understand and explore match-level and ball-by-ball data.
3. To clean and prepare the datasets for analysis.
4. To combine match and delivery information using a common match ID.
5. To analyze team and player performance through Exploratory Data Analysis.
6. To develop classification models for predicting cricket match winners.
7. To develop a regression model for predicting player runs.
8. To evaluate the performance of the machine learning models.

### **1.4 Tools and Technologies**

The following tools and technologies are used in this project:

- **Python:** Programming language used for data processing and analysis.
- **Google Colab:** Cloud-based environment used to write and execute Python code.
- **Apache Spark:** Big Data processing engine used to handle cricket datasets.
- **PySpark:** Python API used to work with Apache Spark.
- **PySpark MLlib:** Used to build and evaluate machine learning models.
- **Pandas:** Used for handling and displaying smaller analytical results.
- **Matplotlib:** Used to create graphs and visualizations.
- **GitHub:** Used to store and manage the project files.

### **1.5 Project Scope**

This project focuses on cricket match analysis and player performance using Big Data processing and machine learning.

The project includes:

- Loading and understanding cricket datasets.
- Apache Spark implementation and basic RDD operations.
- Data cleaning and preprocessing.
- Integration of match and ball-by-ball datasets.
- Feature preparation for team and player analysis.
- Exploratory Data Analysis and visualizations.
- Match winner prediction using classification models.
- Player runs prediction using regression.
- Evaluation and interpretation of model results.

The project uses historical cricket data to identify patterns and develop predictive models. It also discusses the limitations of the models and possible improvements for future work.

## **2. Results and Discussion**

### **2.1 Key Findings from Cricket Data**

The analysis of cricket data provided useful information about matches, teams, and player performances. The project used two datasets: matches and deliveries. The matches dataset contained approximately 1,095 match records, while the deliveries dataset contained approximately 260,920 ball-by-ball records.

The analysis showed that the number of matches varied across seasons. The 2013 season had the highest number of matches in the dataset, with 76 matches. The 2007/08 season had 58 matches.

The analysis also showed that some teams and players performed consistently well across the available data. Mumbai Indians had the highest number of recorded match wins, with 144 wins, followed by Chennai Super Kings with 138 wins and Kolkata Knight Riders with 131 wins.

The player analysis identified leading batters and bowlers. Virat Kohli had the highest total runs in the batter analysis, with 8,014 runs. Yuzvendra Chahal had the highest number of wickets in the bowler analysis, with 213 wickets.

These findings show how cricket data can be used to understand team and individual performances.

### **2.2 Team Performance Insights**

The team performance analysis was conducted using match results and ball-by-ball data.

The results showed that Mumbai Indians had the highest number of match wins, with 144 wins. Chennai Super Kings had 138 wins, while Kolkata Knight Riders had 131 wins.

The batting analysis showed that Royal Challengers Bangalore had the highest average runs per innings in the displayed team results, at approximately 195.33 runs. Gujarat Titans followed with approximately 172.38 runs.

The analysis of match phases showed that the death overs had the highest average runs per delivery, approximately 1.62. The powerplay had an average of approximately 1.26 runs per delivery, while the middle overs had approximately 1.28 runs per delivery.

These results suggest that teams generally scored runs at a higher rate during the death overs in the analyzed data.

### **2.3 Player Performance Insights**

The project analyzed batter and bowler performance using ball-by-ball records.

Virat Kohli was the leading batter in the analysis, with 8,014 total runs. Other top performers included Shikhar Dhawan with 6,769 runs, Rohit Sharma with 6,630 runs, and David Warner with 6,567 runs.

In bowling, Yuzvendra Chahal had the highest number of wickets, with 213 wickets. Dwayne Bravo followed with 207 wickets, while Piyush Chawla had 201 wickets.

The analysis shows that both batting and bowling statistics can help identify important players and compare their performances.

### **2.4 Factors Affecting Match Outcomes**

The project also analyzed the relationship between toss decisions and match results.

The results showed that teams choosing to field after winning the toss won 377 out of 704 matches. Teams choosing to bat first won 177 out of 391 matches.

This represents approximately 53.6% of matches for the fielding decision and 45.3% for the batting decision.

The results suggest that teams choosing to field had a higher winning percentage in this dataset. However, this does not prove that choosing to field directly causes a team to win. Other factors, such as team strength, venue, weather, and match conditions, can also affect the result.

### **2.5 Interpretation of Prediction Results**

The project used three classification models to predict match winners: Logistic Regression, Decision Tree, and Random Forest.

**Classification Model Results**

| Model               | ROC-AUC | Accuracy | F1-Score |
| ------------------- | ------- | -------- | -------- |
| Logistic Regression | 0.5299  | 0.5117   | 0.5117   |
| Decision Tree       | 0.4736  | 0.4930   | 0.3716   |
| Random Forest       | 0.5264  | 0.5446   | 0.5444   |

Random Forest achieved the highest accuracy of approximately 54.46% and the highest F1-score of approximately 0.5444 among the three initial models.

Logistic Regression achieved the highest ROC-AUC of approximately 0.5299. The results show that the classification models had limited ability to predict match winners, as their performance was relatively close to random guessing.

The project also applied hyperparameter tuning to Logistic Regression. The tuned model achieved a ROC-AUC of 0.5299, accuracy of 0.5211, and F1-score of 0.5217. This showed only a small improvement in accuracy and F1-score compared with the original Logistic Regression model.

**Player Runs Prediction Results**

For player runs prediction, Linear Regression was used. The model achieved the following results:

- **RMSE:** 8.52
- **MAE:** 6.37
- **R²:** 0.8377

The R² value indicates that the model explained a large proportion of the variation in player runs within the test data. The MAE indicates an average prediction error of approximately 6.37 runs.

Overall, the regression model performed better on its evaluation metrics than the classification models. However, its predictions use the number of balls faced in the same match, so they should be understood as estimates based partly on information from that match rather than predictions made completely before the match starts.

## **3. Conclusion and Recommendations**

### **3.1 Project Summary**

The project, Cricket Intelligence: Big Data Analytics and Predictive Modeling, focused on analyzing cricket data using Apache Spark, PySpark, and machine learning.

Two datasets were used: matches and deliveries. The datasets were cleaned, processed, and combined using the match ID.

Exploratory Data Analysis was performed to study match trends, team performance, batting performance, bowling performance, and toss decisions.

Machine learning was also applied to predict match winners through classification models and player runs through a regression model.

The project provided practical experience in handling cricket data and applying Big Data technologies to real-world analytical problems.

### **3.2 Achievement of Project Objectives**

The project achieved several of its main objectives:

1. Cricket datasets were loaded and processed using PySpark.
2. Missing values and duplicate match records were checked and handled.
3. Match and delivery datasets were combined using the match ID.
4. Batter, bowler, team, and match features were prepared.
5. Exploratory Data Analysis was performed to identify important patterns.
6. Logistic Regression, Decision Tree, and Random Forest were used for match winner classification.
7. Linear Regression was used for player runs prediction.
8. Model performance was evaluated using appropriate classification and regression metrics.

The project demonstrated how Big Data processing and machine learning can be used together for cricket analytics.

### **3.3 Practical Applications**

The findings of this project can be useful in different areas of cricket:

- **Team analysis:** Teams can study previous match results and performance patterns.
- **Player evaluation:** Coaches and analysts can compare batting and bowling performances.
- **Match preparation:** Historical team and venue information can support match planning.
- **Performance monitoring:** Player statistics can help identify strengths and areas for improvement.
- **Data-based decisions:** Cricket analysts can use statistical information to support decisions.

These applications show the value of data analytics in modern cricket.

### **3.4 Project Limitations**

The project had some limitations:

1. The prediction results depended on the quality and available information in the datasets.
2. The classification models achieved relatively low accuracy, showing that match winner prediction is difficult.
3. Only a limited number of machine learning models were implemented.
4. The regression model used balls faced from the same match as an input feature. Therefore, it is not a fully pre-match prediction model.
5. A random train-test split was used, so the evaluation does not represent a strict test on only future seasons.
6. Player clustering was not implemented in the current notebook.
7. The project was executed in Spark local mode using the available computer cores, rather than on a multi-node cluster.

These limitations provide opportunities for further improvement.

### **3.5 Future Enhancements**

The project can be improved in the future by:

1. Adding more recent cricket match and player data.
2. Testing additional machine learning models to improve prediction performance.
3. Using more pre-match features, such as historical team performance, player availability, and venue conditions.
4. Using a time-based train-test split to test predictions on future matches.
5. Implementing K-Means clustering to group players according to their performance.
6. Improving player runs prediction by using only information available before the prediction point.
7. Running Spark on a distributed cluster to process larger datasets.

These improvements can make the project more accurate and useful for cricket analysis.

## **4. References and Appendices**

### **4.1 Dataset References**

The project used the following datasets:

1. **Matches Dataset (matches.csv):** Contains match-level information, including season, teams, venue, toss decision, and match winner.
2. **Deliveries Dataset (deliveries.csv):** Contains ball-by-ball information, including batter, bowler, runs, extras, and wickets.

**Dataset Source:**\
IPL Complete Dataset 2008–2020 – Kaggle\
[https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020?utm_source=chatgpt.com)

### **4.2 Research References**

The following technical resources were relevant to the project:

1. Apache Spark Documentation: [https://spark.apache.org/docs/latest/](https://spark.apache.org/docs/latest/?utm_source=chatgpt.com)
2. PySpark Documentation: [https://spark.apache.org/docs/latest/api/python/](https://spark.apache.org/docs/latest/api/python/?utm_source=chatgpt.com)
3. Apache Spark MLlib Documentation: [https://spark.apache.org/docs/latest/ml-guide.html](https://spark.apache.org/docs/latest/ml-guide.html?utm_source=chatgpt.com)
4. Matplotlib Documentation: [https://matplotlib.org/stable/](https://matplotlib.org/stable/?utm_source=chatgpt.com)

### **4.3 Important Code Sections**

The following code sections were important in completing the project:

- **Spark Session Creation:** Used to initialize Apache Spark in Google Colab.
- **RDD Operations:** Used to demonstrate filtering, mapping, distinct values, and record counting.
- **Data Cleaning:** Used to remove duplicate match records, handle selected missing values, and filter incomplete delivery records.
- **Dataset Integration:** Used to combine matches and deliveries through the match ID.
- **Feature Preparation:** Used to prepare match, batter, bowler, and team performance information.
- **Exploratory Data Analysis:** Used to analyze season trends, team wins, venues, match phases, batters, and bowlers.
- **Classification Pipeline:** Used to train Logistic Regression, Decision Tree, and Random Forest models.
- **Hyperparameter Tuning:** Used to tune Logistic Regression with cross-validation.
- **Regression Pipeline:** Used to predict batter runs through Linear Regression.

The complete Python implementation is available in the project Jupyter Notebook.

### **4.4 Additional Tables and Results**

The following results are included as supporting material in the appendix.

**Table 1: Top Teams by Match Wins**

| Team                        | Wins |
| --------------------------- | ---- |
| Mumbai Indians              | 144  |
| Chennai Super Kings         | 138  |
| Kolkata Knight Riders       | 131  |
| Royal Challengers Bangalore | 116  |
| Rajasthan Royals            | 112  |

**Table 2: Top Batters by Total Runs**

| Player    | Total Runs |
| --------- | ---------- |
| V Kohli   | 8,014      |
| S Dhawan  | 6,769      |
| RG Sharma | 6,630      |
| DA Warner | 6,567      |
| SK Raina  | 5,536      |

**Table 3: Top Bowlers by Wickets**

| Player    | Wickets |
| --------- | ------- |
| YS Chahal | 213     |
| DJ Bravo  | 207     |
| PP Chawla | 201     |
| SP Narine | 200     |
| R Ashwin  | 198     |

**Table 4: Final Model Evaluation**

| Model                     | Main Evaluation Results           |
| ------------------------- | --------------------------------- |
| Logistic Regression       | Accuracy: 51.17%, ROC-AUC: 0.5299 |
| Decision Tree             | Accuracy: 49.30%, ROC-AUC: 0.4736 |
| Random Forest             | Accuracy: 54.46%, ROC-AUC: 0.5264 |
| Tuned Logistic Regression | Accuracy: 52.11%, ROC-AUC: 0.5299 |
| Linear Regression         | RMSE: 8.52, MAE: 6.37, R²: 0.8377 |

These tables summarize the main findings and evaluation results of the project.

## **Project Repository**

The complete project notebook, source code, dataset information, and project report are available in this GitHub repository.
