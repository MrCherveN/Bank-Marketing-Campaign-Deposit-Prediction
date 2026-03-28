<h1 style="text-align:left;">Bank-Marketing-Campaign-Deposit-Prediction</h1>

This project solves a binary classification task based on data from direct marketing campaigns (phone calls) of a Portuguese bank. The goal is to predict whether a client will subscribe to a term deposit (y = yes/no).

<h3 style="text-align:left;">What Was Done:</h3>

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Model Training
* Hyperparameter Tuning
* Feature Importance Analysis
* Error Analysis 

<h3 style="text-align:left;">Models Used:</h3>

* Logistic Regression
* k-Nearest Neighbors (kNN)
* Decision Tree
* AdaBoost
* XGBoost
* LightGBM

<h3 style="text-align:left;">Evaluation Metric:</h3>

ROC-AUC was chosen as the primary metric due to class imbalance. F1-score, Precision, and Recall are reported as additional metrics.

<h3 style="text-align:left;">Results:</h3>

The table below presents the testing metrics for models on the validation dataset that demonstrated sufficiently high performance.
A comprehensive list of all metric values for the evaluated models can be found in the "Mid_term_Project.ipynb" file.

model                      | dataset    |   roc_auc |   f1_score |   precision |   recall |
---------------------------|:-----------|----------:|-----------:|------------:|---------:|
XGBoost                    | validation |  0.810291 |   0.482428 |    0.383249 | 0.650862 |
LightGBM                   | validation |  0.808676 |   0.475365 |    0.374767 | 0.649784 |
XGBoost_Randomized_Search  | validation |  0.810877 |   0.508047 |    0.425966 | 0.62931  |
LightGBM_Randomized_Search | validation |  0.81327  |   0.52134  |    0.454037 | 0.612069 |
XGBoost_Hyperopt           | validation |  0.814232 |   0.51566  |    0.44549  | 0.612069 |
LightGBM_Hyperopt          | validation |  0.81297  |   0.508855 |    0.424658 | 0.634698 |

<h3 style="text-align:left;">Conclusions:</h3>

XGBoost with Bayesian Optimization achieved the best ROC-AUC of 0.814, which was selected as the final model.
However, it should be noted that tuning the hyperparameters improved the ROC-AUC and F1-score for both models, but this came at the expense of a lower Recall. If minimizing the number of missed customers is a priority, the base XGBoost model remains the best alternative due to its highest Recall (0.651) among all models.

SHAP analysis revealed that the model relies primarily on macroeconomic indicators 
(nr.employed, euribor3m) and client interaction history (is_contacted, campaign), 
while demographic features such as marital status, housing, and loan contributed 
minimally to predictions.

<h3 style="text-align:left;">What Could Be Improved:</h3>

* Add a seasonal feature (grouping months into spring/summer/autumn/winter)
* Add interaction features between month and economic indicators (euribor3m, nr.employed)
* Add features describing individual client behaviour independent of economic context
* Collect more data for problematic months (April, May, June, July)
* Try LightGBM Randomized Search or LightGBM Hyperopt as an alternative to XGBoost Hyperopt

<h3 style="text-align:left;">Dataset:</h3>

[The dataset used in this project is available here](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv)
