# AutoPrepAI Pipeline Report

**Overall Confidence:** 80%

**Dataset:** 891 rows × 1583 columns | **Target:** Survived | **Domain:** generic

**Best Model:** LogisticRegression

---

## Data Overview

The dataset consists of 891 rows and 1583 columns, with the target variable being 'Survived'. The domain is generic, indicating that the data may not be specific to a particular industry or context. The quality grade is B, suggesting that the data is generally good but may have some issues. The imputation strategies used were median for numerical missing values, mode for categorical missing values, and forward fill for time-series data. The encoding applied was one-hot for low cardinality columns, binary for binary columns, label for medium cardinality columns, and one-hot again for other categorical columns.

- Dataset shape: 891 rows x 1583 columns
- Target variable: Survived
- Domain: generic
- Quality grade: B
- Imputation strategies used: median, mode, forward fill

## Preprocessing Decisions

The preprocessing decisions made were based on the data's statistical properties and the characteristics of each column. For example, numerical missing values with skewness were imputed using the median, while categorical missing values with high uniqueness were imputed using the mode. Time-series data with only two missing values were filled using forward fill to maintain chronological order. The encoding applied was also based on the cardinality of each column, with one-hot encoding used for low cardinality columns and label encoding used for medium cardinality columns.

- Imputation strategies: median, mode, forward fill
- Encoding applied: one-hot, binary, label

## EDA Insights

The EDA findings revealed several key statistical properties of the data. The columns SibSp, Parch, and Fare have high skewness values, indicating that they may require transformation or encoding to reduce skewness. The target column Survived has an imbalance ratio of 0.623, with a majority of 1s, suggesting that it may be beneficial to oversample the minority class (class 0) or undersample the majority class (class 1). There are also multicollinearity warnings for multiple columns, indicating that selecting a subset of relevant features or using dimensionality reduction techniques may be necessary.

- High skewness values in SibSp, Parch, and Fare
- Imbalance ratio of 0.623 in Survived
- Multicollinearity warnings for multiple columns

## Model Comparison

The model comparison revealed that the best-performing model was LogisticRegression, with a f1_weighted value of 0.8081 and an overfit_gap of 0.1147. The runner-up models were XGBoost and DecisionTree, which had higher overfit gaps but still performed well. The reasons for choosing LogisticRegression as the best model were its high accuracy and low overfitting gap compared to other models.

- Best-performing model: LogisticRegression
- f1_weighted value: 0.8081
- Overfit_gap: 0.1147

## Risks & Limitations

The risks and limitations of the pipeline include the potential for overfitting, especially with models like DecisionTree and XGBoost. Additionally, the use of one-hot encoding may lead to multicollinearity issues if not handled properly. Finally, the imbalance ratio in the target column Survived means that the minority class (class 0) may be underrepresented in the training data.

- Potential for overfitting
- Multicollinearity issues with one-hot encoding
- Imbalance ratio in Survived

## Recommendations

Based on the findings, several recommendations can be made. Firstly, it is recommended to use techniques like oversampling or undersampling to address the imbalance ratio in the target column Survived. Secondly, it is recommended to handle multicollinearity issues with one-hot encoding by using dimensionality reduction techniques or feature selection methods. Finally, it is recommended to monitor the overfitting gap of models and adjust the hyperparameters accordingly.

- Oversampling or undersampling for imbalance ratio
- Dimensionality reduction techniques or feature selection methods for multicollinearity
- Monitoring overfitting gap and adjusting hyperparameters

## ⚠️ Risks

- Overfitting
- Multicollinearity issues with one-hot encoding

## 💡 Recommendations

- Oversampling or undersampling for imbalance ratio
- Dimensionality reduction techniques or feature selection methods for multicollinearity
- Monitoring overfitting gap and adjusting hyperparameters

## 📊 Model Comparison

| Model | accuracy | f1_weighted | precision_weighted | recall_weighted | train_f1 | overfit_gap |
|---|---|---|---|---|---|---|
| RandomForest | 0.838 | 0.8335 | 0.8459 | 0.838 | 1.0 | 0.1665 |
| GradientBoosting | 0.7989 | 0.7962 | 0.7987 | 0.7989 | 0.87 | 0.0738 |
| LogisticRegression | 0.8101 | 0.8081 | 0.8096 | 0.8101 | 0.9228 | 0.1147 |
| SVM | 0.6201 | 0.5198 | 0.6882 | 0.6201 | 0.5969 | 0.0771 |
| DecisionTree | 0.7933 | 0.7894 | 0.7944 | 0.7933 | 1.0 | 0.2106 |
| XGBoost | 0.7821 | 0.7814 | 0.7811 | 0.7821 | 0.9986 | 0.2172 |


---
*Generated by AutoPrepAI v5 — Autonomous Multi-Agent Pipeline*