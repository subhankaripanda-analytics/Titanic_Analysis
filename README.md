# Titanic Dataset Analysis – EDA and Preprocessing

This project explores the Titanic dataset using Python in PyCharm. The goal is to understand the factors that affected passenger survival during the Titanic disaster. The analysis covers data loading, exploration, visualization, and preprocessing — all steps leading up to model training.

---

## 🔢 Step-by-Step Progress

### ✅ Step 1–5: Load and Understand the Data

- Imported the Titanic dataset using `pandas`.
- Displayed dataset shape and basic information using:
  - `df.shape`
  - `df.head()`
  - `df.columns`
  - `df.info()`
  - `df.describe()`

---

### ✅ Step 6: Correlation Heatmap

- Selected only numeric columns for correlation matrix.
- Visualized feature correlations using `seaborn.heatmap()`.
- Key findings:
  - **Fare** and **Pclass** showed a moderate inverse correlation.
  - Weak overall correlation between most features.

---

### ✅ Step 7–8: Univariate and Bivariate Analysis

- Plotted survival counts grouped by gender using:
  - `sns.countplot(x='Sex', hue='Survived', data=df)`
- Created boxplots for:
  - `Age` vs `Survived`
  - `Fare` vs `Survived`
- Observed that:
  - Women had higher survival rates.
  - 1st class and high-fare passengers had better chances of survival.
  - Younger passengers also showed better survival trends.

---

### ✅ Step 9–10: Observations and Insights

- Based on visual analysis:
  - **Females** and **1st class** passengers had higher survival rates.
  - **Fare** and **Age** were important survival factors.
  - **Cabin**, **Age**, and **Embarked** had missing data.

---

### ✅ Step 11: Final Data Summary

- Printed key findings from the exploratory phase.
- Summarized missing values and features to process next.

---

## 📁 Tools & Libraries Used

- Python 3.x
- PyCharm
- Pandas
- Matplotlib
- Seaborn

---

## 📊 Dataset

The dataset used is the **Titanic - Machine Learning from Disaster** dataset from [Kaggle](https://www.kaggle.com/competitions/titanic).

---

## ✅ Author

This project was developed as part of a step-by-step learning journey using PyCharm and Python for Data Analysis.

