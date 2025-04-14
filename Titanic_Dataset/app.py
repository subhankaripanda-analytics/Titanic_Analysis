import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: for nicer visuals
sns.set(style="whitegrid")
df = pd.read_csv("train.csv")
print(df.head())
print(df.info())             # Data types and missing values
print(df.describe())         # Summary statistics
print(df.isnull().sum())     # Total missing values in each column
print(df['Survived'].value_counts())  # Target variable distribution
import matplotlib.pyplot as plt

# Age distribution
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Countplot for Survived
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Countplot for Sex
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()
# Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()
# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Plot heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplot - Age vs Survival
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

# Fare Distribution by Survival
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Fare vs Survival")
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("Observation: Females had a much higher survival rate than males.")
print("Observation: Higher fare seems correlated with survival.")
print("Summary:")
print("- Survival rate was higher among females and passengers in 1st class.")
print("- Younger passengers and those who paid higher fares had better chances of survival.")
print("- Cabin, Age, and Embarked have missing values.")
from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)  # Features
y = df['Survived']               # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
