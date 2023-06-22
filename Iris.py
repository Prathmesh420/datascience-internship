import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
import seaborn as sns

# Load iris dataset
irs = load_iris()

print(irs)
print(irs.keys())
print(irs.data)
print(irs.target)
print(irs.feature_names)
print(irs.target_names)
print(irs.DESCR)

# Read CSV file
df = pd.read_csv("C:/CSV Files/Iris.csv")
print(df)
print(df.head(10))
print(df.tail())
print(df.columns.values)
print(df.describe())

# Preparing X and Y
X = df.drop(['Id', 'Species'], axis=1)
Y = df['Species']
print(X)
print(Y)

# Feature selection
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
featuresScores.columns = ['Specs', 'Score']
print(featuresScores)

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()

# Numerical to Categorical
df['SepalLengthCm'] = pd.cut(df['SepalLengthCm'], 3, labels=['0', '1', '2'])
df['SepalWidthCm'] = pd.cut(df['SepalWidthCm'], 3, labels=['0', '1', '2'])
df['PetalLengthCm'] = pd.cut(df['PetalLengthCm'], 3, labels=['0', '1', '2'])
df['PetalWidthCm'] = pd.cut(df['PetalWidthCm'], 3, labels=['0', '1', '2'])
print(df)

print("Number of null values")
print(df.isnull().sum())

print("Number of not null")
print(df.notnull().sum())

df['SepalLengthCm'].fillna(df['SepalLengthCm'].mean(), inplace=True)
print(df.isnull().sum())

df['SepalLengthCm'].fillna(df['SepalLengthCm'].max(), inplace=True)
print(df.isnull().sum())

a = (df['Species'] == 'Iris-setosa').sum()
print(a)

b = (df['Species'] == 'Iris-virginica').sum()
print(b)

Y = (df['Species'] == 'Iris-versicolor').sum()
print(Y)

print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))

# Identifying the Outliers by plotting
sns.boxplot(df['SepalLengthCm'])
plt.show()

# Dealing with the Outliers
print(df['SepalLengthCm'])
Q1 = df['SepalLengthCm'].quantile(0.25)
Q3 = df['SepalLengthCm'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR

print(upper)
print(lower)

df['SepalLengthCm'] = df['SepalLengthCm'].clip(lower, upper)
print(df['SepalLengthCm'])

# PCA and Logistic Regression
logr = LogisticRegression()
pca = PCA(n_components=2)

X = df.drop(['Id', 'Species'], axis=1)
Y = df['Species']

pca.fit(X)
X = pca.transform(X)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.3)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(accuracy_score(y_test, y_pred))
