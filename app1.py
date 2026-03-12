# heart_model_training.py

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ===================== Load Dataset =====================
df = pd.read_csv("heart.csv")

print(df.sample(5))
print(df.info())
print(df.describe())

# ===================== Data Cleaning =====================
print("Null values:\n", df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("After removing duplicates:", df.duplicated().sum())

print("Unique values:\n", df.nunique())

# Replace 0 with NaN in chol
df['chol'].replace(0, np.nan, inplace=True)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)

after_impute = imputer.fit_transform(df)
df = pd.DataFrame(after_impute, columns=df.columns)

# Replace 0 with NaN in trestbps
df['trestbps'].replace(0, np.nan, inplace=True)

imputer = KNNImputer(n_neighbors=3)
after_impute = imputer.fit_transform(df)
df = pd.DataFrame(after_impute, columns=df.columns)

print(df['trestbps'].unique())

# Convert columns to int except oldpeak
wop = df.columns
wop = wop.drop('oldpeak')
df[wop] = df[wop].astype('int32')

print(df.info())

# ===================== Visualization =====================
import plotly.express as px
import plotly.io as pio
# pio.renderers.default = "browser"

corr = df.corr()['target'].drop('target').sort_values()

fig = px.bar(
    x=corr.index,
    y=corr.values,
    title="Correlation of Features with Heart Disease"
)
# fig.show()

# px.histogram(df, x='age', color='target').show()
# px.pie(df, names='target', title='Percentage of Heart Disease Classes').show()
# px.histogram(df, x='sex', color='target').show()
# px.histogram(df, x='cp', color='target').show()

# ===================== Train Test Split =====================
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== Logistic Regression =====================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

solver = ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']

best_solver = ''
test_score = np.zeros(6)

for i, n in enumerate(solver):
    lr = LogisticRegression(solver=n).fit(X_train, y_train)
    test_score[i] = lr.score(X_test, y_test)

    if lr.score(X_test, y_test) == test_score.max():
        best_solver = n

print("Best Solver:", best_solver)

lr = LogisticRegression(solver=best_solver)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

import pickle
with open('LogisticR.pkl', 'wb') as file:
    pickle.dump(lr, file)

# ===================== SVM =====================
from sklearn.svm import SVC
from sklearn.metrics import f1_score

kernels = {'linear':0, 'poly':0, 'rbf':0, 'sigmoid':0}
best_kernel = ''

for i in kernels:
    svm = SVC(kernel=i)
    svm.fit(X_train, y_train)

    pred = svm.predict(X_test)

    kernels[i] = f1_score(y_test, pred, average="weighted")

    if kernels[i] == max(kernels.values()):
        best_kernel = i

svm = SVC(kernel=best_kernel)
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("SVM F1 Score:", f1_score(y_test, svm_pred, average="weighted"))

with open('SVM.pkl', 'wb') as file:
    pickle.dump(svm, file)

# ===================== Decision Tree =====================
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dtree = DecisionTreeClassifier(class_weight='balanced')

param_grid = {
    'max_depth': [3,4,5,6,7,8],
    'min_samples_split': [2,3,4],
    'min_samples_leaf': [1,2,3,4],
    'random_state': [0,42]
}

grid = GridSearchCV(dtree, param_grid, cv=5)
grid.fit(X_train, y_train)

best_dt = DecisionTreeClassifier(**grid.best_params_, class_weight='balanced')
best_dt.fit(X_train, y_train)

pred = best_dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, pred))

with open('dt.pkl', 'wb') as file:
    pickle.dump(best_dt, file)

# ===================== Random Forest =====================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

param_grid = {
    'n_estimators': [50,100],
    'max_depth': [5,10],
    'max_features': ['sqrt','log2']
}

grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = RandomForestClassifier(**grid.best_params_)
best_rf.fit(X_train, y_train)

pred = best_rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, pred))

with open('random.pkl', 'wb') as file:
    pickle.dump(best_rf, file)


print("All models trained and saved successfully!")
