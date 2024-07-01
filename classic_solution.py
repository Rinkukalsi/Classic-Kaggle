# %% [markdown]
# Classic Kaggle : Using Titanic Dataset to Predict Survival Prediction (Supervised Learning)

# %%
# Installing and importing necessary libraries

%pip install -U scikit-learn
%pip install --upgrade pip setuptools
import pandas as pd 
import numpy as np 
import matplotlib 
%pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# %%
# Importing train and test files 
train_ = pd.read_csv('train.csv')

# %%
test_ = pd.read_csv('test.csv')

# %% [markdown]
# Preprocessing

# %%
# Making a copy for securing
df_test = test_.copy()

# %%
df_train = train_.copy()

# %%
# Fill null values 
df_train.fillna(0, inplace = True)

# %%
df_test.fillna(0, inplace = True)

# %%
#Encoding columns for better interpretation
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder() 

test_['Sex'] = encoder.fit_transform(test_['Sex'])
test_['Embarked'] = encoder.fit_transform(test_['Embarked'])

train_['Sex'] = encoder.fit_transform(train_['Sex'])
train_['Embarked'] = encoder.fit_transform(train_['Embarked'])

# %%
check = test_.drop(['Name','Ticket','Cabin'], axis = 1)

# %%
prep = train_.drop(['Name','Ticket','Cabin'], axis = 1)

# %%
prep.fillna(0, inplace = True)
check.fillna(0, inplace = True)

# %%
prep

# %%
prep['Fare'] = prep['Fare'].round().astype(int)

# %%
check['Fare'] = check['Fare'].round().astype(int)

# %%
prep['Fare']

# %%
#Checking dimension of Data
if prep.ndim == 2:
    print("DataFrame is 2-dimensional")
else:
    print("DataFrame is not 2-dimensional")

# %%
#Correlation Matrix
mat = prep.corr()

# %% [markdown]
# Visualization for better insights

# %%
sns.heatmap(mat,annot=True )
plt.title('mat')
plt.show()

# %%
sns.lineplot(prep, y = 'Survived', x = 'SibSp')

# %%
sns.lineplot(prep, y = 'Survived', x = 'Parch')

# %%
sns.lineplot(prep, y = 'Survived', x = 'Age')

# %%
sns.barplot(prep, x= 'Sex', y= 'Survived') 

# %% [markdown]
# Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
X = prep[['Fare','Parch','Age','Sex','Pclass']]
y = prep[['Survived']]
x__test = check[['Fare','Parch','Age','Sex','Pclass']]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.20, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predic = model.predict(X_val)

print(predic)

# %%
accuracy = accuracy_score(y_val, predic)
print(accuracy)

# %%
logr_model['Survived'].value_counts()

# %% [markdown]
# Random forest

# %%
from sklearn.ensemble import RandomForestClassifier
#X_train_, X_val_, y_train_, y_val_ = train_test_split(X,y, test_size=0.20, random_state=42)

clf = RandomForestClassifier(n_estimators= 100,random_state= 25)
clf.fit(X, y)

pred_ran = clf.predict(x__test)

print(pred_ran) 

# %%
accuracy = accuracy_score(y_val_, pred_ran)
print(accuracy)

# %%
ran_model = pd.DataFrame({'PassengerId': check.PassengerId, 'Survived': pred_ran})
print(ran_model)

# %%
ran_model['Survived'].value_counts()

# %%
ran_model.to_csv('Submission_Titanic.csv', index=False)

# %%
sub = pd.read_csv('Submissionn.csv')

# %%
sub.head()


