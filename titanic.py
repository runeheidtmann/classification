import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import seaborn as sns



df = pd.read_csv('titanic.csv', delimiter=',')
df['age'] = df['age'].map(lambda x: None if x == '?' else x)

df['age'].fillna(df.median(), inplace = True)

plt.figure()
sns.pairplot(df,hue='survived')
print(df.columns)

X = df.drop(['survived','name','cabin','home.dest','body','boat','ticket','embarked','fare','age'], axis=1)
y = df['survived']

print(X.columns)

le = preprocessing.LabelEncoder()
le.fit(["female", "male", "C", "S","Q"])
p = le.transform(X['sex'])

X['sex'] = p



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  #for accuracy_score

model = RandomForestClassifier(criterion='gini', n_estimators=700,
                             min_samples_split=10, min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=1,n_jobs=-1)
model.fit(X_train,y_train)
prediction_rm=model.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Random Forest Classifier is',round(accuracy_score(prediction_rm,y_test)*100,2))








# X['survived'] = y
# sns.heatmap(X.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
# fig=plt.gcf()
# fig.set_size_inches(20,12)
# plt.show()


# Support Vector Machines
from sklearn.svm import SVC, LinearSVC

model = SVC()
model.fit(X_train,y_train)
prediction_svm=model.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Support Vector Machines Classifier is',round(accuracy_score(prediction_svm,y_test)*100,2))


from sklearn.neighbors import KNeighborsClassifier


model = KNeighborsClassifier(n_neighbors = 24)
model.fit(X_train,y_train)
prediction_knn=model.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the K Nearst Neighbors Classifier is',round(accuracy_score(prediction_knn,y_test)*100,2))


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X_train,y_train)
prediction_gnb=model.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Gaussian Naive Bayes Classifier is',round(accuracy_score(prediction_gnb,y_test)*100,2))
