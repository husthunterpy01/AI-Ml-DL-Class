import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Import data
data = pd.read_csv("csgo.csv")
target = "result"
data1 = data.drop(columns=["month","day","year","date","team_a_rounds","team_b_rounds","map"])
# Drop the unused columns data
x = data1.drop("result", axis=1)
y = data1["result"]

# Apply OneHotEncoder to the entire target variable
ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y.values.reshape(-1,1))
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.7, random_state=3)

# Define x as StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Defining GridSearch for the Logistic regression model
parameters = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'liblinear', 'sag', 'saga']
}

y_train = np.argmax(y_train.toarray(), axis=1) # convert y_train to a numpy array before applying argmax
lr = LogisticRegression()
cls = GridSearchCV(LogisticRegression(), param_grid = parameters, scoring="accuracy", cv=6, verbose= 1, n_jobs=8)
cls.fit(x_train, y_train)
#Print the results
print("Best parameters:", cls.best_params_)
print("Best score", cls.best_score_)
print("----------------------------------------")
y_predict = cls.predict(np.asarray(x_test))
y_test1 = np.argmax(np.asarray(y_test.toarray()), axis=1) # convert y_test to a numpy array before applying argmax
print(classification_report(y_test1, y_predict))

# Save model
with open('stroke_bestmodel_pkl','wb') as files:
    pickle.dump(cls, files)

# Sketch the confusion matrix
conf_matrix = confusion_matrix(y_test1, y_predict)
print(conf_matrix)
# Sketch the confusion matrix using seaborn and matplotlib
plt.figure(figsize=(8,6), dpi=100)
#Scale up the size
sns.set(font_scale = 1.1)
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )

# set x-axis label and ticks.
ax.set_xlabel("Predicted CSGO", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['Lost', 'Tie','Win'])

# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['Lost', 'Tie','Win'])

# set plot title
ax.set_title("Confusion Matrix for the CSGO BATTLE", fontsize=14, pad=20)
plt.savefig("Confusion Matrix for the CSGO BATTLE.png")
#plt.show() - can use this to show the confusion matrix without saving as .png or .jpg type
