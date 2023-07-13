import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pickle

#Load data
data = pd.read_csv("stroke_classification.csv")
target = "stroke"
# In here we will drop the gender column as we should not consider gender to this aspect
x = data.drop("stroke", axis = 1)
x = data.drop("gender", axis = 1)
y = data[target]
# Some columns consists of NaN value so we will fill it with 0
x.fillna(0, inplace=True)
# Split data
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.7, random_state = 3)
# Choose the scaler for data
scaler = MinMaxScaler()
# Identify categorical columns
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# Choose the options for the process
parameters= {
      "n_estimators": [50,100,200,300],
       "criterion": ["gini","entropy","log_loss"],
       "max_depth": [None, 5, 10, 30],
       "max_features": ["sqrt", "log2"]
}

cls = GridSearchCV(RandomForestClassifier(), param_grid = parameters, scoring="accuracy", cv=6, verbose= 1, n_jobs=8)
cls.fit(x_train,y_train)
print(cls.best_score_)
print(cls.best_params_)
y_predict = cls.predict(x_test)
print(classification_report(y_test,y_predict))
# Save model
with open('stroke_bestmodel_pkl','wb') as files:
    pickle.dump(cls, files)
# Print the confusion matrix
cm = np.array(confusion_matrix(y_test,y_predict, labels = [0,1]))
confusion = pd.DataFrame(cm, index=["Not Stroke", "Stroke"], columns = ["Not Stroke", "Stroke"])
sn.heatmap(confusion, annot= True, fmt="g")
plt.savefig("stroke_prediction.png")
