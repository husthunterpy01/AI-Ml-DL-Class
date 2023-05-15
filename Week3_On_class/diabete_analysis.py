import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
data =pd.read_csv("diabetes.csv")
#print(data)
#print(data.head()) #By default 5 column
#print(data.info())
#result = data.corr()
#print(data["Outcome"].value_counts())

#Plot the figure by matplot and seaborn
#plt.figure(figsize=(8,8))
#sn.histplot(data["Outcome"])
#plt.title("Diabetes distribution")
#plt.savefig("diabetes.jpg")


#Phan chia theo chieu doc: Feature doc target
#Phan chia theo chieu ngang: train_test
# Split data, by default 75 -25 for train - test, uu tien train trc test (train > test)
# Random state: used to maintain stable result, dung de giu model co 1 luong data sau moi lan random no phai nhu nhau 20% htrc phai nhu 20% hsau
# Neu xet thi thg random_state = 42

# Den tiem may yeu cau may quan ao
#fit: do de may quan ao
#transform: may quan ao (sau khi da fit) vs bo test ()
#fit_transform: modify after transform vs bo train

#Set features and target
target = "Outcome"
x = data.drop("Outcome",axis = 1) #Index là chỉ số, columns là data
y = data[target]

#Split data
x_train, x_test,y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=42)
#print(len(x_train), len(y_train))
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters= {
      "n_estimators": [50,100,200],
       "criterion": ["gini","entropy","log_loss"],
       "max_depth": [None, 5, 10],
       "max_features": ["sqrt", "log2"]
}
#n_jobs huy ddonogj ram de chay kqua
cls = GridSearchCV(RandomForestClassifier(), param_grid = parameters, scoring="accuracy", cv=6, verbose=1,n_jobs=8) #scoring tieu chi danh gia, verbose messa, cv la so lan thu
cls.fit(x_train,y_train) # fit bo train
print(cls.best_score_)
print(cls.best_params_)
y_predict = cls.predict(x_test)
#for i, j in zip(y_test,y_predict):
#   print("Actual:{} Predict:{}".format(i, j))
print(classification_report(y_test,y_predict))
#cm = np.array(confusion_matrix(y_test,y_predict, labels = [0,1]))
#confusion = pd.DataFrame(cm, index=["Not Diabetic", "Diabetic"], columns = ["Not Diabetic", "Diabetic"])
#sn.heatmap(confusion, annot= True, fmt="g")
#plt.savefig("diabetes_prediction.jpg")

# We can use Grid search to find the optimized parameters