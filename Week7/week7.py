import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
path = "mnist.npz"
with np.load(path) as f:
    x_train = f["x_train"]
    x_test = f["x_test"]
    y_train = f["y_train"]
    y_test = f["y_test"]
#print(y_train[48000])
#print(x_train[40000].dtype)\

num_training_sample = x_train.shape[0]
num_test_sample = x_test.shape[0]
x_train = np.reshape(x_train, (num_training_sample,-1)) # 60000,28,28 ->60000 784
x_test =  np.reshape(x_test, (num_test_sample,-1))

cls = DecisionTreeClassifier()
cls.fit(x_train,y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))
#print(num_training_sample )
#print(num_test_sample)
#cls = RandomForestClassifier()
#cls.fit(x_train,y_train)
#cv2.imshow("image",x_train[48000])
#cv2.waitKey(0)