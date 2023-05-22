import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# def convert_level(level):
   # if level == "some high school":
     #  level = "high school"
   # return level

data = pd.read_csv("StudentScore.xls - StudentScore.xls.csv", delimiter=",")
# Ignore non-numeric columns
#data = data.select_dtypes(include=[float, int])
#print(data.corr())
target = "math score"
#sn.histplot(data["math score"])
#plt.title("Math score distribution")
#plt.savefig("MathDistribution.png")
x = data.drop(target, axis=1)
# x["parental level of education"] = x["parental level of education"].apply(convert_level)
# print(x["parental level of education"].unique())
y = data[target]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
#imputer = SimpleImputer(strategy='mean')

# Using imputer to fit the NaN data
#x["reading score"] = imputer.fit_transform(x[["reading score"]])
#scaler = StandardScaler()
#x["reading score"] = scaler.fit_transform(x[["reading score"]])
#print(x["reading score"])

#print(x["gender"].unique()) # Check data type for processing process



# Using the pipeline Imputer for faster data refilled


num_transformer = Pipeline(steps=[     # Step is the list of tuple
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
                    "master's degree"] # Define from the least important to the most important - this suits yourself
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")), # For strategy equals constant, we will have to set the fill_value with defined value
    ("encoder", OrdinalEncoder(categories=[education_values,gender_values,lunch_values,test_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encoder", OneHotEncoder(sparse=False))
])
#result = ord_transformer.fit_transform(x_train[["parental level of education"]])
#for i, j in zip(x_train["parental level of education"], result):
#    print("Before {} After {}".format(i, j))
#result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])
#for i, j in zip(x_train["race/ethnicity"], result):
#     print("Before {}. After {}".format(i,j))
preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ordinal_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nominal_features", nom_transformer, ["race/ethnicity"]),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print("MAE {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE {}".format(mean_squared_error(y_test, y_predict)))
print("R2 {}".format(r2_score(y_test, y_predict)))
for i, j in zip(y_test, y_predict):
    print("Actual {}. Predict {}".format(i, j))