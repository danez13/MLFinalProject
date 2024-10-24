from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("Processed_Data/train_FD001.csv")
test_df = pd.read_csv("Processed_Data/test_FD001.csv")

X_train = train_df[["s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]].values
y_train = train_df[["failure_within_w1"]].values

X_test = test_df[["s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]].values
y_test = test_df[["failure_within_w1"]].values
kernels = ['linear','poly','rbf','sigmoid','precomputed']

model = SVC(C=1,kernel="poly")
model.fit(X_train, np.ravel(y_train))

y_pred = model.predict(X_test)

accuracy_default = accuracy_score(y_test, y_pred)
print(f"Default Model Accuracy: {accuracy_default:.2f}",end="\n\n")
