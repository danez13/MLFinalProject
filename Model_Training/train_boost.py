import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score,recall_score,accuracy_score,precision_score
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv(f"Processed_Data/train_FD001.csv")
test_df = pd.read_csv(f"Processed_Data/test_FD001.csv")

sensors = ["s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]

# train with sensors
X_train = train_df[sensors].values
y_train = train_df[["failure_within_w1"]].values
# test with sensors
X_test = test_df[sensors].values
y_test = test_df[["failure_within_w1"]].values


# Initialize the AdaBoost classifier
adaboost = AdaBoostClassifier(algorithm='SAMME', learning_rate=0.01, n_estimators=100, random_state=42)


adaboost.fit(X_train,np.ravel(y_train))

y_pred = adaboost.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, np.ravel(y_pred))
print(f"accuracy: {accuracy}")
cm = confusion_matrix(y_test,y_pred)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()

p = precision_score(y_test,y_pred)
print(f"precision score: {p}")

r=recall_score(y_test,y_pred)
print(f"recall score: {r}")

f1 = f1_score(y_test,y_pred)
print(f"f1 score: {f1}")
