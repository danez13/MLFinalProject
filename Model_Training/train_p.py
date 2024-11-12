import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

try:
    os.mkdir("Model_Analysis/Perceptron")
except:
    pass
with open("Model_Analysis/SVM/Accuracy.txt","w+") as file:
    for i in range (1,5):
        train_df = pd.read_csv(f"Processed_Data/train_FD00{i}.csv")
        test_df = pd.read_csv(f"Processed_Data/test_FD00{i}.csv")

        # train with sensors
        X_train = train_df[["s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]].values
        y_train = train_df[["failure_within_w1"]].values

        # test with sensors
        X_test = test_df[["s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]].values
        y_test = test_df[["failure_within_w1"]].values

        model = MLPClassifier()
        model.fit(X_train, np.ravel(y_train))

        y_pred = model.predict(X_test)

        score = model.score(X_test,np.ravel(y_test))
        cm = confusion_matrix(y_test,y_pred)
        cm_display = ConfusionMatrixDisplay(cm)
        file.write(f"FD00{i}: {score}\n")
        cm_display.plot()
        plt.savefig(f"Model_Analysis/Perceptron/FD00{i}.png")
        plt.close()

