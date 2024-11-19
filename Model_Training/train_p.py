import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

try:
    os.mkdir("Model_Analysis/Perceptron")
except:
    pass
max_iter = 300
with open("Model_Analysis/Perceptron/Accuracy.txt","w+") as file:
    for i in range (1,5):

        train_df = pd.read_csv(f"Processed_Data/train_FD00{i}.csv")
        test_df = pd.read_csv(f"Processed_Data/test_FD00{i}.csv")

        # train with sensors
        X_train = train_df[["s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]].values
        y_train = train_df[["failure_within_w1"]].values

        # test with sensors
        X_test = test_df[["s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]].values
        y_test = test_df[["failure_within_w1"]].values

        param_grid = {
            "hidden_layer_sizes":[(50,),(100,),(50,50,),(100,100,),(50,50,50,),(100,100,100,)],
            "activation": ['identity', 'logistic', 'tanh', 'relu'],
            "solver": ['lbfgs','sgd','adam'],
            "alpha": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
            "learning_rate": ["constant",'invscaling', 'adaptive'],
        }

        model = MLPClassifier(max_iter=500)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)

        grid_search.fit(X_train,np.ravel(y_train))

        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        print("best params: ",grid_search.best_params_)
        print("best model: ", grid_search.best_estimator_)
        # model.fit(X_train, np.ravel(y_train))

        # y_pred = model.predict(X_test)

        # score = model.score(X_test,np.ravel(y_test))
        # cm = confusion_matrix(y_test,y_pred)
        # cm_display = ConfusionMatrixDisplay(cm)
        # cm_display.plot()
        # plt.savefig(f"Model_Analysis/Perceptron/FD00{i}.png")
        # plt.close()

