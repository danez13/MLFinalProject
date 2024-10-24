"""
- The following script standardizes the dataset using a columns standard deviation and mean
- all dataset categories (train001-train004, test001 - test 004) are done at the same time
"""

from sklearn import preprocessing
import pandas as pd

for i in range(1,5):
    train_df = pd.read_csv(f"Processed_Data/train_FD00{i}.csv")
    test_df = pd.read_csv(f"Processed_Data/test_FD00{i}.csv")

    # seperate data that is going to be normalized
    cols_normalize = train_df.columns.difference(['ID','cycle','RUL','failure_within_w1'])
    cols_normalize = test_df.columns.difference(['ID','cycle','RUL','failure_within_w1'])

    # normalize
    scaler = preprocessing.StandardScaler()
    norm_train_df = pd.DataFrame(scaler.fit_transform(train_df[cols_normalize]), columns=cols_normalize, index=train_df.index)
    norm_test_df = pd.DataFrame(scaler.fit_transform(test_df[cols_normalize]), columns=cols_normalize, index=test_df.index)

    # Join the normalized and non-normalized data.
    train_join_df = train_df[['ID','cycle','RUL','failure_within_w1']].join(norm_train_df)
    test_join_df = test_df[['ID','cycle','RUL','failure_within_w1']].join(norm_test_df)


    train_df = train_join_df.reindex(columns = train_df.columns)
    test_df = test_join_df.reindex(columns = test_df.columns)

    train_df.to_csv(f"Processed_Data/train_FD00{i}.csv",index=False)
    test_df.to_csv(f"Processed_Data/test_FD00{i}.csv",index=False)





        