"""
- The following script makes the dataset appropriate to be used for classification tasks/models
    - this is done by adding an extra data column that is used as a class label, which indicates that an engine requires mainteinance 
- all dataset categories (RUL001-RUL004, train001-train004, test001 - test 004) are done at the same time
"""

import pandas as pd
import numpy as np

for i in range(1,5):
    train_df = pd.read_csv(f"processed_Data/train_FD00{i}.csv")
    test_df = pd.read_csv(f"processed_Data/test_FD00{i}.csv")

    train_df.sort_values(['ID','cycle'], inplace=True)
    test_df.sort_values(['ID','cycle'], inplace=True)

    rul1 = pd.DataFrame(train_df.groupby('ID')['cycle'].max()).reset_index()
    rul2 = pd.DataFrame(test_df.groupby('ID')['cycle'].max()).reset_index()
    rul1.columns = ['ID', 'max']
    rul2.columns = ['ID', 'max']

    train_df = train_df.merge(rul1, on=['ID'], how='left')
    test_df = test_df.merge(rul2, on=['ID'], how='left')

    train_df['RUL'] = train_df['max'] - train_df['cycle']
    test_df['RUL'] = test_df['max'] - test_df['cycle']

    train_df.drop('max', axis=1, inplace=True)
    test_df.drop('max', axis=1, inplace=True)

    w1 = 30
    train_df['failure_within_w1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
    test_df['failure_within_w1'] = np.where(test_df['RUL'] <= w1, 1, 0 )

    train_df.to_csv(f"processed_Data/train_FD00{i}.csv",index=False)
    test_df.to_csv(f"processed_Data/test_FD00{i}.csv",index=False)