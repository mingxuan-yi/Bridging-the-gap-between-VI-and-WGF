import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def dataset(path):
    
    df = pd.read_csv(path)

    x, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values.reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    scaler = MinMaxScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

    input_dim = x_train.shape[1]

    return x_train, y_train, x_test, y_test