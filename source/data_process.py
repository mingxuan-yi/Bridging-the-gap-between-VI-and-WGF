import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def dataset(args, path):
    df = pd.read_csv(path)
    
    if args.dataset in ['heart', 'pima']:
        x, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values.reshape((-1, 1))
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

        scaler = MinMaxScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

        input_dim = x_train.shape[1]


    elif args.dataset == 'wine':
        x, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values.reshape((-1, 1))
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

        y_train[y_train >= 6] = 1
        y_train[y_train != 1] = 0

        y_test[y_test >= 6] = 1
        y_test[y_test != 1] = 0

        scaler = MinMaxScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        
    elif args.dataset == 'ionos':
        df.rename(columns={'column_ai': 'label'}, inplace=True)
        df['label'] = df.label.astype('category')
        encoding = {'g': 1, 'b': 0}
        df.label.replace(encoding, inplace=True)

        x, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values.reshape((-1, 1))
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    

        scaler = MinMaxScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        
    return x_train, y_train, x_test, y_test