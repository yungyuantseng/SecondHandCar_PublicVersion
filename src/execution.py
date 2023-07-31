import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from preprocess import (distance_transform, normalizer, onehot_encode,
                        parse_car_body, parse_car_interior,
                        reverse_normalize_data)


def main():
    data = pd.read_csv('./secret_df.csv')

    # 處理空白行
    data = data.rename(columns={'Unnamed: 1': 'drop'})
    data = data.drop(['drop'], axis=1)

    data['里程'] = data['里程'].apply(distance_transform)
    data['出廠'] = 2023 - data['出廠'].astype(int)
    data['車體'] = data['車體'].apply(parse_car_body)
    data['內裝'] = data['內裝'].apply(parse_car_interior)

    col_to_one_hot = ['品牌', '排檔', '燃油', '驅動', '車門', '擔保']
    data = onehot_encode(df=data, col_to_encode=col_to_one_hot)

    column_to_selected = [
        '成交價',
        '出廠',
        '排氣',
        '內裝',
        '車體',
        'AUDI',
        'BENTLEY',
        'BENZ',
        'BMW',
        'CHEVROLET',
        'CITROEN',
        'DAIHATSU',
        'DFSK',
        'FORD',
        'HINO',
        'HONDA',
        'HYUNDAI',
        'INFINITI',
        'ISUZU',
        'IVECO',
        'JAGUAR',
        'JEEP',
        'KIA',
        'LAND ROVER',
        'LEXUS',
        'LUXGEN',
        'MAHINDRA',
        'MASERATI',
        'MAZDA',
        'MCC',
        'MINI',
        'MITSUBISHI',
        'NISSAN',
        'PEUGEOT',
        'PORSCHE',
        'SKODA',
        'SMART',
        'SSANGYONG',
        'SUBARU',
        'SUZUKI',
        'TESLA',
        'TOYOTA',
        'VOLKSWAGEN',
        'VOLVO',
        '手',
        '手自',
        '自',
        '柴油',
        '柴油電',
        '汽油',
        '汽油/LPG',
        '汽油LPG',
        '油電',
        '電動',
        '電能汽油',
        '電能（增程）',
        '2WD',
        '4WD',
        2.0,
        3.0,
        4.0,
        5.0,
        '不保證',
        '保證'
    ]

    df_selected = data[column_to_selected]
    df_selected = df_selected.fillna(0)

    columns_to_normalize = ['成交價', '出廠', '排氣', '內裝', '車體']
    df_selected_norm, scaler = normalizer(column_to_norm=columns_to_normalize, df=df_selected)

    X_DATA = df_selected_norm.drop(columns=['成交價']).values
    Y_DATA = df_selected_norm['成交價'].values

    X_train, X_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size=0.2, random_state=42)

    # Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    predictions = lr_model.predict(X_test)
    MSE = mean_squared_error(y_test, predictions)

    print('Logistic Regression Training Accuracy:', lr_model.score(X_train, y_train))
    print('Logistic Regression Testing Accuracy:', lr_model.score(X_test, y_test))
    print('MSE Loss', MSE)

    # Random Forest model
    forest = RandomForestRegressor(max_depth=20, random_state=0)
    forest.fit(X_train, y_train)

    predictions = forest.predict(X_test)
    MSE = mean_squared_error(y_test, predictions)
    print('RandomForestRegressor Training Accuracy:', forest.score(X_train, y_train))
    print('RandomForestRegressor Testing Accuracy:', forest.score(X_test, y_test))
    print('MSE Loss', MSE)

    # Decision Tree model
    tree = DecisionTreeRegressor(random_state=0)
    tree.fit(X_train, y_train)

    predictions = tree.predict(X_test)
    MSE = mean_squared_error(y_test, predictions)
    print('DecisionTreeRegressor Training Accuracy:', tree.score(X_train, y_train))
    print('DecisionTreeRegressor Testing Accuracy:', tree.score(X_test, y_test))
    print('MSE Loss', MSE)

    # concat x & y data
    # origin
    original_concat = np.concatenate([final_y_test, X_test], axis=1)
    original_concat = pd.DataFrame(original_concat, column_to_selected)
    original_concat = reverse_normalize_data(
        df=original_concat,
        scaler=scaler,
        columns_to_reverse=columns_to_normalize)

    # predict
    predictions = predictions.reshape(2766, 1)
    final_y_test = y_test.reshape(2766, 1)
    predict_concat = np.concatenate([predictions, X_test], axis=1)
    predict_concat = pd.DataFrame(predict_concat, column_to_selected)
    # reverse normalize data
    predict_concat = reverse_normalize_data(
        df=predict_concat,
        scaler=scaler,
        columns_to_reverse=columns_to_normalize)

    original_concat.to_csv('./original_df.csv')
    predict_concat.to_csv('./predict_df.csv')


if __name__ == '__main__':
    main()
