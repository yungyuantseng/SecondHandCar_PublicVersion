from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def distance_transform(x: Union[int, float, str]):
    """Column of pandas dataframe to transform

    Args:
        x (str, int, float): column of data

    Returns:
        _type_: _description_
    """
    if isinstance(x, str):
        if 'MILE' in x:
            x = x.replace(' MILE', '')
            return int(x)*1.6
        elif 'KM' in x:
            x = x.replace(' KM', '')
            return int(x)
    else:
        return float(x)


def parse_car_body(x: str):
    """column of pandas dataframe to transform

    Args:
        x (str): denote the newness of the car

    Returns:
        int: label encoding
    """
    if isinstance(x, str):
        if x == 'A+':
            return 8
        elif x == 'A':
            return 7
        elif x == 'A':
            return 6
        elif x == 'A.W':
            return None
        elif x == 'B+':
            return 5
        elif x == 'B':
            return 4
        elif x == 'B.W':
            return None
        elif x == 'C':
            return 3
        elif x == 'C.W':
            return None
        elif x == 'D':
            return 2
        elif x == 'D.W':
            return None
        elif x == 'E':
            return 1
        elif x == 'E.W':
            return None   
    else:
        return None


def parse_car_interior(x: str):
    """column of pandas dataframe to transform

    Args:
        x (str): newness of the car interior

    Returns:
        int: label encoding
    """
    if isinstance(x, str):
        if x == 'A':
            return 4
        elif x == 'B':
            return 3
        elif x == 'C':
            return 2
        elif x == 'D':
            return 1
        else:
            return None
    else:
        return None


def onehot_encode(df: pd.DataFrame, col_to_encode: list):
    """One hot適合用在沒有順序、大小等關係的離散資料

    Args:
        df (pd.DataFrame): original dataframe
        col_to_encode (list): column to encode

    Returns:
        df: df after one encoding
    """
    for col in col_to_encode:
        one_hot_df = pd.DataFrame(pd.get_dummies(df[col]))
        df = pd.concat([df, one_hot_df], axis=1)
    return df


def normalizer(column_to_norm: list, df: pd.DataFrame):
    """normalize data in the dataframe

    Args:
        column_to_norm (list): the column that need to normalized
        df (pd.DataFrame): the origin dataframe

    Returns:
        df: the dataframe after normalized
        scaler: the scale of normalized, can used to inverse normalized
    """
    df_norm = df[column_to_norm]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df_norm)
    df_norm[column_to_norm] = normalized_data
    return df_norm, scaler


def reverse_normalize_data(
        df: pd.DataFrame,
        scaler: MinMaxScaler,
        columns_to_reverse: list
        ):

    inv_data = scaler.inverse_transform(df[columns_to_reverse])
    df[columns_to_reverse] = inv_data
    return df
