import numpy as np
import pandas as pd
def is_nan(data, name):
    if data.isna().any().any():
        # 找到包含缺失值的行和列
        nan_locations = np.where(pd.isna(data))
        rows_with_nan = nan_locations[0]
        columns_with_nan = nan_locations[1]
        # 打印包含缺失值的行和列号
        for row, column in zip(rows_with_nan, columns_with_nan):
            print(f"{name} 缺失值在第 {row+1} 行，第 {column+1} 列")
    else:
        print(f"{name} 不存在 NaN")