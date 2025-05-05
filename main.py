"""
    * 原始示范程序：0.9363
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# 读取数据
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

from numericOobject import obj_num_cols, encode_categorical_features, encoding_dict, box_cut
train_obj_col, train_num_col = obj_num_cols(train_data)
test_obj_col, test_num_col = obj_num_cols(test_data)
print("train_obj_col:", train_obj_col)
print("train_num_col:", train_num_col)
print("test_obj_col:", test_obj_col)
print("test_num_col:", test_num_col)

train_data = encode_categorical_features(train_data, encoding_dict)
test_data = encode_categorical_features(test_data, encoding_dict)

o, _ = obj_num_cols(train_data); assert len(o) == 0
o, _ = obj_num_cols(test_data); assert len(o) == 0
_n = 20; print('-'*_n, 'class features 处理完成', '-'*_n)

train_data = box_cut(train_data)
test_data = box_cut(test_data)


from nan import is_nan
is_nan(train_data, 'train_data')
is_nan(test_data, 'test_data')
_n = 20; print('-'*_n, 'Nan缺失值 处理完成', '-'*_n)

from importance import feature_importance
train_data = feature_importance(train_data, 'id', 'subscribe', test=test_data)

_n = 20; print('-'*_n, '低相关性特征已经剔除: ', '-'*_n)
print(train_data.head())

print(test_data.columns)
X_pred = test_data.drop(['id', 'education', 'housing'], axis=1)
X = train_data.drop(['id', 'subscribe', 'education', 'housing'], axis=1)
y = train_data['subscribe']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1)



from model import train_v1 as train
from eval import eval
model = train(X_train, y_train, X_val, y_val)


eval_result = eval({
    'y_proba': model.predict_proba(X_val)[:, 1],
    'y_pred': model.predict(X_val)
}, y_val)


from res import output_res
output_res(model, X_pred, test_data['id'])