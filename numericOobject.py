import numpy as np
import pandas as pd

def obj_num_cols(data: pd.DataFrame):
    obj_col = list(data.select_dtypes(include=['object']).columns)
    num_col = list(data.select_dtypes(exclude=['object']).columns)
    # print(obj_col)
    # print(num_col)
    return obj_col, num_col





















def get_unique_values(data: pd.DataFrame, col: str):
    """ 获取某一列的所有唯一取值 """
    unique_values = data[col].unique()
    return unique_values

def get_unique_values_for_columns(data: pd.DataFrame, columns: list):
    """ 对多个列应用 get_unique_values 函数 """
    unique_values_dict = {}
    for col in columns:
        unique_values = get_unique_values(data, col)
        unique_values_dict[col] = unique_values
    return unique_values_dict

def _replace(data: pd.DataFrame, col: str, a: list, b):
    if not isinstance(b, list):
        b = list(b)
    data[col] = data[col].replace(a, b)

def replace(data: pd.DataFrame, mapping_dict: dict):
    for col, mapping in mapping_dict.items():
        if col in data.columns:
            data[col] = data[col].replace(mapping.keys(), mapping.values())
        else:
            print(f"Column '{col}' not found in DataFrame.")

def encode_categorical_features(data: pd.DataFrame, encoding_dict: dict):
    for col, mapping in encoding_dict.items():
        if col in data.columns:
            data[col] = data[col].apply(lambda x: mapping.get(x, x))
        else:
            print(f"Column '{col}' not found in DataFrame.")
    return data

# obj_col_unique_dict = {
#     'job': ['admin.', 'services', 'blue-collar', 'entrepreneur', 'management', 'technician', 'housemaid', 'self-employed', 'unemployed', 'retired', 'student', 'unknown'],
#     'marital': ['divorced', 'married', 'single', 'unknown'],
#     'education': ['professional.course', 'high.school', 'basic.9y', 'university.degree', 'unknown', 'basic.4y', 'basic.6y', 'illiterate'],
#     'default': ['no', 'unknown', 'yes'], 
#     'housing': ['yes', 'no', 'unknown'],
#     'loan': ['yes', 'no', 'unknown'],
#     'contact': ['cellular', 'telephone'],
#     'month': ['aug', 'may', 'apr', 'nov', 'jul', 'jun', 'oct', 'dec', 'sep', 'mar'],
#     'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
#     'poutcome': ['failure', 'nonexistent', 'success'],
#     'subscribe': ['no', 'yes']
# }

obj_col_unique_dict = {
    'month': ['aug', 'may', 'apr', 'nov', 'jul', 'jun', 'oct', 'dec', 'sep', 'mar'],
    'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
    'education': ['professional.course', 'high.school', 'basic.9y', 'university.degree', 'unknown', 'basic.4y', 'basic.6y', 'illiterate'],

    'subscribe': ['no', 'yes'],
    'contact': ['cellular', 'telephone'],

    'job': ['admin.', 'services', 'blue-collar', 'entrepreneur', 'management', 'technician', 'housemaid', 'self-employed', 'unemployed', 'retired', 'student', 'unknown'],
    'marital': ['divorced', 'married', 'single', 'unknown'],

    'default': ['yes', 'unknown', 'no'], 
    'housing': ['yes', 'unknown', 'no'],
    'loan':    ['yes', 'unknown', 'no'],
    'poutcome': ['success', 'nonexistent', 'failure'],
}

encoding_dict = {
    'job': {v: i for i, v in enumerate(obj_col_unique_dict['job'])},
    'marital': {v: i for i, v in enumerate(obj_col_unique_dict['marital'])},
    'education': {v: i for i, v in enumerate(obj_col_unique_dict['education'])},
    'default': {v: i for i, v in enumerate(obj_col_unique_dict['default'])},
    'housing': {v: i for i, v in enumerate(obj_col_unique_dict['housing'])},
    'loan': {v: i for i, v in enumerate(obj_col_unique_dict['loan'])},
    'contact': {v: i for i, v in enumerate(obj_col_unique_dict['contact'])},
    'month': {v: i for i, v in enumerate(obj_col_unique_dict['month'])},
    'day_of_week': {v: i for i, v in enumerate(obj_col_unique_dict['day_of_week'])},
    'poutcome': {v: i for i, v in enumerate(obj_col_unique_dict['poutcome'])},
    'subscribe': {v: i for i, v in enumerate(obj_col_unique_dict['subscribe'])}
}


















































def box_cut(data: pd.DataFrame):
    age_bins = [0, 18, 30, 45, 110]
    duration_bins = [-1, 143, 353, 1873, 5149]
    campaign_bins = [-1, 10, 20, 30, 40, 50, 60]
    pdays_bins = [-1, 200, 400, 600, 800, 1000, 1200]
    emp_var_rate_bins = [-4, -1, 0.2, 1.4]
    cons_price_index_bins = [87.40, 90.00, 92.37, 94.73, 97.10, 99.46]
    cons_conf_index_bins = [-53.40, -47.13, -40.99, -37.84, -28.70, -22.55]
    lending_rate3m_bins = [0.4, 1.17, 2.9, 4.1, 5.27]
    nr_employed_bins = [4715, 4915.42, 5115.42, 5315.42, 5498.5]

    data['age'] = pd.cut(data['age'], bins=age_bins, labels=[0, 1, 2, 3]).astype(np.int32)
    data['duration'] = pd.cut(data['duration'], bins=duration_bins, labels=[0, 1, 2, 3]).astype(np.int32)
    data['campaign'] = pd.cut(data['campaign'], bins=campaign_bins, labels=[0, 1, 2, 3, 4, 5]).astype(np.int32)
    data['pdays'] = pd.cut(data['pdays'], bins=pdays_bins, labels=[0, 1, 2, 3, 4, 5]).astype(np.int32)
    data['emp_var_rate'] = pd.cut(data['emp_var_rate'], bins=emp_var_rate_bins, labels=[0, 1, 2]).astype(np.int32)
    data['cons_price_index'] = pd.cut(data['cons_price_index'], bins=cons_price_index_bins, labels=[0, 1, 2, 3, 4]).astype(np.int32)
    data['cons_conf_index'] = pd.cut(data['cons_conf_index'], bins=cons_conf_index_bins, labels=[0, 1, 2, 3, 4]).astype(np.int32)
    data['lending_rate3m'] = pd.cut(data['lending_rate3m'], bins=lending_rate3m_bins, labels=[0, 1, 2, 3]).astype(np.int32)
    data['nr_employed'] = pd.cut(data['nr_employed'], bins=nr_employed_bins, labels=[0, 1, 2, 3]).astype(np.int32)

    return data