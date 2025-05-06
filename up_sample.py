import pandas as pd
import matplotlib.pyplot as plt
from UTIL.colorful import *

def up_sample(data: pd.DataFrame, col: str, target_pos_ratio: float = 0.1):
    # 统计正负样本的数量
    pos_count = data[data[col] == 1].shape[0]
    neg_count = data[data[col] == 0].shape[0]

    # 计算当前正样本占比
    current_pos_ratio = pos_count / (pos_count + neg_count)
    print(f"Current positive ratio: {current_pos_ratio}")
    # visualize
    plt.figure(figsize=(12, 8))
    plt.title('Positive Ratio')
    plt.bar(['positive', 'negative'], [pos_count, neg_count])
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('./result/positive_ratio.png')
    plt.cla()


    # 如果当前正样本占比已经满足目标正样本占比，则无需上采样
    if current_pos_ratio >= target_pos_ratio:
        print黄(f"Current positive ratio {current_pos_ratio} is already greater than or equal to target positive ratio ({target_pos_ratio}).")
        return data

    # 计算需要上采样的数量
    up_sample_count = int(neg_count * target_pos_ratio / (1 - target_pos_ratio)) - pos_count
    # 随机选择需要上采样的样本
    up_sample_data = data[data[col] == 1].sample(n=up_sample_count, replace=True, random_state=1)
    # 将上采样的样本与原数据合并
    up_sampled_data = pd.concat([data, up_sample_data], axis=0)
    return up_sampled_data




def up_sample_smote(data: pd.DataFrame, col: str, target_pos_ratio: float = 0.1, random_state: int = 42):
    from imblearn.over_sampling import SMOTE
    """
    使用SMOTE方法进行上采样，保持与原始up_sample相同的接口
    
    参数:
        data: 包含特征和目标列的数据框
        col: 目标列名(二分类，1表示正类)
        target_pos_ratio: 目标正样本比例
        random_state: 随机种子
    
    返回:
        上采样后的数据框
    """
    # 统计正负样本的数量
    pos_count = data[data[col] == 1].shape[0]
    neg_count = data[data[col] == 0].shape[0]

    # 计算当前正样本占比
    current_pos_ratio = pos_count / (pos_count + neg_count)
    print(f"Current positive ratio: {current_pos_ratio}")
    
    # 可视化
    plt.figure(figsize=(12, 8))
    plt.title('Positive Ratio Before SMOTE')
    plt.bar(['positive', 'negative'], [pos_count, neg_count])
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('./result/positive_ratio_before_smote.png')
    plt.close()

    # 如果当前正样本占比已经满足目标正样本占比，则无需上采样
    if current_pos_ratio >= target_pos_ratio:
        print(f"Current positive ratio {current_pos_ratio} is already greater than or equal to target positive ratio ({target_pos_ratio}).")
        return data

    # 计算需要达到的正样本数量
    desired_pos = int(neg_count * target_pos_ratio / (1 - target_pos_ratio))
    
    # 设置SMOTE的采样策略
    sampling_strategy = {1: desired_pos}  # 只对正类上采样到目标数量
    
    # 分离特征和目标
    X = data.drop(columns=[col])
    y = data[col]
    
    # 应用SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(5, pos_count - 1),  # 确保k_neighbors不超过正样本数-1
        random_state=random_state
    )
    
    X_res, y_res = smote.fit_resample(X, y)
    
    # 合并回DataFrame
    resampled_data = pd.concat([pd.DataFrame(X_res), pd.Series(y_res, name=col)], axis=1)
    
    # 验证结果
    new_pos = sum(y_res == 1)
    new_neg = sum(y_res == 0)
    print(f"After SMOTE - Positive: {new_pos}, Negative: {new_neg}, Ratio: {new_pos/(new_pos+new_neg):.4f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.title('Positive Ratio After SMOTE')
    plt.bar(['positive', 'negative'], [new_pos, new_neg])
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('./result/positive_ratio_after_smote.png')
    plt.close()
    
    return resampled_data


def up_sample_ratio_vs_auc(data: pd.DataFrame, col: str, target_pos_ratio: float = 0.5):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    # split data to data_train and data_val
    data_shuffled = data.sample(frac=1, random_state=42)

    train_size = 0.9
    train_data = data_shuffled.iloc[:int(len(data)*train_size)]
    test_data = data_shuffled.iloc[int(len(data)*train_size):]
    X_val = test_data.drop(columns=[col])
    y_val = test_data[col]

    def test_model(model, model_n):
        auc_scores = []
        # from 0. to target_pos_ratio
        pos_ratio_list = [i / 400 for i in range(0, int(target_pos_ratio * 400) + 1)]
        for ratio in pos_ratio_list:
            up_sample_data = up_sample_smote(train_data, col, target_pos_ratio=ratio)
            X_train = up_sample_data.drop(columns=[col])
            y_train = up_sample_data[col]
            
            # up_sample_data = up_sample(data, col, target_pos_ratio=ratio)
            # # train test split
            # X, y = up_sample_data.drop(columns=[col]), up_sample_data[col]
            # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1)

            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred)
            auc_scores.append(auc_score)
            print(f"Positive ratio: {ratio}, AUC score: {auc_score}")
        plt.figure(figsize=(12, 8))
        plt.title(f'AUC vs Positive Ratio {model_n}')
        plt.plot(pos_ratio_list, auc_scores, marker='o')
        plt.xlabel('Positive Ratio')
        plt.ylabel('AUC Score')
        plt.grid(True)
        plt.savefig(f'./result/AUC_vs_Positive_Ratio_{model_n}.png')
        plt.cla()
        return auc_scores, pos_ratio_list
    # xgb
    model_name = 'xgb'
    tmp_model = XGBClassifier(eval_metric='auc', random_state=1, n_jobs=-1)
    auc_scores, pos_ratio_list = test_model(tmp_model, model_name)
    # random forest
    model_name = 'random forest'
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
    auc_scores, pos_ratio_list = test_model(rf_model, model_name)
    return data