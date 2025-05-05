import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_demo(X_train, y_train, X_val, y_val):
    """
    0.9363
    """
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=1)
    model.fit(X_train, y_train)
    return model


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score

def train_v1(X_train, y_train, X_val, y_val):
    """
    0.9363 没有区别
    评价指标：将 XGBClassifier 的 eval_metric 改为 auc
    """
    xgb_model = xgb.XGBClassifier(eval_metric='auc', random_state=1, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    return xgb_model



def train_v2(X_train, y_train, X_val, y_val):
    """
    0.9415
    评价指标：将 XGBClassifier 的 eval_metric 改为 auc
    引入 RandomForestClassifier，并通过 VotingClassifier 进行软投票集成。
    """
    xgb_model = xgb.XGBClassifier(eval_metric='auc', random_state=1, n_jobs=-1)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

    # Voting Classifier (Ensemble)
    ensemble_model = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model)],
        voting='soft'
    )

    ensemble_model.fit(X_train, y_train)
    return ensemble_model

from xgboost import XGBClassifier
from skopt import BayesSearchCV
import numpy as np

def train_v3(X_train, y_train, X_val, y_val, NJOBS=28):
    """
    使用贝叶斯优化对 XGBClassifier 和 RandomForestClassifier 进行参数优化，并通过 VotingClassifier 集成。
    """
    # 贝叶斯优化搜索空间
    xgb_param_space = {
        'n_estimators': (50, 500),
        'max_depth': (3, 15),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'gamma': (0, 5),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10)
    }

    rf_param_space = {
        'n_estimators': (50, 500),
        'max_depth': (3, 15),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 10),
        'max_features': (0.5, 1.0)
    }

    # 贝叶斯优化 XGBoost
    xgb_model = XGBClassifier(eval_metric='auc', random_state=1, n_jobs=1)
    xgb_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=xgb_param_space,
        n_iter=30,  # 调整为更高的值以获得更好的结果
        cv=3,
        scoring='roc_auc',
        n_jobs=NJOBS,
        random_state=1,
        verbose=3
    )
    print("Optimizing XGBoost parameters...")
    xgb_search.fit(X_train, y_train)
    print(f"Best XGBoost parameters: {xgb_search.best_params_}")

    rf_model = RandomForestClassifier(random_state=1, n_jobs=4)
    rf_search = BayesSearchCV(
        estimator=rf_model,
        search_spaces=rf_param_space,
        n_iter=30,
        cv=3,
        scoring='roc_auc',
        n_jobs=max(NJOBS//4, 1),
        random_state=1,
        verbose=3
    )
    print("Optimizing RandomForest parameters...")
    rf_search.fit(X_train, y_train)
    print(f"Best RandomForest parameters: {rf_search.best_params_}")

    best_xgb = xgb_search.best_estimator_
    best_rf = rf_search.best_estimator_

    # Voting Classifier (Ensemble)
    ensemble_model = VotingClassifier(
        estimators=[('xgb', best_xgb), ('rf', best_rf)],
        voting='soft',
        n_jobs=28
    )

    print("Training the ensemble model...")
    ensemble_model.fit(X_train, y_train)
    return ensemble_model