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

def train_v12(X_train, y_train, X_val, y_val):
    xgb_model = RandomForestClassifier(random_state=1, n_jobs=-1)
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


from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def train_v4(X_train, y_train, X_val, y_val, n_iter=20, n_jobs=28):
    """
    Optimized ensemble model with:
    1. Faster Bayesian optimization with reduced search space
    2. Early stopping for XGBoost
    3. Model selection based on validation AUC
    4. Ensemble methods (Voting and Stacking)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping and model selection
        n_iter: Number of Bayesian optimization iterations
        n_jobs: Number of parallel jobs (for RF and ensemble)
    
    Returns:
        Best performing model (single or ensemble) based on validation AUC
    """
    # Split training data for early stopping
    X_train_part, X_eval, y_train_part, y_eval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1)
    
    # Reduced search spaces for faster optimization
    xgb_param_space = {
        'n_estimators': (100, 300),  # Reduced range
        'max_depth': (3, 10),
        'learning_rate': (0.05, 0.2),  # Narrower range
        'subsample': (0.7, 1.0),
        'colsample_bytree': (0.7, 1.0),
        'gamma': (0, 2),  # Reduced range
        'reg_alpha': (0, 5),  # Reduced range
        'reg_lambda': (0, 5)  # Reduced range
    }
    
    rf_param_space = {
        'n_estimators': (100, 300),  # Reduced range
        'max_depth': (5, 15),
        'min_samples_split': (2, 5),  # Reduced range
        'min_samples_leaf': (1, 3),  # Reduced range
        'max_features': (0.7, 1.0)  # Narrower range
    }
    
    # Optimize XGBoost with early stopping
    print("Optimizing XGBoost...")
    xgb_model = XGBClassifier(
        eval_metric='auc',
        random_state=1,
        n_jobs=1,  # XGBoost must be single-threaded
        early_stopping_rounds=10
    )
    
    xgb_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=xgb_param_space,
        n_iter=n_iter,
        cv=3,
        scoring='roc_auc',
        n_jobs=n_jobs,  # Parallelize the search
        random_state=1,
        verbose=3
    )
    
    xgb_search.fit(X_train_part, y_train_part, eval_set=[(X_eval, y_eval)], verbose=False)
    best_xgb = xgb_search.best_estimator_
    print(f"Best XGBoost AUC: {xgb_search.best_score_:.4f}")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    
    # Optimize RandomForest
    print("\nOptimizing RandomForest...")
    rf_model = RandomForestClassifier(random_state=1, n_jobs=n_jobs//2)
    
    rf_search = BayesSearchCV(
        estimator=rf_model,
        search_spaces=rf_param_space,
        n_iter=n_iter,
        cv=3,
        scoring='roc_auc',
        n_jobs=n_jobs//2,  # Use fewer jobs for RF
        random_state=1,
        verbose=3
    )
    
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Best RandomForest AUC: {rf_search.best_score_:.4f}")
    print(f"Best RandomForest params: {rf_search.best_params_}")
    
    # Evaluate single models on validation set
    xgb_val_pred = best_xgb.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_val_pred)
    
    rf_val_pred = best_rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_val_pred)
    
    print(f"\nValidation AUC - XGBoost: {xgb_auc:.4f}, RandomForest: {rf_auc:.4f}")
    
    # Create ensemble candidates
    models = [
        ('xgb', best_xgb),
        ('rf', best_rf)
    ]
    
    # 1. Soft Voting Ensemble
    voting_model = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=n_jobs
    )
    fit_params = {
        'eval_set': [(X_val, y_val)],
        'verbose': False
    }
    voting_model.fit(X_train, y_train, **fit_params)
    voting_pred = voting_model.predict_proba(X_val)[:, 1]
    voting_auc = roc_auc_score(y_val, voting_pred)
    print(f"Voting Ensemble Validation AUC: {voting_auc:.4f}")
    
    # 2. Stacking Ensemble
    stacking_model = StackingClassifier(
        estimators=models,
        final_estimator=XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=1,
            n_jobs=1
        ),
        n_jobs=n_jobs,
        passthrough=True
    )
    stacking_model.fit(X_train, y_train)
    stacking_pred = stacking_model.predict_proba(X_val)[:, 1]
    stacking_auc = roc_auc_score(y_val, stacking_pred)
    print(f"Stacking Ensemble Validation AUC: {stacking_auc:.4f}")
    
    # Select best model based on validation AUC
    model_performance = {
        'xgb': xgb_auc,
        'rf': rf_auc,
        'voting': voting_auc,
        'stacking': stacking_auc
    }
    
    best_model_name = max(model_performance, key=model_performance.get)
    best_auc = model_performance[best_model_name]
    
    print(f"\nSelected best model: {best_model_name} with AUC: {best_auc:.4f}")
    
    if best_model_name == 'xgb':
        return best_xgb
    elif best_model_name == 'rf':
        return best_rf
    elif best_model_name == 'voting':
        return voting_model
    else:
        return stacking_model




def train_v41(X_train, y_train, X_val, y_val, n_iter=20, n_jobs=28):
    """
    Optimized ensemble model with:
    1. Faster Bayesian optimization with reduced search space
    2. Early stopping for XGBoost
    3. Model selection based on validation AUC
    4. Ensemble methods (Voting and Stacking)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping and model selection
        n_iter: Number of Bayesian optimization iterations
        n_jobs: Number of parallel jobs (for RF and ensemble)
    
    Returns:
        Best performing model (single or ensemble) based on validation AUC
    """
    # Split training data for early stopping
    X_train_part, X_eval, y_train_part, y_eval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1)
    
    # Reduced search spaces for faster optimization
    xgb_param_space = {
        'n_estimators': (100, 300),  # Reduced range
        'max_depth': (3, 10),
        'learning_rate': (0.05, 0.2),  # Narrower range
        'subsample': (0.7, 1.0),
        'colsample_bytree': (0.7, 1.0),
        'gamma': (0, 2),  # Reduced range
        'reg_alpha': (0, 5),  # Reduced range
        'reg_lambda': (0, 5)  # Reduced range
    }
    
    rf_param_space = {
        'n_estimators': (100, 300),  # Reduced range
        'max_depth': (5, 15),
        'min_samples_split': (2, 5),  # Reduced range
        'min_samples_leaf': (1, 3),  # Reduced range
        'max_features': (0.7, 1.0)  # Narrower range
    }
    
    # Optimize XGBoost with early stopping
    print("Optimizing XGBoost...")
    xgb_model = XGBClassifier(
        eval_metric='auc',
        random_state=1,
        n_jobs=1,  # XGBoost must be single-threaded
        early_stopping_rounds=10
    )
    
    xgb_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=xgb_param_space,
        n_iter=n_iter,
        cv=3,
        scoring='roc_auc',
        n_jobs=n_jobs,  # Parallelize the search
        random_state=1,
        verbose=3
    )
    
    xgb_search.fit(X_train_part, y_train_part, eval_set=[(X_eval, y_eval)], verbose=False)
    best_xgb = xgb_search.best_estimator_
    print(f"Best XGBoost AUC: {xgb_search.best_score_:.4f}")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    
    # Optimize RandomForest
    print("\nOptimizing RandomForest...")
    rf_model = RandomForestClassifier(random_state=1, n_jobs=n_jobs//2)
    
    rf_search = BayesSearchCV(
        estimator=rf_model,
        search_spaces=rf_param_space,
        n_iter=n_iter,
        cv=3,
        scoring='roc_auc',
        n_jobs=n_jobs//2,  # Use fewer jobs for RF
        random_state=1,
        verbose=3
    )
    
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Best RandomForest AUC: {rf_search.best_score_:.4f}")
    print(f"Best RandomForest params: {rf_search.best_params_}")
    
    # Evaluate single models on validation set
    xgb_val_pred = best_xgb.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_val_pred)
    
    rf_val_pred = best_rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_val_pred)
    
    print(f"\nValidation AUC - XGBoost: {xgb_auc:.4f}, RandomForest: {rf_auc:.4f}")
    
    xgb_for_ensemble = XGBClassifier(
        **xgb_search.best_params_,
        eval_metric='auc',
        random_state=1,
        n_jobs=1,
        early_stopping_rounds=None  # 禁用early stopping
    )
    
    # 创建集成候选模型
    models = [
        ('xgb', xgb_for_ensemble),  # 使用禁用early stopping的版本
        ('rf', best_rf)
    ]
    
    # 1. Soft Voting Ensemble
    print("\nTraining Voting Ensemble...")
    voting_model = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=n_jobs
    )
    voting_model.fit(X_train, y_train)
    voting_pred = voting_model.predict_proba(X_val)[:, 1]
    voting_auc = roc_auc_score(y_val, voting_pred)
    print(f"Voting Ensemble Validation AUC: {voting_auc:.4f}")
    
    # 2. Stacking Ensemble
    print("\nTraining Stacking Ensemble...")
    stacking_model = StackingClassifier(
        estimators=models,
        final_estimator=XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=1,
            n_jobs=1,
            early_stopping_rounds=None  # 禁用early stopping
        ),
        n_jobs=n_jobs,
        passthrough=True
    )
    stacking_model.fit(X_train, y_train)
    stacking_pred = stacking_model.predict_proba(X_val)[:, 1]
    stacking_auc = roc_auc_score(y_val, stacking_pred)
    print(f"Stacking Ensemble Validation AUC: {stacking_auc:.4f}")
    
    # Select best model based on validation AUC
    model_performance = {
        'xgb': xgb_auc,
        'rf': rf_auc,
        'voting': voting_auc,
        'stacking': stacking_auc
    }
    
    best_model_name = max(model_performance, key=model_performance.get)
    best_auc = model_performance[best_model_name]
    
    print(f"\nSelected best model: {best_model_name} with AUC: {best_auc:.4f}")
    
    if best_model_name == 'xgb':
        return best_xgb
    elif best_model_name == 'rf':
        return best_rf
    elif best_model_name == 'voting':
        return voting_model
    else:
        return stacking_model