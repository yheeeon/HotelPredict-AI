import xgboost as xgb


def build_xgb_classifier(random_state=42, **kwargs):
    """
    XGBClassifier를 생성합니다.

    기본값은 현재 코드베이스의 값을 유지하되,
    kwargs로 전달된 파라미터가 있으면 해당 값을 우선 사용합니다.
    (Optuna 튜닝 결과를 직접 전달할 수 있음)
    """
    from xgboost import XGBClassifier

    # 기본 하이퍼파라미터 (현재 코드베이스 기준)
    defaults = dict(
        max_depth=9,
        learning_rate=0.05,
        n_estimators=2000,
        subsample=0.95,
        colsample_bytree=0.95,
        colsample_bylevel=0.95,
        colsample_bynode=0.95,
        scale_pos_weight=3.3,
        reg_alpha=0.0,
        reg_lambda=0.5,
        min_child_weight=1,
        gamma=0.0,
        early_stopping_rounds=150,
        eval_metric='logloss',
    )

    # kwargs가 있으면 기본값을 덮어쓰기
    defaults.update(kwargs)

    return XGBClassifier(
        random_state=random_state,
        use_label_encoder=False,
        tree_method='hist',
        grow_policy='lossguide',
        max_leaves=0,
        max_bin=512,
        objective='binary:logistic',
        booster='gbtree',
        verbosity=0,
        **defaults,
    )
