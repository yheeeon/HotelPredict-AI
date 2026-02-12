from typing import Any


def train_xgb_classifier(
    X, y, random_state=42,
    max_depth=9,  # F1 최적화: 8→9 (0.01 향상을 위한 논리적 조정)
    learning_rate=0.05,  # F1 최적화: 0.06→0.05 (더 정밀한 학습)
    n_estimators=2000,  # F1 최적화: 1800→2000 (충분한 학습)
    subsample=0.95,  # F1 최적화: 0.9→0.95 (거의 모든 데이터 활용)
    colsample_bytree=0.95,  # F1 최적화: 0.9→0.95 (거의 모든 피처 활용)
    colsample_bylevel=0.95,  # F1 최적화: 0.9→0.95 (레벨별 거의 모든 피처)
    colsample_bynode=0.95,  # F1 최적화: 0.9→0.95 (노드별 거의 모든 피처)
    scale_pos_weight=3.3,  # F1 최적화: 3.2→3.3 (클래스 불균형 처리 미세 조정)
    reg_alpha=0.0,  # F1 최적화: 유지 (피처 활용 극대화)
    reg_lambda=0.5,  # F1 최적화: 0.6→0.5 (L2 정규화 완화로 학습 강화)
    min_child_weight=1,  # F1 최적화: 유지 (세밀한 분할)
    gamma=0.0,  # F1 최적화: 유지 (분할 제한 제거)
    early_stopping_rounds=150,  # F1 최적화: 120→150 (더 많은 학습 허용)
    eval_metric='logloss'  # F1-score 최적화를 위해 logloss 사용
) -> Any:
    from .model import build_xgb_classifier
    model = build_xgb_classifier(
        random_state=random_state,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric=eval_metric
    )
    
    # F1-score 최적화를 위한 커스텀 메트릭과 조기 종료
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    import numpy as np
    
    # 커스텀 F1-score 메트릭 함수
    def f1_eval(y_pred, y_true):
        y_pred_binary = (y_pred > 0.5).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        return 'f1', f1, True  # True는 높을수록 좋음을 의미
    
    # 검증 세트 분할 (F1-score 최적화용)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # 조기 종료를 위한 fit
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # F1-score 최적화를 위한 임계값 찾기
    y_val_proba = model.predict_proba(X_val)[:, 1]
    best_f1 = 0
    best_threshold = 0.5
    
    # F1-score 0.7 달성을 위한 고급 임계값 탐색
    # 1단계: 넓은 범위에서 대략적인 최적값 찾기
    for threshold in np.arange(0.15, 0.85, 0.01):
        y_val_pred = (y_val_proba > threshold).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 2단계: 최적값 주변에서 정밀한 탐색 (0.7 달성용)
    if best_threshold > 0.1:
        start_range = max(0.1, best_threshold - 0.03)
        end_range = min(0.9, best_threshold + 0.03)
        
        for threshold in np.arange(start_range, end_range, 0.0005):
            y_val_pred = (y_val_proba > threshold).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    # 3단계: F1-score 0.7 달성을 위한 추가 최적화
    if best_f1 < 0.7:
        # 0.7 달성을 위한 특별 탐색
        for threshold in np.arange(0.2, 0.6, 0.002):
            y_val_pred = (y_val_proba > threshold).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    # 최적 임계값을 모델에 저장 (커스텀 속성으로)
    model.best_threshold_ = best_threshold
    model.best_f1_ = best_f1
    
    print(f"🎯 F1-score 최적화 완료!")
    print(f"   최적 임계값: {best_threshold:.3f}")
    print(f"   최적 F1-score: {best_f1:.3f}")
    
    return model


def train_with_tuned_params(X, y, best_params: dict, random_state: int = 42):
    """
    Optuna가 찾은 최적 하이퍼파라미터로 모델 학습

    Parameters
    ----------
    X : 훈련 피처 데이터
    y : 훈련 타겟 데이터
    best_params : Optuna best_trial.params 딕셔너리
    random_state : 랜덤 시드
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    import numpy as np

    from .model import build_xgb_classifier

    model = build_xgb_classifier(
        random_state=random_state,
        **best_params,
        n_estimators=2000,
        early_stopping_rounds=100,
        eval_metric='logloss',
    )

    # Early stopping용 검증 세트 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 최적 임계값 탐색 (기존 로직 유지)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.15, 0.85, 0.01):
        y_val_pred = (y_val_proba > threshold).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # 정밀 탐색
    for threshold in np.arange(
        max(0.1, best_threshold - 0.03),
        min(0.9, best_threshold + 0.03),
        0.001
    ):
        y_val_pred = (y_val_proba > threshold).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    model.best_threshold_ = best_threshold
    model.best_f1_ = best_f1

    print(f"  최적 임계값: {best_threshold:.3f}")
    print(f"  내부 검증 F1: {best_f1:.3f}")

    return model



