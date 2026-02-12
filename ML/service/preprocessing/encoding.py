from typing import Tuple, List
import pandas as pd
import numpy as np


def apply_target_encoding(
    X_tr: pd.DataFrame, 
    X_te: pd.DataFrame, 
    y_tr: pd.Series, 
    cols: List[str], 
    smoothing: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    범주형 피처를 타겟 데이터의 평균값(취소율)으로 변환합니다.
    Smoothing을 적용하여 오버피팅을 방지합니다.
    """
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    
    global_mean = y_tr.mean()
    
    for col in cols:
        if col not in X_tr.columns:
            continue
            
        # 훈련 데이터에서 각 범주별 통계 계산
        agg = y_tr.groupby(X_tr[col]).agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        
        # Smoothing 적용: (count * mean + smoothing * global_mean) / (count + smoothing)
        smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
        
        # 매핑 적용
        X_tr[f'{col}_trg'] = X_tr[col].map(smooth).fillna(global_mean)
        X_te[f'{col}_trg'] = X_te[col].map(smooth).fillna(global_mean)
        
        # 원본 컬럼 제거 (나중에 drop_original_columns에서 일괄 처리해도 되지만 여기서 명시적으로 제거 가능)
        # 여기서는 유지하고 최종 단계에서 드롭하도록 설계함
        
    return X_tr, X_te


def one_hot_encode_and_align(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    
    # Target encoding으로 이미 변환된 문자열 컬럼들이 있을 수 있으므로 
    # 현재 남아있는 모든 object 타입을 원-핫 인코딩
    cat_cols = X_tr.select_dtypes(include='object').columns.tolist()
    
    if len(cat_cols) > 0:
        X_tr = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True)
        X_te = pd.get_dummies(X_te, columns=cat_cols, drop_first=True)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    return X_tr, X_te


def drop_original_columns(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    
    # 삭제 대상: 원본 데이터 중 수치형으로 변환되었거나 노이즈가 너무 심한 것들
    columns_to_drop = [
        'hotel', 'lead_time', 'adr', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'total_guests', 'reserved_room_type',
        'assigned_room_type', 'customer_type',
        'reservation_status',       # 직접적 유출 피처
        'reservation_status_date',  # 날짜 (의미 없음)
        'arrival_date_full',        # 날짜 (의미 없음)
        # 'deposit_type',           <-- 복구 (매우 강력한 시그널)
        'agent',                    # Target encoded로 대체
        'company',                  # Target encoded로 대체
        'country',                  # Target encoded로 대체
    ]
    
    X_tr.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    X_te.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    return X_tr, X_te
