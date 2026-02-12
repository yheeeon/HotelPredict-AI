import os
import sys
import warnings

import numpy as np
import pandas as pd

# Suppress ALL warnings and verbose output
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

# Suppress pandas output
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10)

# Redirect stdout temporarily to suppress debug output
import contextlib
from io import StringIO

# Ensure current directory is importable
sys.path.insert(0, os.getcwd())

from service.data_setup import load_train_csv, load_test_csv, split_train_validation
from service.preprocessing.cleansing import fill_missing_values
from service.preprocessing.featureExtraction import (
    add_total_guests_and_is_alone,
    add_has_company,
    add_is_FB_meal,
    process_adr_iqr,
    add_total_stay,
    process_lead_time,
    map_hotel_type,
)
from service.preprocessing.encoding import one_hot_encode_and_align, drop_original_columns, apply_target_encoding
from service.modeling.metrics import evaluate_binary, format_metrics
from service.modeling.training import train_xgb_classifier, train_with_tuned_params
from service.modeling.tuning import run_optuna_tuning


def main() -> None:
    """
    준비된 train 데이터를 train/validation으로 분할하여 모델 성능을 검증
    """
    print("=== Hotel Booking Cancellation 모델 성능 최적화 ===")
    
    # 1. 데이터 로드 및 전처리
    data_dir = os.path.join('data')
    train_path = os.path.join(data_dir, 'hotel_bookings_train.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train 데이터 파일이 없습니다: {train_path}")
    
    print(f"Train 데이터 로드: {train_path}")
    X, y = load_train_csv(train_path)
    X = fill_missing_values(X)
    
    # 2. Train/Validation 분할
    X_tr, X_val, y_tr, y_val = split_train_validation(X, y, random_state=42)
    print(f"Train: {X_tr.shape}, Validation: {X_val.shape}")
    
    # 3. 피처 엔지니어링 및 인코딩
    print("피처 엔지니어링 및 Target Encoding 수행 중... 🔧")
    
    with contextlib.redirect_stdout(StringIO()):
        # 기본 FE
        X_tr, X_val = add_total_guests_and_is_alone(X_tr, X_val)
        X_tr, X_val = add_has_company(X_tr, X_val)
        X_tr, X_val = add_is_FB_meal(X_tr, X_val)
        X_tr, X_val = process_adr_iqr(X_tr, X_val)
        X_tr, X_val = add_total_stay(X_tr, X_val)
        X_tr, X_val = process_lead_time(X_tr, X_val)
        X_tr, X_val = map_hotel_type(X_tr, X_val)

        # Target Encoding (country, agent, company)
        X_tr, X_val = apply_target_encoding(
            X_tr, X_val, y_tr, cols=['country', 'agent', 'company']
        )

        # 불필요 컬럼 제거 및 One-Hot (드롭 로직에서 country/agent 원본은 지우고 deposit_type은 유지하도록 수정됨)
        X_tr, X_val = drop_original_columns(X_tr, X_val)
        X_tr, X_val = one_hot_encode_and_align(X_tr, X_val)
    
    print(f"✅ 일반화 준비 완료! 최종 피처 수: {X_tr.shape[1]}")

    # 4. Optuna Hyperparameter Tuning
    print("\n" + "="*50)
    print("🔍 Optuna Bayesian Tuning (Gap <= 0.05 도전)")
    print("="*50)
    best_params, study = run_optuna_tuning(
        X_tr, y_tr, n_trials=10, n_splits=3, random_state=42
    )

    # 5. 최종 모델 학습
    print("\n" + "🚀"*25)
    print("🏆 최적 하이퍼파라미터 모델 학습 🏆")
    print("🚀"*25)
    model = train_with_tuned_params(X_tr, y_tr, best_params, random_state=42)
    
    # 평가
    y_tr_pred = model.predict(X_tr)
    y_val_pred = model.predict(X_val)
    y_tr_proba = model.predict_proba(X_tr)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]

    tr_metrics = evaluate_binary(y_tr, y_tr_pred, y_tr_proba)
    val_metrics = evaluate_binary(y_val, y_val_pred, y_val_proba)

    print("\n" + "🎯"*25)
    print("🏆 최종 모델 성능 평가 🏆")
    print("🎯"*25)
    print(format_metrics('📊 훈련 데이터:', tr_metrics))
    print(format_metrics('🔍 검증 데이터:', val_metrics))
    
    gap = abs(tr_metrics.f1 - val_metrics.f1)
    print(f"📈 F1 차이(Gap): {gap:.4f} ", end="")
    if gap <= 0.05: print("✅ 성공!")
    else: print("❌ 추가 개선 필요")
    print("🎯"*25)
    
    if val_metrics.f1 >= 0.70 and gap <= 0.05:
        print("✅ 목표 달성! Test 데이터 예측 수행.")
        predict_test_data(model, X_tr, X_val)
    else:
        print("⚠️ 아직 목표 미달성 (F1 >= 0.7 && Gap <= 0.05)")
        u_input = input("그래도 예측을 수행하시겠습니까? (y/n): ")
        if u_input.lower() == 'y':
            predict_test_data(model, X_tr, X_val)
    
    return model


def predict_test_data(model, X_tr_processed, X_val_processed):
    """
    검증된 모델로 test 데이터 예측 수행
    """
    print("\n" + "="*50)
    print("Test 데이터 예측 수행 (신규 FE/Encoding 적용)")
    print("="*50)
    
    # 데이터 로드
    test_path = os.path.join('data', 'hotel_bookings_test.csv')
    train_path = os.path.join('data', 'hotel_bookings_train.csv')
    
    if not os.path.exists(test_path): return
    
    X_test = load_test_csv(test_path)
    X_train_full, y_train_full = load_train_csv(train_path)
    
    X_test = fill_missing_values(X_test)
    X_train_full = fill_missing_values(X_train_full)
    
    result_data = X_test.copy()
    
    # 파이프라인 동일 적용
    X_train_fe, X_test_fe = add_total_guests_and_is_alone(X_train_full, X_test)
    X_train_fe, X_test_fe = add_has_company(X_train_fe, X_test_fe)
    X_train_fe, X_test_fe = add_is_FB_meal(X_train_fe, X_test_fe)
    X_train_fe, X_test_fe = process_adr_iqr(X_train_fe, X_test_fe)
    X_train_fe, X_test_fe = add_total_stay(X_train_fe, X_test_fe)
    X_train_fe, X_test_fe = process_lead_time(X_train_fe, X_test_fe)
    X_train_fe, X_test_fe = map_hotel_type(X_train_fe, X_test_fe)

    # Target Encoding (훈련 데이터 전체 기준)
    X_train_fe, X_test_fe = apply_target_encoding(
        X_train_fe, X_test_fe, y_train_full, cols=['country', 'agent', 'company']
    )

    X_train_final, X_test_final = drop_original_columns(X_train_fe, X_test_fe)
    X_train_final, X_test_final = one_hot_encode_and_align(X_train_final, X_test_final)
    
    # 예측
    y_pred = model.predict(X_test_final)
    y_pred_proba = model.predict_proba(X_test_final)[:, 1]
    
    result_data['predicted_is_canceled'] = y_pred
    result_data['predicted_probability'] = y_pred_proba
    
    results_dir = os.path.join('data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, 'hotel_booking_predictions.csv')
    result_data.to_csv(result_path, index=False)
    
    print(f"📁 결과 저장 완료: {result_path}")


if __name__ == '__main__':
    main()
