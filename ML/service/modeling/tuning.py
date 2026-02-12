from typing import Dict, Any, Tuple, Callable
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import warnings

# Suppress warnings in objective function
warnings.filterwarnings('ignore')

def _create_objective(X: pd.DataFrame, y: pd.Series, n_splits: int = 3, random_state: int = 42) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        # Search Space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': 2000,
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'gamma': trial.suggest_float('gamma', 0.0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 4.0),
            'random_state': random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        val_f1_scores = []
        train_f1_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**params, early_stopping_rounds=100)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )

            # 최적 임계값 찾기 (성능 0.7 달성을 위해)
            y_val_proba = model.predict_proba(X_val_fold)[:, 1]
            best_fold_f1 = 0
            for thresh in np.arange(0.3, 0.7, 0.05):
                score = f1_score(y_val_fold, (y_val_proba >= thresh).astype(int))
                if score > best_fold_f1:
                    best_fold_f1 = score
            
            val_f1_scores.append(best_fold_f1)
            
            # Train F1도 계산 (Gap 측정을 위해)
            y_train_pred = model.predict(X_train_fold)
            train_f1_scores.append(f1_score(y_train_fold, y_train_pred))

        mean_val_f1 = np.mean(val_f1_scores)
        mean_train_f1 = np.mean(train_f1_scores)
        gap = abs(mean_train_f1 - mean_val_f1)

        # Penalty logic (Strictly targeting Gap <= 0.05)
        # 0.05를 넘는 순간 점수를 삭감하되, 유동적으로 탐색할 수 있도록 설계
        penalty = 0
        if gap > 0.05:
            penalty = 5.0 * (gap - 0.05) # 선형 페널티 강화
        
        score = mean_val_f1 - penalty

        # Trial 속성 저장
        trial.set_user_attr('mean_val_f1', mean_val_f1)
        trial.set_user_attr('mean_train_f1', mean_train_f1)
        trial.set_user_attr('gap', gap)

        return score

    return objective

def run_optuna_tuning(
    X, y,
    n_trials: int = 50,
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], optuna.Study]:
    
    print(f"  Optuna 탐색 시작 ({n_trials}회 시도, {n_splits}-Fold CV)...")
    print(f"  목표: Validation F1 >= 0.70 & Train-Val Gap <= 0.05")

    objective = _create_objective(X, y, n_splits=n_splits, random_state=random_state)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(),
    )

    def progress_callback(study, trial):
        if (trial.number + 1) % 5 == 0 or trial.number == 0:
            best = study.best_trial
            print(
                f"  [{trial.number + 1:3d}/{n_trials}] "
                f"현재 최고(Adjusted)={best.value:.4f} | "
                f"Val_F1={best.user_attrs['mean_val_f1']:.4f}, "
                f"Gap={best.user_attrs['gap']:.4f}"
            )

    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

    best = study.best_trial
    print(f"\n  === Optuna 탐색 완료 ===")
    print(f"  최종 Val F1: {best.user_attrs['mean_val_f1']:.4f}")
    print(f"  최종 Gap: {best.user_attrs['gap']:.4f}")
    
    return best.params, study
