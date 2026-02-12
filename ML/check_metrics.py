"""Quick script to extract final metrics after tuning"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())
import warnings; warnings.filterwarnings('ignore')
import contextlib
from io import StringIO
from service.data_setup import load_train_csv, split_train_validation
from service.preprocessing.cleansing import fill_missing_values
from service.preprocessing.featureExtraction import *
from service.preprocessing.encoding import one_hot_encode_and_align, drop_original_columns
from service.modeling.metrics import evaluate_binary
from service.modeling.training import train_with_tuned_params

X, y = load_train_csv('data/hotel_bookings_train.csv')
X = fill_missing_values(X)
X_tr, X_val, y_tr, y_val = split_train_validation(X, y, random_state=42)

with contextlib.redirect_stdout(StringIO()):
    X_tr, X_val = add_total_guests_and_is_alone(X_tr, X_val)
    X_tr, X_val = add_has_company(X_tr, X_val)
    X_tr, X_val = add_is_FB_meal(X_tr, X_val)
    X_tr, X_val = process_adr_iqr(X_tr, X_val)
    X_tr, X_val = add_total_stay(X_tr, X_val)
    X_tr, X_val = process_lead_time(X_tr, X_val)
    X_tr, X_val = map_hotel_type(X_tr, X_val)
    X_tr, X_val = drop_original_columns(X_tr, X_val)
    X_tr, X_val = one_hot_encode_and_align(X_tr, X_val)

# Optuna best params from tuning run
best_params = {
    'max_depth': 3,
    'gamma': 0.8154614284548342,
    'reg_alpha': 1.4137146876952342,
    'reg_lambda': 3.780532256184443,
    'min_child_weight': 8,
    'subsample': 0.6222133955202271,
    'colsample_bytree': 0.7075397185632818,
    'scale_pos_weight': 2.347607178575389,
    'learning_rate': 0.07296312489711157,
}

with contextlib.redirect_stdout(StringIO()):
    model = train_with_tuned_params(X_tr, y_tr, best_params, random_state=42)

tr = evaluate_binary(y_tr, model.predict(X_tr), model.predict_proba(X_tr)[:,1])
va = evaluate_binary(y_val, model.predict(X_val), model.predict_proba(X_val)[:,1])

print("---TRAIN---")
print(f"acc={tr.accuracy:.4f}")
print(f"pre={tr.precision:.4f}")
print(f"rec={tr.recall:.4f}")
print(f"f1={tr.f1:.4f}")
print(f"auc={tr.auc:.4f}")
print("---VAL---")
print(f"acc={va.accuracy:.4f}")
print(f"pre={va.precision:.4f}")
print(f"rec={va.recall:.4f}")
print(f"f1={va.f1:.4f}")
print(f"auc={va.auc:.4f}")
print("---GAP---")
print(f"f1_gap={abs(tr.f1-va.f1):.4f}")
print(f"auc_gap={abs(tr.auc-va.auc):.4f}")
