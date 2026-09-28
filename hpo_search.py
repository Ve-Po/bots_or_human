"""Подбор гиперпараметров с сохранением результатов на диск (hpo_log.csv).

Каждая проверенная комбинация параметров дописывается в CSV сразу после расчёта — прогресс
не теряется, даже если процесс прервать. Повторный запуск пропускает уже проверенные
комбинации (по хэшу параметров) и продолжает поиск дальше — то есть его можно
останавливать/возобновлять сколько угодно раз, в том числе в других сессиях.

Использование:
    python hpo_search.py --family histgb --n_iter 30
    python hpo_search.py --family extratrees --n_iter 20
    python hpo_search.py --family histgb --n_iter 30   # повторный запуск — добавит ещё 30
                                                         # НОВЫХ комбинаций поверх уже сохранённых

Результаты: hpo_log.csv (все попытки) + hpo_best_<family>.pkl 
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterSampler

from core import precision_at_recall, numeric_frame, SEED

LOG_PATH = "hpo_log.csv"
LOG_FIELDS = ["timestamp", "family", "params_hash", "params_json", "mean_score", "std_score", "fold_scores"]

PARAM_SPACES = {
    "histgb": {
        "max_iter": [150, 200, 300, 500],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 30, 50],
        "l2_regularization": [0.0, 0.5, 1.0, 2.0],
        "max_depth": [None, 4, 6, 8],
    },
    "extratrees": {
        "n_estimators": [300, 500, 800, 1000],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": [0.5, 0.6, 0.8, "sqrt"],
    },
    "randomforest": {
        "n_estimators": [300, 500, 800],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": ["sqrt", 0.6, 0.8],
    },
}

MODEL_CTORS = {
    "histgb": lambda p: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, **p),
    "extratrees": lambda p: ExtraTreesClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **p),
    "randomforest": lambda p: RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **p),
}

try:
    import lightgbm as lgb
    PARAM_SPACES["lightgbm"] = {
        "n_estimators": [300, 600, 1000, 1500, 2000],
        "learning_rate": [0.02, 0.04, 0.06],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 40, 80],
        "subsample": [0.75, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.0, 0.5, 2.0, 5.0],
        "scale_pos_weight": [1.0, 1.5, 2.0, 3.0],
    }
    MODEL_CTORS["lightgbm"] = lambda p: lgb.LGBMClassifier(objective="binary", random_state=SEED, verbosity=-1, **p)
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier
    PARAM_SPACES["catboost"] = {
        "iterations": [300, 600, 1000],
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.02, 0.04, 0.06, 0.1],
        "l2_leaf_reg": [1, 3, 7, 12],
        "auto_class_weights": ["Balanced"],
    }
    MODEL_CTORS["catboost"] = lambda p: CatBoostClassifier(random_seed=SEED, verbose=False, **p)
except ImportError:
    pass


def params_hash(family, params):
    key = family + json.dumps(params, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def load_tried_hashes(log_path):
    if not os.path.exists(log_path):
        return set()
    df = pd.read_csv(log_path)
    return set(df["params_hash"].astype(str))


def append_log_row(log_path, row):
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def run_search(family, X, y, folds, cols, fold_pop_feats, pop_cols, n_iter=20, log_path=LOG_PATH, seed=SEED):
    if family not in PARAM_SPACES:
        print(f"Семейство '{family}' недоступно (библиотека не установлена?). Доступны: {list(PARAM_SPACES)}")
        return None

    tried = load_tried_hashes(log_path)
    print(f"[{family}] уже проверено ранее: {len(tried)} комбинаций (из {log_path})")

    sampler = list(ParameterSampler(PARAM_SPACES[family], n_iter=n_iter * 3, random_state=seed))
    best_score, best_params, best_oof = -1.0, None, None

    # подхватываем лучший результат из уже сохранённого лога, чтобы не терять прогресс
    if os.path.exists(log_path):
        df_prev = pd.read_csv(log_path)
        df_prev_fam = df_prev[df_prev["family"] == family]
        if len(df_prev_fam):
            row = df_prev_fam.loc[df_prev_fam["mean_score"].idxmax()]
            best_score, best_params = float(row["mean_score"]), json.loads(row["params_json"])
            print(f"[{family}] лучший результат из прошлых запусков: {best_score:.4f}  {best_params}")

    done = 0
    t0 = time.time()
    for params in sampler:
        if done >= n_iter:
            break
        h = params_hash(family, params)
        if h in tried:
            continue
        model_factory = lambda p=params: MODEL_CTORS[family](p)
        sc, oof = cv_predict(model_factory, X, y, folds, cols, fold_pop_feats, pop_cols)
        mean_s, std_s = float(np.mean(sc)), float(np.std(sc))

        append_log_row(log_path, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "family": family,
            "params_hash": h,
            "params_json": json.dumps(params),
            "mean_score": round(mean_s, 5),
            "std_score": round(std_s, 5),
            "fold_scores": json.dumps([round(x, 4) for x in sc]),
        })
        done += 1
        marker = ""
        if mean_s > best_score:
            best_score, best_params, best_oof = mean_s, params, oof
            marker = "  <- новый лучший"
        print(f"[{family}] {done}/{n_iter}  {mean_s:.4f} +- {std_s:.4f}{marker}  [{time.time()-t0:.0f}s]")

    if best_oof is not None:
        with open(f"hpo_best_{family}.pkl", "wb") as f:
            pickle.dump({"family": family, "score": best_score, "params": best_params, "oof": best_oof}, f)

    print(f"[{family}] лучшее за весь лог: {best_score:.4f}  {best_params}")
    return {"score": best_score, "params": best_params, "oof": best_oof}



from core import cv_predict  


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(PARAM_SPACES) or ["histgb"])
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--log", default=LOG_PATH)
    args = ap.parse_args()

    with open("features_cache.pkl", "rb") as f:
        d = pickle.load(f)
    Xtr_base = d["Xtr_base"]
    cols = [c for c in Xtr_base.columns if c != "cookie_id"]
    pop_cols = ["avg_item_popularity", "max_item_popularity"]

    run_search(args.family, Xtr_base, d["y"], d["folds"], cols, d["fold_pop_feats"], pop_cols,
               n_iter=args.n_iter, log_path=args.log)
