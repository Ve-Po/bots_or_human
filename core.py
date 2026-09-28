
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

RECALL_TARGET = 0.70
SEED = 42


def precision_at_recall(y_true, y_score, recall_threshold=RECALL_TARGET):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if y_true.sum() == 0 or mask.sum() == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    rmask = recall >= recall_threshold
    return float(precision[rmask].max()) if rmask.any() else 0.0


def events_in_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["cookie_id", "window_start_ts", "window_end_ts"]
    ev = events.merge(meta[cols], on="cookie_id", how="inner", validate="many_to_one")
    mask = (ev["event_ts"] >= ev["window_start_ts"]) & (ev["event_ts"] < ev["window_end_ts"])
    return ev.loc[mask].copy()


def _entropy(s):
    s = s.dropna()
    if len(s) == 0:
        return 0.0
    p = s.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(p * np.log(p.clip(1e-12))).sum())


def item_popularity_table(ref_events: pd.DataFrame) -> pd.Series:
    """Сколько РАЗНЫХ кук смотрели каждый item_id — считается только по 'известным на
    момент предсказания' событиям (ref_events), чтобы не заглядывать в будущее."""
    return ref_events.dropna(subset=["item_id"]).groupby("item_id")["cookie_id"].nunique()


def build_base_features(ev: pd.DataFrame, meta: pd.DataFrame, top_event_names=None) -> pd.DataFrame:
    """Все признаки, которые НЕ зависят от того, как разбито train/valid: паузы между
    событиями, бёрсты, энтропии, UA-эвристики, вовлечённость. Считается один раз для всех
    кук сразу"""
    base = meta[["cookie_id", "cookie_created_at", "window_start_ts", "window_end_ts"]].copy()
    ev = ev.copy().sort_values(["cookie_id", "event_ts"])
    g = ev.groupby("cookie_id", sort=False)

    ev["gap"] = g["event_ts"].diff().dt.total_seconds()
    ev["gap_round"] = ev["gap"].round(0)
    ev["same_gap"] = (ev["gap_round"] == g["gap_round"].shift()).astype(float).where(ev["gap"].notna())
    ev["same_event"] = (ev["event_name"] == g["event_name"].shift()).astype(float).where(g["event_name"].shift().notna())

    ev["hour"] = ev["event_ts"].dt.hour
    ev["is_night"] = ev["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    ua = ev["user_agent"].astype("string").fillna("").str.lower()
    ev["okhttp"] = ua.str.contains("okhttp", regex=False).astype(int)
    ev["mobile"] = ua.str.contains("mobile|android|iphone|ipad", regex=True).astype(int)
    ev["botword"] = ua.str.contains("bot|spider|crawler|scrapy|python|curl|wget|go-http|java", regex=True).astype(int)
    ev["headless"] = ua.str.contains("headless|selenium|playwright|puppeteer", regex=True).astype(int)

    out = g.agg(
        n_events=("event_ts", "size"),
        n_unique_events=("event_name", "nunique"),
        n_unique_ua=("user_agent", "nunique"),
        n_unique_platforms=("platform", "nunique"),
        n_unique_categories=("item_category", "nunique"),
        n_unique_locations=("item_location", "nunique"),
        gap_mean=("gap", "mean"), gap_std=("gap", "std"),
        gap_gt60_frac=("gap", lambda s: float(s.gt(60).mean()) if s.notna().any() else 0.0),
        same_gap_frac=("same_gap", "mean"), same_event_frac=("same_event", "mean"),
        active_span_sec=("event_ts", lambda s: (s.max() - s.min()).total_seconds()),
        night_frac=("is_night", "mean"),
        okhttp_frac=("okhttp", "mean"), mobile_frac=("mobile", "mean"),
        botword_frac=("botword", "mean"), headless_frac=("headless", "mean"),
        seller_type_pro_frac=("seller_type", lambda s: float((s == "pro").mean())),
        search_count=("search_query", lambda s: int(s.notna().sum())),
        search_query_nunique=("search_query", "nunique"),
        search_page_max=("search_page", "max"),
    )

    out["burstiness"] = out["gap_std"] / (out["gap_mean"].abs() + 1e-6)
    out["event_rate_per_min"] = out["n_events"] / (out["active_span_sec"].clip(lower=1) / 60.0)
    out["hour_entropy"] = g["hour"].apply(_entropy)
    out["category_entropy"] = g["item_category"].apply(_entropy)
    out["location_entropy"] = g["item_location"].apply(_entropy)
    out["event_entropy"] = g["event_name"].apply(_entropy)
    out["queries_per_search"] = (out["search_query_nunique"] / out["search_count"].replace(0, np.nan)).fillna(0)
    out["same_gap_frac"] = out["same_gap_frac"].fillna(0)
    out["same_event_frac"] = out["same_event_frac"].fillna(0)

    # Движение мыши: pointer_x/pointer_y логируются вместе примерно у трети событий. Раньше
    # использовался только pointer_std (std одной координаты X, без Y и без времени) — по
    # литературе (Zi Chu et al., "Blog or Block"; Acien et al., "BeCAPTCHA-Mouse") именно
    # СКОРОСТЬ курсора и "прямолинейность" траектории — одни из сильнейших человек/бот
    # различителей (боты physически не ограничены скоростью движения руки).
    mp = ev.dropna(subset=["pointer_x", "pointer_y"]).sort_values(["cookie_id", "event_ts"]).copy()
    mg = mp.groupby("cookie_id", sort=False)
    dt = mg["event_ts"].diff().dt.total_seconds()
    disp = np.sqrt(mg["pointer_x"].diff() ** 2 + mg["pointer_y"].diff() ** 2)
    # dt.clip(lower=0.05) — защита от деления на ~0, когда два замера пришли с одинаковой
    # секундной меткой времени
    mp["mouse_speed"] = (disp / dt.clip(lower=0.05)).where(dt > 0)
    mp["mouse_disp"] = disp

    mouse = mg.agg(
        pointer_n=("pointer_x", "size"),
        pointer_x_std=("pointer_x", "std"), pointer_y_std=("pointer_y", "std"),
        x_first=("pointer_x", "first"), y_first=("pointer_y", "first"),
        x_last=("pointer_x", "last"), y_last=("pointer_y", "last"),
    )
    speed_agg = mp.groupby("cookie_id")["mouse_speed"].agg(mouse_speed_mean="mean", mouse_speed_max="max")
    disp_sum = mp.groupby("cookie_id")["mouse_disp"].sum().rename("mouse_disp_sum")
    mouse = mouse.join(speed_agg).join(disp_sum)

    straight_dist = np.sqrt((mouse["x_last"] - mouse["x_first"]) ** 2 + (mouse["y_last"] - mouse["y_first"]) ** 2)
    # "прямолинейность" пути: расстояние по прямой от первого до последнего замера / реальная
    # длина пройденного пути — ближе к 1, если курсор шёл почти по прямой (характерно для
    # автоматизации), ближе к 0 — у более "петляющей" траектории человека
    mouse["mouse_straightness"] = (straight_dist / mouse["mouse_disp_sum"].replace(0, np.nan)).clip(upper=1.0)
    mouse = mouse.drop(columns=["x_first", "y_first", "x_last", "y_last"])

    for c in mouse.columns:
        out[c] = mouse[c]

    engagement_names = {"photo_swipe", "favorite_add", "login", "contact_message_sent"}
    event_ct = pd.crosstab(ev["cookie_id"], ev["event_name"], normalize="index")
    out["engagement_frac"] = sum(event_ct[n] if n in event_ct.columns else 0.0 for n in engagement_names)
    out["login_frac"] = event_ct["login"] if "login" in event_ct.columns else 0.0

    if top_event_names is not None:
        event_binary = pd.crosstab(ev["cookie_id"], ev["event_name"])
        for name in top_event_names:
            safe = "event_" + re.sub(r"[^0-9A-Za-z_]+", "_", str(name))[:50]
            out[safe] = event_binary[name] if name in event_binary.columns else 0.0

    out = out.reset_index()
    age = (base["window_start_ts"] - base["cookie_created_at"]).dt.total_seconds().clip(lower=0)
    out = base[["cookie_id"]].assign(cookie_age_log=np.log1p(age).to_numpy()).merge(
        out, on="cookie_id", how="left", validate="one_to_one"
    )
    numeric = [c for c in out.columns if c != "cookie_id"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def popularity_features(ev: pd.DataFrame, popularity: pd.Series) -> pd.DataFrame:
    ev = ev.dropna(subset=["item_id"]).copy()
    ev["item_pop"] = ev["item_id"].map(popularity)
    out = ev.groupby("cookie_id")["item_pop"].agg(avg_item_popularity="mean", max_item_popularity="max")
    return out.reset_index()


def event_transition_table(ref_events: pd.DataFrame) -> pd.DataFrame:
    """Матрица переходов P(след. событие | пред. событие) """
    ev = ref_events.sort_values(["cookie_id", "event_ts"])
    prev = ev.groupby("cookie_id")["event_name"].shift().rename("prev_event")
    counts = pd.crosstab(prev, ev["event_name"])
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    long = probs.stack().rename("trans_p").reset_index()
    long.columns = ["prev_event", "event_name", "trans_p"]
    return long


def transition_features(ev: pd.DataFrame, trans_probs: pd.DataFrame) -> pd.DataFrame:
    """Для каждой куки — средний log P(переход) по её собственной последовательности событий
    под  матрицей переходов Низкое значение — куке
    свойственны переходы между событиями, нетипичные для общей популяции (возможный признак
    скриптованного, а не органического поведения)."""
    ev = ev.sort_values(["cookie_id", "event_ts"]).copy()
    ev["prev_event"] = ev.groupby("cookie_id")["event_name"].shift()
    merged = ev.merge(trans_probs, on=["prev_event", "event_name"], how="left")
    merged["log_trans_p"] = np.log(merged["trans_p"].clip(lower=1e-4))
    out = merged.groupby("cookie_id")["log_trans_p"].mean().rename("avg_log_transition_p")
    return out.reset_index()


def build_features(ev: pd.DataFrame, meta: pd.DataFrame, popularity: pd.Series,
                    top_event_names=None) -> pd.DataFrame:
 
    base = build_base_features(ev, meta, top_event_names=top_event_names)
    pop = popularity_features(ev, popularity)
    out = base.merge(pop, on="cookie_id", how="left")
    for c in ["avg_item_popularity", "max_item_popularity"]:
        out[c] = out[c].fillna(0.0)
    return out


def make_time_folds(df: pd.DataFrame, n_splits: int = 5):
    dates = np.sort(df["window_start_ts"].dropna().unique())
    n_splits = min(n_splits, len(dates) - 1)
    chunks = np.array_split(dates, n_splits + 1)
    folds = []
    for i in range(1, n_splits + 1):
        tr_dates = np.concatenate(chunks[:i])
        va_dates = chunks[i]
        tr_mask = df["window_start_ts"].isin(tr_dates).to_numpy()
        va_mask = df["window_start_ts"].isin(va_dates).to_numpy()
        if tr_mask.sum() and va_mask.sum() and np.unique(df.loc[tr_mask, "target"]).size == 2:
            folds.append((tr_mask, va_mask))
    return folds


def cv_predict(model_factory, X, y, folds, cols, fold_pop_feats=None, pop_cols=None):

    oof = np.full(len(y), np.nan)
    scores = []
    X_indexed = X.set_index("cookie_id")
    for i, (tm, vm) in enumerate(folds):
        if fold_pop_feats is not None:
            Xf = X_indexed.join(fold_pop_feats[i], how="left").fillna(0.0).reset_index()
            use_cols = cols + pop_cols
        else:
            Xf = X
            use_cols = cols
        Xt = numeric_frame(Xf.iloc[np.flatnonzero(tm)], use_cols)
        Xv = numeric_frame(Xf.iloc[np.flatnonzero(vm)], use_cols)
        model = model_factory()
        model.fit(Xt, y[tm])
        p = model.predict_proba(Xv)[:, 1]
        oof[vm] = p
        scores.append(precision_at_recall(y[vm], p))
    return scores, oof


def numeric_frame(df, cols):
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def cv_overfit_check(model_factory, X, y, folds, cols, fold_pop_feats=None, pop_cols=None):
    train_scores, valid_scores = [], []
    X_indexed = X.set_index("cookie_id")
    for i, (tm, vm) in enumerate(folds):
        if fold_pop_feats is not None:
            Xf = X_indexed.join(fold_pop_feats[i], how="left").fillna(0.0).reset_index()
            use_cols = cols + pop_cols
        else:
            Xf = X
            use_cols = cols
        Xt = numeric_frame(Xf.iloc[np.flatnonzero(tm)], use_cols)
        Xv = numeric_frame(Xf.iloc[np.flatnonzero(vm)], use_cols)
        model = model_factory()
        model.fit(Xt, y[tm])
        p_tr = model.predict_proba(Xt)[:, 1]
        p_va = model.predict_proba(Xv)[:, 1]
        train_scores.append(precision_at_recall(y[tm], p_tr))
        valid_scores.append(precision_at_recall(y[vm], p_va))
    return train_scores, valid_scores


def robust_best_params(log: pd.DataFrame, family: str, tol_stds: float = 1.0):
    """Выбор лучших гиперпараметров семейства из hpo_log.csv 
    """
    fam = log[log["family"] == family]
    if fam.empty:
        return None
    best_row = fam.loc[fam["mean_score"].idxmax()]
    band = best_row["mean_score"] - tol_stds * best_row["std_score"]
    candidates = fam[fam["mean_score"] >= band]
    row = candidates.loc[candidates["std_score"].idxmin()]
    return {
        "params": json.loads(row["params_json"]),
        "mean_score": float(row["mean_score"]),
        "std_score": float(row["std_score"]),
        "picked_argmax": bool(row["params_hash"] == best_row["params_hash"]),
    }
