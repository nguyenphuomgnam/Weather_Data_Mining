from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA

# Reuse loading/cleaning helpers
from classification_library import (
    Paths,
    _ensure_dirs,
    load_beijing_air_quality,
    clean_air_quality_df,
)


@dataclass(frozen=True)
class StationSeriesConfig:
    station: str
    value_col: str = "PM2.5"
    freq: str = "H"              # hourly
    fill_method: str = "interpolate_time"  # interpolate_time | ffill | none
    clip_negative: bool = True


def make_hourly_station_series(df: pd.DataFrame, cfg: StationSeriesConfig) -> pd.Series:
    """
    Build a *single* univariate hourly series for ARIMA.
    Steps:
      1) filter one station
      2) set datetime index and sort
      3) resample to hourly, keeping gaps explicit
      4) fill missing (for teaching + practical ARIMA fitting)

    Returns: pd.Series with DatetimeIndex (freq may be inferred).
    """
    if "datetime" not in df.columns:
        raise ValueError("df must contain a datetime column.")
    if "station" not in df.columns:
        raise ValueError("df must contain station column (multi-site dataset).")
    if cfg.value_col not in df.columns:
        raise ValueError(f"Missing value_col={cfg.value_col}")

    sdf = df[df["station"] == cfg.station].copy()
    sdf = sdf.sort_values("datetime")
    s = pd.to_numeric(sdf[cfg.value_col], errors="coerce")
    s.index = pd.DatetimeIndex(sdf["datetime"].values)

    # resample hourly (mean if duplicates)
    s = s.resample(cfg.freq).mean()

    if cfg.clip_negative:
        s = s.where((s.isna()) | (s >= 0), 0.0)

    if cfg.fill_method == "interpolate_time":
        # time interpolation requires DatetimeIndex
        s = s.interpolate(method="time", limit_direction="both")
    elif cfg.fill_method == "ffill":
        s = s.ffill().bfill()
    elif cfg.fill_method == "none":
        pass
    else:
        raise ValueError("fill_method must be interpolate_time | ffill | none")

    return s


def describe_time_series(s: pd.Series, seasonal_periods=(24, 24*7)) -> dict:
    """
    Produce diagnostics used for 'đúng bài giảng + ra quyết định chọn mô hình':
      - length, missing ratio
      - basic stats
      - stationarity tests (ADF and KPSS)
      - simple seasonality strength by comparing autocorr at given lags

    NOTE: This returns numbers; the notebook will visualize/interpret.
    """
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.shape[0])
    miss = float(s.isna().mean())

    s_clean = s.dropna()
    stats = {
        "n": n,
        "missing_ratio": miss,
        "min": float(s_clean.min()) if not s_clean.empty else None,
        "max": float(s_clean.max()) if not s_clean.empty else None,
        "mean": float(s_clean.mean()) if not s_clean.empty else None,
        "std": float(s_clean.std(ddof=1)) if not s_clean.empty else None,
    }

    # Stationarity tests
    adf_p = None
    kpss_p = None
    if len(s_clean) >= 50:
        try:
            adf_p = float(adfuller(s_clean, autolag="AIC")[1])
        except Exception:
            adf_p = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_p = float(kpss(s_clean, regression="c", nlags="auto")[1])
        except Exception:
            kpss_p = None

    # autocorr at seasonal lags
    ac = {}
    for lag in seasonal_periods:
        if len(s_clean) > lag + 10:
            ac[f"autocorr_lag_{lag}"] = float(s_clean.autocorr(lag=lag))
        else:
            ac[f"autocorr_lag_{lag}"] = None

    return {**stats, "adf_pvalue": adf_p, "kpss_pvalue": kpss_p, **ac}


def choose_d_by_adf(s: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    """
    Heuristic for d (non-seasonal differencing order):
      - increase d until ADF p-value < alpha or reach max_d
    """
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 50:
        return 0

    d = 0
    while d <= max_d:
        try:
            p = adfuller(x, autolag="AIC")[1]
        except Exception:
            break
        if p < alpha:
            return d
        x = x.diff().dropna()
        d += 1
    return min(d, max_d)


def grid_search_arima_order(
    s: pd.Series,
    p_max: int = 3,
    d_max: int = 2,
    q_max: int = 3,
    d: int | None = None,
    ic: str = "aic",
) -> dict:
    """
    Brute-force search over (p,d,q) with chosen information criterion.
    Keeps it small for a lab notebook.

    Returns dict: best_order, table (list of tried configs).
    """
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 200:
        raise ValueError("Series too short after cleaning; pick another station or fill gaps.")

    if d is None:
        d = choose_d_by_adf(x, max_d=d_max)

    results = []
    best = {"order": None, ic: np.inf}

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            order = (p, d, q)
            try:
                m = ARIMA(x, order=order, enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit()
                score = float(getattr(r, ic))
                results.append({"p": p, "d": d, "q": q, ic: score})
                if np.isfinite(score) and score < best[ic]:
                    best = {"order": order, ic: score}
            except Exception:
                results.append({"p": p, "d": d, "q": q, ic: None})
                continue

    return {"best_order": best["order"], "best_score": best[ic], "table": results}


def train_test_split_series(s: pd.Series, cutoff: str = "2017-01-01") -> tuple[pd.Series, pd.Series]:
    cutoff_ts = pd.Timestamp(cutoff)
    train = s[s.index < cutoff_ts].copy()
    test = s[s.index >= cutoff_ts].copy()
    return train, test


def fit_arima_and_forecast(
    train: pd.Series,
    steps: int,
    order: tuple[int, int, int],
) -> dict:
    """
    Fit ARIMA on train and forecast `steps` ahead.
    Returns: forecast series, conf_int DataFrame, fitted model result.
    """
    x = pd.to_numeric(train, errors="coerce").dropna()
    model = ARIMA(x, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit()

    fc = res.get_forecast(steps=steps)
    yhat = fc.predicted_mean
    ci = fc.conf_int()

    return {"result": res, "forecast": yhat, "conf_int": ci}


def forecast_workflow(
    paths: Paths,
    station: str = "Aotizhongxin",
    value_col: str = "PM2.5",
    cutoff: str = "2017-01-01",
    p_max: int = 3,
    q_max: int = 3,
    d_max: int = 2,
    ic: str = "aic",
    artifacts_prefix: str = "arima",
) -> dict:
    """
    End-to-end helper:
      - load + clean
      - build single-station series
      - diagnostics + grid search ARIMA(p,d,q)
      - fit on train and forecast length(test)
      - save diagnostics + predictions

    Saved:
      data/processed/{prefix}_diagnostics.json
      data/processed/{prefix}_predictions.csv
      data/processed/{prefix}_model.pkl (pickle)
    """
    _ensure_dirs(paths.data_raw, paths.data_processed)

    df = load_beijing_air_quality(use_ucimlrepo=False, raw_zip_path=str(paths.data_raw / "PRSA2017_Data_20130301-20170228.zip"))
    df = clean_air_quality_df(df)

    cfg = StationSeriesConfig(station=station, value_col=value_col)
    s = make_hourly_station_series(df, cfg)

    diag = describe_time_series(s)
    train, test = train_test_split_series(s, cutoff=cutoff)

    gs = grid_search_arima_order(train, p_max=p_max, q_max=q_max, d_max=d_max, d=None, ic=ic)
    order = gs["best_order"] or (1, 1, 1)

    out = fit_arima_and_forecast(train, steps=len(test), order=order)
    yhat = out["forecast"]
    ci = out["conf_int"]

    pred_df = pd.DataFrame({
        "datetime": test.index[:len(yhat)],
        "y_true": test.values[:len(yhat)],
        "y_pred": yhat.values,
        "lower": ci.iloc[:, 0].values,
        "upper": ci.iloc[:, 1].values,
    })

    # metrics (on overlap)
    y_true = pd.to_numeric(test, errors="coerce").values[:len(yhat)]
    y_pred = yhat.values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))) if mask.any() else None
    mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if mask.any() else None

    summary = {
        "station": station,
        "value_col": value_col,
        "cutoff": cutoff,
        "best_order": order,
        "ic": ic,
        "grid_best_score": gs["best_score"],
        "rmse": rmse,
        "mae": mae,
        "diagnostics": diag,
    }

    # save
    with open(paths.data_processed / f"{artifacts_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pred_df.to_csv(paths.data_processed / f"{artifacts_prefix}_predictions.csv", index=False)

    # statsmodels results are picklable
    out["result"].save(paths.data_processed / f"{artifacts_prefix}_model.pkl")

    return {"summary": summary, "pred_df": pred_df, "grid": gs}
# ==============================================================================
# PHẦN MỞ RỘNG CHO CHỦ ĐỀ 3: SARIMAX (ARIMA + Biến ngoại sinh)
# ==============================================================================

import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SARIMAXForecaster:
    """
    Nâng cấp từ ARIMA: Hỗ trợ thêm biến ngoại sinh (Exogenous variables)
    như TEMP, RAIN, WSPM để dự báo PM2.5 chính xác hơn.
    """
    def __init__(self, target_col='PM2.5', exog_cols=['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']):
        self.target_col = target_col
        self.exog_cols = exog_cols
        self.model = None

    def fit(self, train_df, seasonal=False, m=1):
        """
        Huấn luyện SARIMAX với biến ngoại sinh.
        """
        print(f"🚀 [SARIMAX] Bắt đầu huấn luyện với biến ngoại sinh: {self.exog_cols}...")
        
        # 1. Tách biến Mục tiêu (y) và Biến ngoại sinh (X)
        y_train = train_df[self.target_col]
        
        # Kiểm tra xem có đủ cột ngoại sinh không
        missing_cols = [c for c in self.exog_cols if c not in train_df.columns]
        if missing_cols:
            raise ValueError(f"❌ Thiếu cột biến ngoại sinh trong dữ liệu train: {missing_cols}")
            
        X_train = train_df[self.exog_cols].copy()
        
        # 2. Xử lý Missing Value cho X (Bắt buộc phải lấp đầy mới chạy được)
        # Dùng ffill (lấy giá trị trước đó) và bfill (lấy giá trị sau đó)
        X_train = X_train.ffill().bfill()
        
        # 3. Gọi auto_arima có tham số X
        self.model = pm.auto_arima(
            y_train,
            X=X_train,           # <--- ĐIỂM KHÁC BIỆT QUAN TRỌNG NHẤT (Thêm X)
            start_p=1, start_q=1,
            max_p=3, max_q=3,    # Giới hạn p, q nhỏ để chạy nhanh
            d=None,              # Để tự động tìm d (thường là 0 hoặc 1)
            seasonal=seasonal,   # True nếu muốn bật chế độ mùa vụ (Chủ đề 2)
            m=m,                 # Chu kỳ mùa vụ (ví dụ 24 nếu seasonal=True)
            test='adf',
            trace=True,          # Hiện log chạy
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        print("✅ Đã tìm thấy mô hình tối ưu:")
        print(f"   - Order: {self.model.order}")
        print(f"   - Seasonal Order: {self.model.seasonal_order}")

    def predict(self, test_df):
        """
        Dự báo PM2.5 dựa trên thời tiết của tập Test.
        """
        if self.model is None:
            raise ValueError("❌ Model chưa được train! Hãy gọi fit() trước.")
            
        # Lấy X_test tương ứng với độ dài muốn dự báo
        n_periods = len(test_df)
        
        # Xử lý Missing Value cho X_test
        X_test = test_df[self.exog_cols].copy().ffill().bfill()
        
        print(f"🔮 Đang dự báo cho {n_periods} giờ tiếp theo...")
        
        # Dự báo
        forecast, conf_int = self.model.predict(
            n_periods=n_periods, 
            X=X_test,            # <--- Cung cấp thông tin thời tiết để dự báo PM2.5
            return_conf_int=True
        )
        
        return forecast, conf_int, self.model.order