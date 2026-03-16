import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class WeatherTimeSeries:
    def train_and_forecast_hw(self, ts_data):
        split_ratio = self.config['timeseries']['train_split_ratio']
        split_idx = int(len(ts_data) * split_ratio)
        train, test = ts_data.iloc[:split_idx], ts_data.iloc[split_idx:]
        
        print(f"⏳ Đang huấn luyện Holt-Winters (Bắt chu kỳ 365 ngày) trên {len(train)} ngày...")
        # Cấu hình chu kỳ 365 ngày
        model = ExponentialSmoothing(
            train, 
            seasonal_periods=365, 
            trend='add', 
            seasonal='add', 
            initialization_method="estimated"
        )
        fitted_model = model.fit()
        
        print("⏳ Đang dự báo trên tập Test...")
        forecast = fitted_model.forecast(steps=len(test))
        forecast.index = test.index
        
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"✅ Đánh giá: MAE = {mae:.2f}°C | RMSE = {rmse:.2f}°C")
        
        plt.figure(figsize=(14, 6))
        # Trực quan hóa 2 năm cuối của tập Train cho dễ nhìn
        plt.plot(train.index[-730:], train.values[-730:], label='Train (Past 2 Years)')
        plt.plot(test.index, test.values, label='Test (Actual)')
        plt.plot(forecast.index, forecast.values, color='red', label='Forecast (Holt-Winters)')
        plt.title('Dự báo Nhiệt độ bằng Holt-Winters (Tính đến chu kỳ năm)')
        plt.legend()
        plt.show()
        
        return test, forecast, fitted_model
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def prepare_time_series(self, df):
        print("⏳ Đang chuẩn bị chuỗi thời gian...")
        dt_col = self.config['columns']['datetime_col']
        target = self.config['timeseries']['target_col']
        rule = self.config['timeseries']['resample_rule']
        
        # Đảm bảo index là datetime
        df_ts = df.copy()
        df_ts[dt_col] = pd.to_datetime(df_ts[dt_col])
        df_ts = df_ts.set_index(dt_col)
        
        # Resample: Tính nhiệt độ trung bình theo Ngày (Daily)
        ts_data = df_ts[target].resample(rule).mean().dropna()
        print(f"✅ Chuỗi thời gian đã gộp theo ngày. Tổng số ngày: {len(ts_data)}")
        return ts_data

    def plot_acf_pacf(self, ts_data):
        print("⏳ Đang vẽ biểu đồ ACF và PACF...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        plot_acf(ts_data, lags=40, ax=axes[0], title="Autocorrelation (ACF) - Lags 40 days")
        plot_pacf(ts_data, lags=40, ax=axes[1], title="Partial Autocorrelation (PACF) - Lags 40 days")
        plt.tight_layout()
        plt.show()

    def train_and_forecast_arima(self, ts_data):
        # 1. Chronological Split (CHỐNG LEAKAGE)
        split_ratio = self.config['timeseries']['train_split_ratio']
        split_idx = int(len(ts_data) * split_ratio)
        
        train, test = ts_data.iloc[:split_idx], ts_data.iloc[split_idx:]
        print(f"⏳ Đang huấn luyện ARIMA trên tập Train ({len(train)} ngày)...")
        
        # 2. Huấn luyện mô hình ARIMA
        order = tuple(self.config['timeseries']['arima_order'])
        model = ARIMA(train, order=order)
        fitted_model = model.fit()
        
        # 3. Dự báo (Forecasting)
        print("⏳ Đang dự báo trên tập Test...")
        forecast = fitted_model.forecast(steps=len(test))
        forecast.index = test.index
        
        # 4. Đánh giá MAE, RMSE
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"✅ Đánh giá: MAE = {mae:.2f}°C | RMSE = {rmse:.2f}°C")
        
        # 5. Vẽ biểu đồ dự báo (Chỉ vẽ 1 năm cuối cho dễ nhìn)
        plt.figure(figsize=(12, 6))
        plt.plot(train.index[-365:], train.values[-365:], label='Train (Past Year)')
        plt.plot(test.index, test.values, label='Test (Actual)')
        plt.plot(forecast.index, forecast.values, color='red', label='Forecast (ARIMA)')
        plt.title('Dự báo Nhiệt độ (ARIMA)')
        plt.legend()
        plt.show()
        
        return test, forecast, fitted_model

    def analyze_residuals(self, fitted_model):
        print("⏳ Đang chẩn đoán Phần dư (Residuals)...")
        residuals = fitted_model.resid
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Biểu đồ đường của phần dư
        residuals.plot(title="Biểu đồ Phần dư (Residuals)", ax=ax[0])
        ax[0].axhline(0, color='r', linestyle='--', alpha=0.5)
        
        # Biểu đồ phân phối mật độ
        residuals.plot(kind='kde', title="Phân phối mật độ của Phần dư", ax=ax[1])
        plt.tight_layout()
        plt.show()
    