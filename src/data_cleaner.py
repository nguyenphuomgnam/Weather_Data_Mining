import pandas as pd
import yaml
import os

class WeatherCleaner:
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.df = None

    def load_data(self):
        print("⏳ Đang tải dữ liệu gốc...")
        self.df = pd.read_csv(self.config['data']['raw_path'])
        print(f"✅ Đã tải xong! Kích thước: {self.df.shape}")
        return self.df

    def clean_data(self):
        if self.df is None:
            self.load_data()

        df_clean = self.df.copy()
        print("⏳ Đang làm sạch dữ liệu...")
        
        # 1. Xử lý thời gian (Rất quan trọng cho Time Series)
        dt_col = self.config['columns']['datetime_col']
        # Ép kiểu datetime và đồng bộ hóa múi giờ (UTC)
        df_clean[dt_col] = pd.to_datetime(df_clean[dt_col], utc=True)
        # Sắp xếp theo thứ tự thời gian tăng dần
        df_clean = df_clean.sort_values(by=dt_col).reset_index(drop=True)

        # 2. Xử lý giá trị thiếu (Missing values)
        if 'Precip Type' in df_clean.columns:
            fill_val = self.config['preprocessing']['fill_missing_precip']
            df_clean['Precip Type'] = df_clean['Precip Type'].fillna(fill_val)

        # 3. Bỏ các cột rác
        cols_to_drop = self.config['columns']['drop_cols']
        df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

        self.df = df_clean
        print(f"✅ Làm sạch hoàn tất! Kích thước mới: {self.df.shape}")
        return self.df

    def save_data(self):
        out_path = self.config['data']['processed_path']
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.df.to_csv(out_path, index=False)
        print(f"💾 Đã lưu dữ liệu sạch vào: {out_path}")