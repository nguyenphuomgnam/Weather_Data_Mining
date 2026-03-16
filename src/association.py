import pandas as pd
import yaml
from mlxtend.frequent_patterns import fpgrowth, association_rules

class WeatherAssociation:
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
            
    def get_season(self, month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'

    def discretize_data(self, df):
        print("⏳ Đang rời rạc hoá dữ liệu...")
        df_cat = pd.DataFrame()
        
        # 1. Trích xuất Mùa
        dt_col = self.config['columns']['datetime_col']
        df_cat['Season'] = pd.to_datetime(df[dt_col]).dt.month.apply(self.get_season)
        
        # 2. Rời rạc hoá Nhiệt độ, Độ ẩm, Sức gió
        bins_cfg = self.config['association']['discretize_bins']
        df_cat['Temp_Cat'] = pd.cut(df['Temperature (C)'], bins=bins_cfg['temp'], labels=bins_cfg['temp_labels'])
        df_cat['Humidity_Cat'] = pd.cut(df['Humidity'], bins=bins_cfg['humidity'], labels=bins_cfg['humidity_labels'])
        df_cat['Wind_Cat'] = pd.cut(df['Wind Speed (km/h)'], bins=bins_cfg['wind'], labels=bins_cfg['wind_labels'])
        
        # 3. Thêm các nhãn đã có sẵn
        df_cat['Summary'] = df['Summary']
        
        # Gộp tiền tố để phân biệt rõ ràng (Ví dụ: Temp_Hot, Humidity_High)
        for col in df_cat.columns:
            df_cat[col] = col + "_" + df_cat[col].astype(str)
            
        return df_cat

    def run_fpgrowth(self, df_cat, season=None):
        if season:
            df_cat = df_cat[df_cat['Season'] == f"Season_{season}"]
            df_cat = df_cat.drop(columns=['Season']) # Không cần nhãn mùa trong luật của chính mùa đó
            
        print(f"⏳ Đang chạy FP-Growth cho {season if season else 'toàn bộ dữ liệu'}...")
        
        # Chuyển đổi dữ liệu sang dạng One-Hot Encoding để mlxtend hiểu
        transactions = []
        for i in range(df_cat.shape[0]):
            transactions.append([str(df_cat.values[i, j]) for j in range(df_cat.shape[1])])
            
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

        # Chạy thuật toán
        min_sup = self.config['association']['min_support']
        min_conf = self.config['association']['min_confidence']
        
        frequent_itemsets = fpgrowth(df_onehot, min_support=min_sup, use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
            
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf, num_itemsets=len(frequent_itemsets))
        
        # Lọc và sắp xếp theo Lift
        rules = rules.sort_values('lift', ascending=False)
        return rules