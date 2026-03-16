import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
class WeatherClassification:
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
            
        self.le = LabelEncoder()
        # Khởi tạo mô hình Random Forest
        self.model = RandomForestClassifier(
            n_estimators=self.config['classification']['rf_params']['n_estimators'],
            max_depth=self.config['classification']['rf_params']['max_depth'],
            random_state=self.config['classification']['random_state'],
            class_weight='balanced', # Giúp tăng F1-macro cho các lớp thiểu số
            n_jobs=-1
        )

    def prepare_data(self, df):
        print("⏳ Đang chuẩn bị dữ liệu và Feature Engineering...")
        df_model = df.copy()
        
        # 1. Bỏ các dòng bị lỗi áp suất = 0 (Insight từ phần Clustering)
        df_model = df_model[df_model['Pressure (millibars)'] > 0]
        
        # 2. Trích xuất đặc trưng thời gian
        dt_col = self.config['columns']['datetime_col']
        df_model['Month'] = pd.to_datetime(df_model[dt_col]).dt.month
        df_model['Hour'] = pd.to_datetime(df_model[dt_col]).dt.hour
        
        # 3. Mã hoá cột Precip Type (rain/snow -> 0/1)
        df_model['Precip Type'] = df_model['Precip Type'].astype('category').cat.codes
        
        # 4. Giữ lại Top K nhãn thời tiết phổ biến nhất để tránh nhiễu
        target_col = self.config['classification']['target_col']
        top_classes = df_model[target_col].value_counts().nlargest(self.config['classification']['top_k_classes']).index
        df_model = df_model[df_model[target_col].isin(top_classes)]
        
        print(f"✅ Đang phân lớp cho {len(top_classes)} nhãn: {list(top_classes)}")
        
        # Khai báo Features và Target
        features = self.config['clustering']['features'] + ['Month', 'Hour', 'Precip Type']
        X = df_model[features]
        y = self.le.fit_transform(df_model[target_col])
        
        # Chia tập Train/Test
        ts = self.config['classification']['test_size']
        rs = self.config['classification']['random_state']
        return train_test_split(X, y, test_size=ts, random_state=rs), self.le.classes_

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, target_names):
        print("⏳ Đang huấn luyện Random Forest...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        # Tính F1-Macro
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"✅ Huấn luyện xong! F1-Macro Score: {macro_f1:.4f}")
        
        print("\n" + "="*50)
        print("BÁO CÁO PHÂN LỚP (CLASSIFICATION REPORT)")
        print("="*50)
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return y_pred
        
    def plot_confusion_matrix(self, y_test, y_pred, target_names):
        print("⏳ Đang vẽ Confusion Matrix để phân tích lỗi...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
        plt.ylabel('Thực Tế (True)')
        plt.xlabel('Dự Đoán (Predicted)')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title('Mức độ quan trọng của các Đặc trưng (Feature Importances)')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Độ quan trọng tương đối')
        plt.tight_layout()
        plt.show()
    def compare_models(self, X_train, X_test, y_train, y_test):
        print("⏳ Đang huấn luyện và so sánh các mô hình...")
        
        # Khai báo 3 mô hình từ dễ đến khó
        models = {
            "Decision Tree (Baseline 1)": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
            "Random Forest (Baseline 2)": self.model, # Model cũ của chúng ta
            "XGBoost (Cải tiến)": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        }
        
        results = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            f1 = f1_score(y_test, preds, average='macro')
            results[name] = f1
            print(f"✅ {name} - F1-Macro: {f1:.4f}")
            
        # Vẽ biểu đồ so sánh
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(results.values()), y=list(results.keys()), palette='magma')
        plt.title('So sánh hiệu suất F1-Macro giữa các Mô hình', fontsize=14, fontweight='bold')
        plt.xlabel('F1-Macro Score')
        plt.xlim(0, 1)
        for i, v in enumerate(results.values()):
            plt.text(v + 0.01, i, f"{v:.4f}", color='black', va='center', fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return results