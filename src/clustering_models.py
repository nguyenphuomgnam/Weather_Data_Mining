import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class WeatherClustering:
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.scaler = StandardScaler()
        self.kmeans = None
        self.features = self.config['clustering']['features']

    def preprocess_for_clustering(self, df):
        print("⏳ Đang chuẩn hoá dữ liệu (StandardScaler)...")
        # Lấy các cột số thực
        X = df[self.features].copy()
        
        # Điền khuyết bằng trung bình nếu có
        X = X.fillna(X.mean())
        
        # Chuẩn hoá
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=self.features)

    def run_kmeans(self, X_scaled):
        k = self.config['clustering']['n_clusters']
        rs = self.config['clustering']['random_state']
        
        print(f"⏳ Đang huấn luyện K-Means với K={k}...")
        self.kmeans = KMeans(n_clusters=k, random_state=rs, n_init=10)
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Đánh giá Silhouette Score (nếu dữ liệu quá lớn, ta lấy mẫu để tính cho nhanh)
        sample_size = min(10000, len(X_scaled))
        score = silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=rs)
        print(f"✅ Phân cụm xong! Silhouette Score: {score:.4f}")
        
        return labels

    def get_cluster_profile(self, df, labels):
        print("⏳ Đang tạo hồ sơ cụm (Profiling)...")
        df_profile = df.copy()
        df_profile['Cluster'] = labels
        
        # Tính trung bình các đặc trưng theo cụm
        profile = df_profile.groupby('Cluster')[self.features].mean()
        profile['Số ngày'] = df_profile.groupby('Cluster').size()
        return profile.round(2)

    def plot_clusters_pca(self, X_scaled, labels):
        print("⏳ Đang giảm chiều bằng PCA để vẽ biểu đồ...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', alpha=0.6)
        plt.title('Biểu đồ phân cụm thời tiết (PCA 2D)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()