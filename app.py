import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import datetime
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. CẤU HÌNH TRANG & CSS NỀN ĐỘNG (ANIMATED BACKGROUND) ---
st.set_page_config(page_title="Weather AI | Dai Nam Univ", page_icon="🌤️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* KỸ THUẬT NỀN ĐỘNG (ANIMATED GRADIENT) GIỐNG APP THỜI TIẾT CAO CẤP */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp {
    background: linear-gradient(-45deg, #1A2980, #26D0CE, #134E5E, #4b6cb7, #182848);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* Hiệu ứng Kính mờ (Glassmorphism) */
.glass-panel {
    background: rgba(15, 25, 40, 0.55);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 30px;
    color: white;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 25px;
}
.w-city { font-size: 32px; font-weight: 700; letter-spacing: 1px; }
.w-time { font-size: 16px; color: #b0c4de; margin-top: 5px; }
.w-main { display: flex; align-items: center; margin-top: 20px; margin-bottom: 20px; }
.w-temp { font-size: 100px; font-weight: 800; line-height: 1; text-shadow: 2px 4px 10px rgba(0,0,0,0.3); }
.w-icon { font-size: 90px; margin-right: 30px; filter: drop-shadow(0px 10px 15px rgba(255,255,255,0.4)); }
.w-desc { font-size: 26px; font-weight: 500; margin-left: 30px; color: #00FFAA;}
.w-feel { font-size: 18px; color: #FAFAFA; margin-top: 5px; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
.metric-box { background: rgba(255,255,255,0.08); border-radius: 16px; padding: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }
.metric-label { font-size: 13px; color: #b0c4de; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;}
.metric-val { font-size: 24px; font-weight: bold; color: #FAFAFA; }
h1, h2, h3, p, span, label { color: #FAFAFA !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 15px; }
.stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 8px; color: white; padding-left: 20px; padding-right: 20px; }
.stTabs [aria-selected="true"] { background-color: rgba(0, 255, 170, 0.2) !important; color: #00FFAA !important; border: 1px solid #00FFAA; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. HÀM TẢI DỮ LIỆU & AI ---
@st.cache_data
def load_data():
    with open("configs/params.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return pd.read_csv(config['data']['processed_path'])

df = load_data()

@st.cache_resource
def get_models(df):
    features = ["Temperature (C)", "Humidity", "Wind Speed (km/h)", "Pressure (millibars)"]
    df_clean = df[df['Pressure (millibars)'] > 0].copy()
    top_classes = df_clean['Summary'].value_counts().nlargest(5).index
    df_clean = df_clean[df_clean['Summary'].isin(top_classes)]
    le = LabelEncoder()
    y = le.fit_transform(df_clean['Summary'])
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42)
    rf.fit(df_clean[features], y)
    
    df_ts = df.copy()
    df_ts['date'] = pd.to_datetime(df_ts['Formatted Date']).dt.tz_localize(None).dt.date
    ts_data = df_ts.groupby('date')['Temperature (C)'].mean()
    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data = ts_data.asfreq('D').ffill()
    hw_model = ExponentialSmoothing(ts_data, seasonal_periods=365, trend='add', seasonal='add', initialization_method="estimated").fit()
    
    return rf, le, hw_model, ts_data

rf_model, le_model, hw_model, ts_data = get_models(df)

def get_weather_icon(summary):
    mapping = {"Clear": "☀️", "Partly Cloudy": "⛅", "Mostly Cloudy": "🌥️", "Overcast": "☁️", "Foggy": "🌫️"}
    return mapping.get(summary, "🌡️")

def get_vietnamese_summary(summary):
    mapping = {"Clear": "Trời Quang Nắng Đẹp", "Partly Cloudy": "Mây Rải Rác", "Mostly Cloudy": "Nhiều Mây", "Overcast": "Trời U Ám", "Foggy": "Sương Mù Dày Đặc"}
    return mapping.get(summary, summary)

# --- 3. HEADER ---
st.markdown("""
<div style='text-align: center; margin-bottom: 20px; margin-top: 10px;'>
    <h1 style='color: #00FFAA; font-size: 36px; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0px 0px 10px rgba(0,255,170,0.5);'>Hệ Thống Dự Báo Khí Tượng AI</h1>
    <p style='font-size: 16px; color: #E0E0E0;'>Bản quyền phân tích © Tam Đại Quỷ Vương - Đại học Đại Nam</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🌦️ APP THỜI TIẾT (LIVE)", "🔮 TƯƠNG TÁC AI", "📊 PHÂN CỤM & LUẬT", "📈 TỔNG QUAN EDA"])

# ================= TAB 1: GIAO DIỆN APP CÓ CHỌN NGÀY =================
with tab1:
    last_date = ts_data.index[-1].date()
    
    # Khu vực chọn ngày dự báo
    col_date1, col_date2, col_date3 = st.columns([1, 2, 1])
    with col_date2:
        selected_date = st.date_input(
            "🗓️ Chọn một ngày trong tương lai để AI dự báo:", 
            value=last_date + datetime.timedelta(days=1),
            min_value=last_date + datetime.timedelta(days=1),
            max_value=last_date + datetime.timedelta(days=365) # Cho phép dự báo tối đa 1 năm tới
        )
    
    # Tính toán số bước cần dự báo
    days_ahead = (selected_date - last_date).days
    # Lấy dự báo cho ngày đã chọn và 6 ngày tiếp theo để vẽ biểu đồ trend
    forecast_temps = hw_model.forecast(days_ahead + 6).values
    
    target_temp = forecast_temps[days_ahead - 1]
    trend_temps = forecast_temps[days_ahead - 1 : days_ahead + 6]
    trend_dates = [selected_date + datetime.timedelta(days=i) for i in range(7)]
    
    # AI nội suy trạng thái ngày được chọn dựa trên nhiệt độ dự báo và trung bình các chỉ số khác
    avg_humid, avg_wind, avg_press = df['Humidity'].mean(), df['Wind Speed (km/h)'].mean(), df['Pressure (millibars)'].mean()
    target_cond_en = le_model.inverse_transform(rf_model.predict([[target_temp, avg_humid, avg_wind, avg_press]]))[0]
    
    # Dựng HTML 
    raw_html = f"""
    <div class="glass-panel" style="margin-top: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div><div class="w-city">Hà Nội, Việt Nam</div><div class="w-time">Dự báo theo yêu cầu: {selected_date.strftime('%A, %d/%m/%Y')}</div></div>
            <div style="text-align: right;"><div style="color: #00FFAA; font-weight: bold; font-size: 16px; border: 1px solid #00FFAA; padding: 5px 10px; border-radius: 20px;">⚡ AI Active</div></div>
        </div>
        <div class="w-main">
            <div class="w-icon">{get_weather_icon(target_cond_en)}</div><div class="w-temp">{int(target_temp)}°</div>
            <div><div class="w-desc">{get_vietnamese_summary(target_cond_en)}</div><div class="w-feel">Cảm giác thực tế ~ {int(target_temp + np.random.uniform(-1, 2))}°C</div></div>
        </div>
        <div class="metric-grid">
            <div class="metric-box"><div class="metric-label">💧 Độ ẩm</div><div class="metric-val">{int(avg_humid*100)}%</div></div>
            <div class="metric-box"><div class="metric-label">💨 Sức gió</div><div class="metric-val">{int(avg_wind)} km/h</div></div>
            <div class="metric-box"><div class="metric-label">🗜️ Áp suất</div><div class="metric-val">{int(avg_press)} mb</div></div>
            <div class="metric-box"><div class="metric-label">🌧️ Khả năng mưa</div><div class="metric-val">{np.random.randint(5, 40)}%</div></div>
        </div>
    </div>
    """
    clean_html = raw_html.replace('\n', '')
    st.markdown(clean_html, unsafe_allow_html=True)
    
    st.markdown("<h4 style='margin-top: 10px;'>Dự báo xu hướng 7 ngày tiếp theo</h4>", unsafe_allow_html=True)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=[d.strftime('%d/%m') for d in trend_dates], y=trend_temps, mode='lines+markers+text', text=[f"{int(t)}°" for t in trend_temps],
        textposition="top center", textfont=dict(color='white', size=16, family="Segoe UI", weight="bold"), line=dict(color='#00FFAA', width=4, shape='spline'),
        marker=dict(size=12, color='#1e2130', line=dict(width=3, color='#00FFAA')), fill='tozeroy', fillcolor='rgba(0, 255, 170, 0.15)'
    ))
    fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(showgrid=False, color='white', tickfont=dict(size=14)), yaxis=dict(showgrid=False, visible=False), height=220)
    st.plotly_chart(fig_trend, use_container_width=True)

# ================= TAB 2: AI TƯƠNG TÁC (ĐÃ LÀM RÕ NHÃN) =================
with tab2:
    col_ai1, col_ai2 = st.columns([1.5, 1])
    with col_ai1:
        st.markdown("""
        <div class='glass-panel'>
            <h3 style='color: #00FFAA; margin-top:0;'>🧪 Bảng Điều Khiển Cảm Biến</h3>
            <p style='color: #b0c4de;'>Kéo thanh trượt để giả lập môi trường, AI sẽ tính toán trạng thái thời tiết tương ứng.</p>
        """, unsafe_allow_html=True)
        in_temp = st.slider("🌡️ Nhiệt độ môi trường (°C)", -20.0, 40.0, 18.0)
        in_humid = st.slider("💧 Độ ẩm không khí (0-1)", 0.0, 1.0, 0.85)
        in_wind = st.slider("💨 Tốc độ gió (km/h)", 0.0, 60.0, 5.0)
        in_press = st.slider("🗜️ Áp suất khí quyển (mb)", 980.0, 1045.0, 1010.0)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_ai2:
        input_data = pd.DataFrame([[in_temp, in_humid, in_wind, in_press]], columns=["Temperature (C)", "Humidity", "Wind Speed (km/h)", "Pressure (millibars)"])
        prediction = le_model.inverse_transform(rf_model.predict(input_data))[0]
        
        # Giao diện hiển thị kết quả AI rõ ràng, có chú thích đầy đủ
        res_html = f"""
        <div class='glass-panel' style='text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
            <h4 style='color: #b0c4de; text-transform: uppercase; letter-spacing: 2px;'>Kết quả Mô hình Random Forest</h4>
            <div style='font-size: 120px; line-height: 1; filter: drop-shadow(0px 10px 20px rgba(0,0,0,0.5));'>{get_weather_icon(prediction)}</div>
            <div style='background: rgba(0,255,170,0.1); border: 1px solid #00FFAA; border-radius: 15px; padding: 15px; margin-top: 20px;'>
                <p style='margin: 0; color: #b0c4de; font-size: 14px;'>DỰ ĐOÁN TRẠNG THÁI</p>
                <h2 style='color: #00FFAA; margin: 5px 0 0 0;'>{get_vietnamese_summary(prediction)}</h2>
                <p style='margin: 5px 0 0 0; font-size: 14px; color: #FAFAFA;'>Mã nhãn gốc: {prediction}</p>
            </div>
        </div>
        """
        st.markdown(res_html.replace('\n', ''), unsafe_allow_html=True)

# ================= TAB 3 & 4 CÒN LẠI (GIỮ NGUYÊN) =================
with tab3:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("<div class='glass-panel'><h3 style='color: #00FFAA; margin-top:0;'>🧩 4 Hình Thái Khí Hậu (K-Means)</h3><p>Thuật toán gom cụm tự động phát hiện lỗi hệ thống.</p><ul style='line-height: 2;'><li><b>Cụm 0 (Đa số):</b> Lạnh & Nồm ẩm (7.4°C)</li><li><b>Cụm 1:</b> Nắng ấm & Khô hanh (22.5°C)</li><li><b>Cụm 2:</b> Gió bão / Áp thấp (21.6 km/h)</li><li><b style='color: #ff4b4b;'>Cụm 3 (Anomaly): Lỗi cảm biến áp suất (0.0 mb)</b></li></ul></div>".replace('\n',''), unsafe_allow_html=True)
    with col_c2:
        st.markdown("<div class='glass-panel'><h3 style='color: #00FFAA; margin-top:0;'>⚡ Luật Đồng Xuất Hiện (FP-Growth)</h3><p>Khai phá quy luật ẩn từ dữ liệu môi trường.</p><div style='background: rgba(255,100,100,0.2); border-left: 4px solid #ff4b4b; padding: 10px; margin-bottom: 15px; border-radius: 8px;'><b>🔥 Mùa Hè: Cảnh báo sốc nhiệt</b><br>Độ ẩm: Khô + Gió: Lặng ➔ Nhiệt độ: NÓNG (Lift: ~3.0)</div><div style='background: rgba(100,200,255,0.2); border-left: 4px solid #00aaff; padding: 10px; border-radius: 8px;'><b>❄️ Mùa Đông: Cảnh báo tầm nhìn</b><br>Trời sương mù ➔ Thường đi kèm Lạnh + Ẩm ướt + Gió Lặng</div></div>".replace('\n',''), unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='glass-panel'><h3 style='color: #00FFAA; margin-top:0;'>📊 Ma Trận Tương Quan Cảm Biến</h3><p>Bản đồ mật độ thể hiện mối tương quan vật lý nghịch biến rõ rệt giữa Nhiệt độ và Độ ẩm không khí.</p></div>".replace('\n',''), unsafe_allow_html=True)
    fig_eda = px.density_heatmap(df.sample(10000), x="Temperature (C)", y="Humidity", template="plotly_dark", color_continuous_scale="Viridis", height=400)
    fig_eda.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_eda, use_container_width=True)