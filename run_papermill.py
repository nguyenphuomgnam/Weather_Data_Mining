import papermill as pm
import yaml
import os

def run_pipeline():
    print("🚀 BẮT ĐẦU KHỞI CHẠY TỰ ĐỘNG TOÀN BỘ PIPELINE KHAI PHÁ DỮ LIỆU...")
    print("-" * 60)

    # 1. Đọc các siêu tham số từ file config
    try:
        with open("configs/params.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print("✅ Đã nạp thành công cấu hình từ configs/params.yaml")
    except Exception as e:
        print(f"❌ Lỗi đọc file config: {e}")
        return

    # 2. Tạo thư mục chứa file kết quả (tránh ghi đè file code gốc)
    os.makedirs("outputs", exist_ok=True)

    # 3. Danh sách các Notebooks cần chạy tuần tự (Sửa tên cho khớp với file trong thư mục notebooks/ của bạn)
    notebooks_to_run = [
        "notebooks/01_eda_and_cleaning.ipynb",
        "notebooks/02_association_rules.ipynb",
        "notebooks/03_clustering.ipynb",
        "notebooks/04_classification.ipynb",
        "notebooks/05_time_series.ipynb"
    ]

    # 4. Vòng lặp thực thi tự động bằng Papermill
    for nb_path in notebooks_to_run:
        if os.path.exists(nb_path):
            # Tạo tên file output tương ứng
            out_nb = nb_path.replace("notebooks/", "outputs/executed_")
            print(f"\n⏳ Đang thực thi khối lệnh: {nb_path}...")
            
            try:
                # Lệnh execute này sẽ mở notebook, truyền parameters vào và chạy toàn bộ cell
                pm.execute_notebook(
                    input_path=nb_path,
                    output_path=out_nb,
                    parameters=config  # Bơm tham số từ yaml vào thẳng notebook
                )
                print(f"✅ Hoàn tất! Kết quả đã được lưu tại: {out_nb}")
            except Exception as e:
                print(f"❌ Pipeline thất bại tại {nb_path}. Chi tiết lỗi: {e}")
                break # Dừng pipeline ngay nếu có lỗi
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file {nb_path}, đang bỏ qua...")

    print("-" * 60)
    print("🎉 TOÀN BỘ PIPELINE ĐÃ CHẠY XONG! HỆ THỐNG SẴN SÀNG.")

if __name__ == "__main__":
    run_pipeline()