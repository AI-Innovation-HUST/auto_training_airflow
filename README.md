* auto_training_airflow

1. tools auto_training and crawl data from okx platform
2. Các bước khởi chạy auto tool trên local machine
   * install package bằng lệnh: sh install_all_packages.sh
   * Khởi chạy airflow webserver bằng lệnh : airflow webserver --port 8080
   * copy các file sau vào thư mục airflow(khởi tạo bảo airflow):
     * mkdir ~/airflow/dags
     * cp dataloader_v2.py ~/airflow/dags
     * cp full_pipeline.py ~/airflow/dags
     * cp model.py ~/airflow/dags
     * cp scaler.py ~/airflow/dags
     * cp transformer.py ~/airflow/dags
   * Reload lại Airflow webserver và turn on DAG trên web
