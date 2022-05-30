# azamanov
ДЗ2 Машинное обучение в продакшене

Скачивание образа:
docker pull zamanov/ml_in_prod_hw2:latest

Запуск контейнера:
docker run -p 8000:8000 zamanov/ml_in_prod_hw2:latest

Запуск скрипта для запросов к сервису:
python online_inference/request_service.py URL NUM_REQUESTS
где URL - адрес, по которому делать запросы, NUM_REQUESTS - количество запросов
