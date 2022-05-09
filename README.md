# azamanov
ДЗ1 Машинное обучение в продакшене

Запуск:
python ml_project/train_pipeline.py $CONFIG_PATH 
, где CONFIG_PATH - путь к конфигу (всего 2 конфига для трейна: train_rf.yaml, train_knn.yaml для методов RandomForest и KNN)

Запуск предикта:
python ml_project/predict_pipeline.py $CONFIG_PATH 
, где CONFIG_PATH - путь к конфигу (пример конфига: configs/pred.yml) 