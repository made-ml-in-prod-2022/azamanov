# azamanov
ДЗ1 Машинное обучение в продакшене

Установка:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Запуск:
PYTHONPATH=. python ml_project/train_pipeline.py hydra.run.dir=. [config params]
где config params - параметры конфигурации (например, train_params.model_type=RandomForestClassifier)

Запуск предикта:
PYTHONPATH=. python ml_project/predict_pipeline.py hydra.run.dir=. [config params]
где config params - параметры конфигурации (например, train_params.model_type=RandomForestClassifier)