ingest:
	python -m src.ingest

preprocess:
	python -m src.preprocess

serve:
	python -m src.serve.main

test-api:
	python scripts/test_api.py

docker-build:
	docker build -t multilabel-clf .

docker-up:
	docker-compose up

mlflow-ui:
	mlflow ui --backend-store-uri mlruns
