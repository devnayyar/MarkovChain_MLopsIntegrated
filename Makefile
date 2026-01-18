# Makefile for Financial Risk Markov MLOps Project

.PHONY: help install setup test lint format clean run-api run-dashboard docs

help:
	@echo "Financial Risk Markov MLOps - Available Commands"
	@echo "=================================================="
	@echo "make install          - Install dependencies"
	@echo "make setup            - Full project setup"
	@echo "make lint             - Run code linting"
	@echo "make format           - Format code with black"
	@echo "make test             - Run all tests"
	@echo "make test-unit        - Run unit tests only"
	@echo "make test-cov         - Run tests with coverage"
	@echo "make clean            - Clean up build artifacts"
	@echo "make run-api          - Start FastAPI server"
	@echo "make run-dashboard    - Start Streamlit dashboard"
	@echo "make data-download    - Download FRED data"
	@echo "make eda              - Run EDA notebooks"
	@echo "make train            - Train Markov models"
	@echo "make monitor          - Start monitoring dashboard"
	@echo "make docs             - Generate documentation"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

setup: install
	@echo "Setting up project structure..."
	mkdir -p data/bronze data/silver data/gold
	mkdir -p logs models/checkpoints output/{results,reports}
	@echo "Project setup complete!"

lint:
	flake8 preprocessing modeling serving monitoring utils --max-line-length=100
	mypy preprocessing modeling serving --ignore-missing-imports

format:
	black preprocessing modeling serving monitoring dashboards utils tests

test:
	pytest tests/ -v --cov=. --cov-report=html

test-unit:
	pytest tests/unit -v

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ *.egg-info/

run-api:
	uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run dashboards/streamlit_app.py

run-mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

data-download:
	python data/bronze/raw_fred_download.py

eda-bronze:
	jupyter notebook eda/bronze_analysis/

eda-silver:
	jupyter notebook eda/silver_analysis/

eda-gold:
	jupyter notebook eda/gold_analysis/

train-models:
	python modeling/train_pipeline.py

monitor:
	python monitoring/performance/monitoring_dashboard.py

retrain:
	python retraining/retrain_pipeline.py

deploy-docker:
	docker-compose -f docker/docker-compose.yml up -d

stop-docker:
	docker-compose -f docker/docker-compose.yml down

logs-docker:
	docker-compose -f docker/docker-compose.yml logs -f

version:
	@echo "Financial Risk Markov MLOps v0.1.0"
