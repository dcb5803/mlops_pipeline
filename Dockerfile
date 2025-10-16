FROM python:3.10-slim
WORKDIR /app
COPY mlops_pipeline.py ./
RUN pip install flask scikit-learn pandas joblib mlflow
EXPOSE 8080
CMD ["python", "mlops_pipeline.py"]
