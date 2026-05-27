FROM python:3.12-slim

WORKDIR /app

# 装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY api/ ./api/
COPY dbt_project/ ./dbt_project/
COPY ml_pipeline/ ./ml_pipeline/

# 暴露端口
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]