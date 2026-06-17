FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ecommerce_rag /app/ecommerce_rag
RUN python -m ecommerce_rag.data_loader

EXPOSE 8501
CMD ["streamlit", "run", "ecommerce_rag/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
