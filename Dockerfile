FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 -r requirements.txt

COPY *.py .

CMD ["python", "main.py"]
