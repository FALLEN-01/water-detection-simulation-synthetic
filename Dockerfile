FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install pandas==2.0.3 scikit-learn==1.3.0 matplotlib==3.7.2

COPY *.py .

CMD ["python", "main.py"]
