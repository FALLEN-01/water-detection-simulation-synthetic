FROM tensorflow/tensorflow:2.13.0

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install pandas==2.0.3 scikit-learn==1.3.0 matplotlib==3.7.2

COPY *.py .
COPY priors_india/ ./priors_india/

CMD ["python", "-u", "main.py"]
