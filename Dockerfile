FROM tensorflow/tensorflow:2.13.0

# Disable Python output buffering for Docker logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install pandas==2.0.3 scikit-learn==1.3.0 matplotlib==3.7.2

COPY *.py .
COPY priors_india/ ./priors_india/

#CMD ["python", "-u", "main.py"]
#CMD ["python", "-u", "esp32_optimize.py"]

CMD ["sh", "-c", "python -u main.py && python -u esp32_optimize.py"]