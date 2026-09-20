FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip uninstall urllib3 -y
RUN pip install urllib3

COPY run_evaluation.py .

CMD ["python3", "run_evaluation.py"]