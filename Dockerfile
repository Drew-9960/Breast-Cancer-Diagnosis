FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model/best_breast_cancer_model.pkl .
COPY src/ .

CMD ["bash", "run.sh"]
