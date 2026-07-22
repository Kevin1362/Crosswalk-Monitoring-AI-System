FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install -r dev/requirements.txt

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "dashboard/dashboard.py", "--server.address=0.0.0.0"]