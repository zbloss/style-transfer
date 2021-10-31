FROM python:3.7.6

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY .streamlit/config.toml .streamlit/config.toml

COPY utils utils

RUN python utils/get_data.py

COPY app app

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py"]