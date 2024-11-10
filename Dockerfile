FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

EXPOSE 8501

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 80 & streamlit run streamlit1.py --server.port 8501 --server.address 0.0.0.0"]
