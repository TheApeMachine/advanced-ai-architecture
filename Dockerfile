FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt || true
RUN pip install qdrant-client sentence-transformers
CMD ["python", "test.py"]
