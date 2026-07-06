FROM python:3.11-slim

WORKDIR /app

# install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the app
COPY app.py .

# HF Spaces expects the app on port 7860
EXPOSE 7860

# a writable cache dir for models / chroma (HF containers need this set)
ENV HF_HOME=/tmp/hf_home
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/st_home

CMD ["python", "app.py"]
