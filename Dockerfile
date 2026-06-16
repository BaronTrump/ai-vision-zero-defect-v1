FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY 01-ai-vision-defect-detector/requirements.txt /app/p1-requirements.txt
COPY 02-robot-vision-guidance/requirements.txt /app/p2-requirements.txt
COPY 03-production-anomaly-detector/requirements.txt /app/p3-requirements.txt
COPY 04-production-monitor-dashboard/requirements.txt /app/p4-requirements.txt

RUN pip install --no-cache-dir -r p1-requirements.txt \
    && pip install --no-cache-dir -r p2-requirements.txt \
    && pip install --no-cache-dir -r p3-requirements.txt \
    && pip install --no-cache-dir -r p4-requirements.txt

COPY . /app

RUN mkdir -p /app/01-ai-vision-defect-detector/data \
    && mkdir -p /app/01-ai-vision-defect-detector/models \
    && mkdir -p /app/02-robot-vision-guidance/data \
    && mkdir -p /app/03-production-anomaly-detector/data \
    && mkdir -p /app/03-production-anomaly-detector/models \
    && mkdir -p /app/04-production-monitor-dashboard/data \
    && mkdir -p /app/04-production-monitor-dashboard/reports

EXPOSE 8501 8502 8503 8504

CMD ["streamlit", "run"]
