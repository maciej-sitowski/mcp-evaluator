FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# docker-compose bind-mounts this directory over /app for quick iteration;
# the COPY keeps the image runnable on its own.
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]
