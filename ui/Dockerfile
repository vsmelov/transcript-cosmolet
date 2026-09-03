FROM python:3.12-slim

# ffmpeg/ffprobe — вырезка фрагментов и конвертация в wav16k для эмбеддингов
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" "psycopg[binary]" \
    numpy "sherpa-onnx>=1.10"

COPY *.py .
COPY static ./static

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
