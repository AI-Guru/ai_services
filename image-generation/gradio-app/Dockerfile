# Gradio frontend for the self-hosted text-to-image services.
# Tiny CPU-only image — it only does HTTP to the model containers.
FROM python:3.12-slim

RUN pip install --no-cache-dir gradio requests pillow

COPY app.py /app/app.py
WORKDIR /app

EXPOSE 7860
CMD ["python3", "app.py"]
