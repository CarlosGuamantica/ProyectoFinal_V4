FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias necesarias (cliente Kafka y cliente de Prometheus)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script de monitoreo
COPY kafka-monitoring.py ./

# Puerto de exposición de métricas
EXPOSE 8765

# Comando por defecto: ejecutar el script
CMD ["python", "kafka-monitoring.py"]
