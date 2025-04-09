# Dockerfile para el servicio de monitoreo Kafka
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema, en este caso build-essential por si se necesita
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos (conteniendo únicamente kafka-python y prometheus_client)
COPY requirements.txt ./

# Instalar las dependencias de Python necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script de monitoreo (kafka-monitoring.py) al directorio de trabajo
COPY kafka-monitoring.py ./

# Exponer el puerto donde se servirán las métricas (8765)
EXPOSE 8765

# Comando por defecto para ejecutar el script
CMD ["python", "kafka-monitoring.py"]
