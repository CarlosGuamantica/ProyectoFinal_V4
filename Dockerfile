# Imagen base con Python 3.9
FROM python:3.9-slim

# Directorio de trabajo de la app
WORKDIR /app

# Instalar dependencias del sistema y Python
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y build-essential && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copiar el código de la aplicación, modelos entrenados y datos necesarios
COPY . .

# Exponer el puerto de la aplicación (Flask/Gunicorn)
EXPOSE 5000

# Comando para ejecutar la aplicación con Gunicorn (4 workers)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--workers=4"]
