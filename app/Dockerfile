# Dockerfile para la aplicación Flask
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias de sistema esenciales (compilador, cabeceras de Python, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos (requirements.txt propio de la aplicación)
COPY requirements.txt ./

# Instalar las dependencias de Python necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la aplicación, datos, modelos y demás archivos
COPY . ./

# Exponer el puerto donde se ejecuta la app (5000)
EXPOSE 5000

# Comando por defecto para ejecutar la aplicación Flask
CMD ["python", "app.py"]
