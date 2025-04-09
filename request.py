import requests
import time
import sys
import logging
from statistics import mean

# Configurar logging para mostrar mensajes informativos y de error
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_url = "http://localhost:5011/recommend"
user_ids = range(1, 100)  # 50 user_ids
total_requests = 2000
successful_requests = 0
failed_requests = 0
response_times = []

logging.info(f"Enviando {total_requests} solicitudes...")

start_time = time.perf_counter()

for i in range(total_requests):
    user_id = user_ids[i % len(user_ids)]
    try:
        request_start = time.perf_counter()
        response = requests.get(f"{base_url}/{user_id}", timeout=3.0)
        request_time = time.perf_counter() - request_start
        response_times.append(request_time)
        if response.status_code == 200:
            successful_requests += 1
        else:
            failed_requests += 1
            logging.warning(f"Respuesta con status {response.status_code} para user_id {user_id}")
        sys.stdout.write(f"\rSolicitudes exitosas: {successful_requests} | Fallidas: {failed_requests}")
        sys.stdout.flush()
    except requests.exceptions.RequestException as e:
        failed_requests += 1
        logging.error(f"Error en la solicitud para user_id {user_id}: {e}")
        sys.stdout.write(f"\rSolicitudes exitosas: {successful_requests} | Fallidas: {failed_requests}")
        sys.stdout.flush()

end_time = time.perf_counter()
total_time = end_time - start_time

# Calcular estadísticas de tiempo de respuesta
avg_response_time = mean(response_times) if response_times else 0
min_response_time = min(response_times) if response_times else 0
max_response_time = max(response_times) if response_times else 0

logging.info("Pruebas completadas.")
logging.info(f"Total de solicitudes: {total_requests}")
logging.info(f"Solicitudes exitosas: {successful_requests}")
logging.info(f"Solicitudes fallidas: {failed_requests}")
logging.info(f"Tiempo total: {total_time:.2f} segundos")
logging.info(f"Tiempo promedio de respuesta: {avg_response_time:.3f} s")
logging.info(f"Tiempo mínimo de respuesta: {min_response_time:.3f} s")
logging.info(f"Tiempo máximo de respuesta: {max_response_time:.3f} s")
