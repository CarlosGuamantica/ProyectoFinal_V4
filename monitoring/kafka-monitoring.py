from kafka import KafkaConsumer
from prometheus_client import Counter, Histogram, start_http_server

# Definir el tópico a consumir
topic = 'movielog1'

# Iniciar servidor de métricas en el puerto 8765
start_http_server(8765)

# Definir métricas Prometheus
REQUEST_COUNT = Counter(
    'request_count_total', 'Recommendation Request Count',
    ['http_status']
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 'Request latency (seconds)'
)

def main():
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='kafka:9092',
        auto_offset_reset='latest',
        group_id=topic,            # grupo de consumidor
        enable_auto_commit=True,
        auto_commit_interval_ms=1000
    )

    for message in consumer:
        # Decodificar evento a texto y separar campos por coma
        event = message.value.decode('utf-8')
        values = event.split(',')
        # Filtrar solo eventos de "recommendation request"
        if len(values) > 3 and 'recommendation request' in values[2]:
            status_code = values[3].strip()
            # Incrementar métrica de contador según código HTTP
            REQUEST_COUNT.labels(http_status=status_code).inc()
            # Intentar extraer el tiempo de respuesta (latencia) del último campo
            try:
                # El último valor tiene formato "XXX ms"
                time_ms_str = values[-1].strip().split(" ")[0]   # toma la parte numérica
                time_taken = float(time_ms_str)
                # Registrar en histograma (conversión a segundos)
                REQUEST_LATENCY.observe(time_taken / 1000.0)
            except Exception as e:
                print(f"Error procesando tiempo: {values[-1]} - {e}")

if __name__ == "__main__":
    main()
