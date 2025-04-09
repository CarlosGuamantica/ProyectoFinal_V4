from kafka import KafkaProducer
import time
import random

producer = KafkaProducer(bootstrap_servers='kafka:9092')
topic = 'movielog1'
acciones = ['recommendation request', 'search', 'click']

print(f"Enviando mensajes de prueba al tópico '{topic}'...")
for i in range(20):
    status_code = random.choice(['200', '500'])
    latency = random.randint(100, 800)  # ms
    action = random.choice(acciones)
    message = f"user{i},item{i},{action},{status_code},message text {i},{latency} ms"
    print(f"[{i+1}] {message}")
    producer.send(topic, value=message.encode('utf-8'))
    time.sleep(1)

producer.flush()
print("Envío de mensajes de prueba completado.")
