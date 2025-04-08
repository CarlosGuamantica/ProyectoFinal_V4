import requests

base_url = "http://localhost:5000/recommend"
num_users = 100 
user_ids = range(1, num_users + 1)  
recommendations = set()

print(f"Verificando porcentaje de recomendaciones únicas con {num_users} usuarios...")

# Enviar solicitudes y recolectar recomendaciones
for user_id in user_ids:
    response = requests.get(f"{base_url}/{user_id}")
    recommendations.add(tuple(response.json()))  # Convertir lista a tupla para poder usar en un set

# Calcular el porcentaje de recomendaciones únicas
unique_recommendations = len(recommendations)
total_recommendations = len(user_ids)
percentage_unique = (unique_recommendations / total_recommendations) * 100

print(f"Número de usuarios probados: {num_users}")
print(f"Recomendaciones únicas: {unique_recommendations}")
print(f"Porcentaje de recomendaciones únicas: {percentage_unique:.2f}%")