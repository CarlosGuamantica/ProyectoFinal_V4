from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# **Integración Kafka: importar productor Kafka y time**
from kafka import KafkaProducer
import time

# Cargar datos de películas y calificaciones
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Cargar modelos de recomendación entrenados
with open('models/svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)
with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
nn_model = load_model('models/nn_model.h5')
with open('models/mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)
user2idx = mapping['user2idx']
movie2idx = mapping['movie2idx']

# Precalcular popularidad de películas (número de calificaciones por película)
movie_counts = ratings['movieId'].value_counts()

# Obtener lista de todos los géneros disponibles en el conjunto de datos
genres_set = set()
for g_list in movies['genres'].unique():
    for g in g_list.split('|'):
        if g != '(no genres listed)':
            genres_set.add(g)
genres_list = sorted(genres_set)

# Inicializar aplicación Flask
app = Flask(__name__)

# **Integración Kafka: inicializar productor de Kafka**
# Se asume que Kafka broker es alcanzable en "kafka:9092" (según docker-compose)
producer = KafkaProducer(bootstrap_servers='kafka:9092')

# **Función auxiliar para enviar eventos a Kafka**
def log_event_to_kafka(user_id, status_code, latency_ms, action="recommendation request"):
    """Envía un mensaje de evento al tópico Kafka 'movielog1' con el formato requerido."""
    user_str = f"user{user_id}" if user_id is not None else "userNone"
    item_str = f"item{user_id}" if user_id is not None else "itemNone"
    message = f"{user_str},{item_str},{action},{status_code},result,{int(latency_ms)} ms"
    try:
        producer.send('movielog1', value=message.encode('utf-8'))
    except Exception as e:
        print(f"[Kafka] Error enviando evento: {e}")

# ------------------------------------------------------------------------------
# Función principal de recomendación
# ------------------------------------------------------------------------------
def get_recommendations(user_id=None, model_choice='SVD', selected_genres=None, popular_only=False, n=10):
    # Convertir user_id a entero si viene como texto (o a None si vacío)
    user_id = int(user_id) if user_id is not None and user_id != '' else None
    all_movies = movies['movieId'].tolist()
    recommended_movies = []

    # Determinar si el usuario es conocido (existe en los datos y en el modelo)
    user_known = False
    if user_id is not None:
        user_known = user_id in ratings['userId'].values
        if user_known:
            try:
                # Verificar en el modelo Surprise (SVD) si el usuario está en trainset
                _ = svd_model.trainset.to_inner_uid(user_id)
            except Exception:
                user_known = False

    if user_id is None or not user_known:
        # Caso 1: Recomendación no personalizada (usuario no proporcionado o desconocido)
        candidates = set(all_movies)
        if selected_genres:
            # Filtrar películas por géneros seleccionados
            genre_filter = set()
            for genre in selected_genres:
                genre_filter |= set(movies[movies['genres'].str.contains(genre, regex=False)]['movieId'].tolist())
            candidates &= genre_filter
        if popular_only:
            # Filtrar solo películas populares (ej: al menos 50 calificaciones)
            popular_ids = set(movie_counts[movie_counts >= 50].index.tolist())
            candidates &= popular_ids
        if not candidates:
            return []
        # Ordenar candidatos por popularidad (descendente por número de calificaciones)
        candidates_list = list(candidates)
        candidates_list.sort(key=lambda x: movie_counts.get(x, 0), reverse=True)
        top_candidates = candidates_list[:n]
        # Preparar lista de películas recomendadas con detalles
        for mid in top_candidates:
            movie_info = movies[movies['movieId'] == mid].iloc[0]
            recommended_movies.append({
                'title': movie_info['title'],
                'genres': movie_info['genres']
            })
    else:
        # Caso 2: Recomendación personalizada (usuario conocido)
        predicted_ratings = []
        # Construir lista inicial de candidatos (todas las películas o filtradas por género)
        candidates = set(all_movies)
        if selected_genres:
            genre_filter = set()
            for genre in selected_genres:
                genre_filter |= set(movies[movies['genres'].str.contains(genre, regex=False)]['movieId'].tolist())
            candidates &= genre_filter
        if popular_only:
            popular_ids = set(movie_counts[movie_counts >= 50].index.tolist())
            candidates &= popular_ids
        # Si no hay candidatos (tras filtros), retornar vacío
        if not candidates:
            return []
        # Evaluar predicción de rating para cada candidato según el modelo seleccionado
        if model_choice == 'KNN':
            for mid in candidates:
                pred = knn_model.predict(user_id, mid)
                predicted_ratings.append((mid, pred.est))
        elif model_choice == 'SVD':
            for mid in candidates:
                pred = svd_model.predict(user_id, mid)
                predicted_ratings.append((mid, pred.est))
        elif model_choice == 'NN':
            # Para el modelo neuronal: necesitamos índices internos
            if user_id in user2idx:
                user_idx = user2idx[user_id]
                # Preparar lista de índices de películas para los candidatos conocidos por el modelo
                movie_idx_list = []
                mid_list = []
                for mid in candidates:
                    if mid in movie2idx:
                        movie_idx_list.append(movie2idx[mid])
                        mid_list.append(mid)
                if movie_idx_list:
                    user_idx_array = np.array([user_idx] * len(movie_idx_list))
                    movie_idx_array = np.array(movie_idx_list)
                    preds = nn_model.predict([user_idx_array, movie_idx_array], verbose=0).reshape(-1)
                    for m, est in zip(mid_list, preds):
                        predicted_ratings.append((m, float(est)))
        else:
            # Modelo no reconocido
            return []
        # Ordenar películas por rating predicho descendente
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recs = predicted_ratings[:n]
        # Preparar lista de películas recomendadas con detalles
        for mid, score in top_recs:
            movie_info = movies[movies['movieId'] == mid].iloc[0]
            recommended_movies.append({
                'title': movie_info['title'],
                'genres': movie_info['genres']
            })
    return recommended_movies

# ------------------------------------------------------------------------------
# Ruta principal ('/') maneja GET (formulario) y POST (submit del formulario)
# ------------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Marcar tiempo de inicio para medir latencia
        start_time = time.time()
        # Obtener datos del formulario
        user_id = request.form.get('user_id')
        selected_genres = request.form.getlist('genres')
        popular_only = True if request.form.get('popular') == 'on' else False
        model_choice = request.form.get('model_choice')

        # Obtener recomendaciones según la entrada del usuario
        recommendations = get_recommendations(
            user_id=user_id,
            model_choice=model_choice,
            selected_genres=selected_genres,
            popular_only=popular_only,
            n=10
        )

        # Determinar título para mostrar en la página según si es rec personalizada o general
        if user_id is None or user_id == '' or not recommendations:
            recommendations = get_recommendations(
                user_id=None,
                selected_genres=selected_genres,
                popular_only=popular_only,
                n=10
            )
            rec_title = "Películas recomendadas"
        else:
            rec_title = f"Recomendaciones para el usuario {user_id}"

        # Renderizar la plantilla con resultados
        html_response = render_template(
            'index.html',
            genres_list=genres_list,
            recommendations=recommendations,
            rec_title=rec_title,
            selected_genres=selected_genres,
            popular_only=popular_only,
            user_id=user_id,
            model_choice=model_choice
        )
        # Calcular latencia y enviar evento a Kafka
        elapsed_ms = (time.time() - start_time) * 1000.0
        log_event_to_kafka(user_id if user_id else None, 200, elapsed_ms)
        # Devolver la respuesta HTML
        return html_response
    else:
        # Petición GET: mostrar formulario inicial vacío
        return render_template('index.html', genres_list=genres_list, recommendations=None)

# ------------------------------------------------------------------------------
# Ruta API: obtener recomendaciones vía JSON para un user_id dado (GET)
# ------------------------------------------------------------------------------
@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend_userid(user_id):
    start_time = time.time()
    # Por defecto usamos modelo SVD, sin filtros ni popularidad, top 10
    recommendations = get_recommendations(user_id=user_id, model_choice="SVD", selected_genres=[], popular_only=False, n=10)
    elapsed_ms = (time.time() - start_time) * 1000.0
    log_event_to_kafka(user_id, 200, elapsed_ms)
    return jsonify(recommendations)

# ------------------------------------------------------------------------------
# Ejecutar la aplicación (modo standalone)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Ejecutar Flask en host 0.0.0.0 para que sea accesible desde fuera del contenedor
    app.run(host='0.0.0.0', port=5000)
