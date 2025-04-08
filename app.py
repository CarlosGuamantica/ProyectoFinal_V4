from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import numpy as np
from prometheus_flask_exporter import PrometheusMetrics  # NUEVO: para monitoreo

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

# Precalcular datos de popularidad de películas (número de calificaciones por película)
movie_counts = ratings['movieId'].value_counts()

# Obtener lista de todos los géneros disponibles
genres_set = set()
for g_list in movies['genres'].unique():
    # algunos géneros están separados por '|'
    for g in g_list.split('|'):
        if g != '(no genres listed)':
            genres_set.add(g)
genres_list = sorted(genres_set)

# Inicializar aplicación Flask
app = Flask(__name__)

# Inicializar PrometheusMetrics para exponer métricas en /metrics automáticamente
metrics = PrometheusMetrics(app)

# ---------------------------------------------------------------------------------
# Función de recomendación principal
# ---------------------------------------------------------------------------------
def get_recommendations(user_id=None, model_choice='SVD', selected_genres=None, popular_only=False, n=10):
    # Convertir user_id a tipo correcto
    user_id = int(user_id) if user_id is not None and user_id != '' else None
    # Obtener lista completa de películas
    all_movies = movies['movieId'].tolist()
    recommended_movies = []
    # Si no se especifica usuario (o usuario no conocido), usar recomendación general basada en popularidad
    user_known = False
    if user_id is not None:
        # Verificar si el usuario existe en los datos
        user_known = user_id in ratings['userId'].values
        # También verificar si el modelo colaborativo conoce al usuario (por seguridad)
        if user_known:
            try:
                # Surprise model check (si es modelo Surprise)
                _ = svd_model.trainset.to_inner_uid(user_id)
            except Exception:
                user_known = False

    if user_id is None or not user_known:
        # Recomendación no personalizada (popularidad global, con filtros aplicados)
        candidates = set(all_movies)
        if selected_genres:
            # Filtrar películas por los géneros seleccionados
            genre_filter = set()
            for genre in selected_genres:
                genre_filter |= set(movies[movies['genres'].str.contains(genre, regex=False)]['movieId'].tolist())
            candidates &= genre_filter
        if popular_only:
            # Filtrar solo películas populares (ej: al menos 50 calificaciones)
            popular_ids = set(movie_counts[movie_counts >= 50].index.tolist())
            candidates &= popular_ids
        # Si no hay candidato tras filtros, retornar lista vacía
        if not candidates:
            return []
        # Ordenar candidatos por popularidad (número de calificaciones) descendentemente
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
        return recommended_movies

    # Si usuario es conocido y se solicita recomendación personalizada
    # Generar lista de candidatos (películas no calificadas por el usuario)
    user_rated_movies = set(ratings[ratings['userId'] == user_id]['movieId'].tolist())
    candidates = [m for m in all_movies if m not in user_rated_movies]

    # Aplicar filtro de género si corresponde
    if selected_genres:
        filtered_candidates = []
        for mid in candidates:
            genres_str = movies[movies['movieId'] == mid]['genres'].values[0]
            # Comprobar si la película tiene al menos uno de los géneros seleccionados
            for genre in selected_genres:
                if genre in genres_str:
                    filtered_candidates.append(mid)
                    break
        candidates = filtered_candidates

    # Aplicar filtro de popularidad si corresponde
    if popular_only:
        candidates = [mid for mid in candidates if movie_counts.get(mid, 0) >= 50]

    # Si no quedan candidatos, retornar vacío
    if not candidates:
        return []

    # Seleccionar modelo de recomendación
    model_choice = model_choice.upper()
    predicted_ratings = []

    if model_choice == 'KNN':
        for mid in candidates:
            pred = knn_model.predict(user_id, mid)
            predicted_ratings.append((mid, pred.est))
    elif model_choice == 'SVD':
        for mid in candidates:
            pred = svd_model.predict(user_id, mid)
            predicted_ratings.append((mid, pred.est))
    elif model_choice == 'NN':
        # Modelo neuronal requiere índices internos
        if user_id in user2idx:
            user_idx = user2idx[user_id]
            # preparar lista de índices de películas para candidatos conocidos por el modelo
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
        # Modelo no reconocido, retornar vacío
        return []

    # Ordenar películas por la estimación de calificación descendente
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

# ---------------------------------------------------------------------------------
# Ruta principal (inicio) que maneja tanto GET (mostrar formulario) como POST
# ---------------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
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

        # Si no se ingresó un usuario o no hay recomendaciones personalizadas,
        # mostrar recomendaciones generales
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

        return render_template(
            'index.html',
            genres_list=genres_list,
            recommendations=recommendations,
            rec_title=rec_title,
            selected_genres=selected_genres,
            popular_only=popular_only,
            user_id=user_id,
            model_choice=model_choice
        )

    else:
        # GET: mostrar página inicial con algunas películas populares por defecto
        # Top 10 películas más populares (por número de calificaciones)
        popular_ids = movie_counts.nlargest(10).index.tolist()
        popular_movies = []
        for mid in popular_ids:
            movie_info = movies[movies['movieId'] == mid].iloc[0]
            popular_movies.append({
                'title': movie_info['title'],
                'genres': movie_info['genres']
            })

        return render_template(
            'index.html',
            genres_list=genres_list,
            recommendations=popular_movies,
            rec_title="Películas más populares",
            selected_genres=[],
            popular_only=False,
            user_id="",
            model_choice="SVD"
        )

# ---------------------------------------------------------------------------------
# NUEVA RUTA PARA /recommend/<int:user_id> (Soporta GET)
# Esto permite a requests2000.py llamar directamente a /recommend/1, /recommend/2, etc.
# ---------------------------------------------------------------------------------
@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend_userid(user_id):
    # Por defecto usamos SVD, sin filtros ni popularidad
    recommendations = get_recommendations(user_id=user_id, model_choice="SVD", selected_genres=[], popular_only=False, n=10)
    return jsonify(recommendations)

# ---------------------------------------------------------------------------------
# Endpoints adicionales para monitoreo y salud
# ---------------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    # Endpoint de salud para liveness/readiness de Kubernetes
    return jsonify({"status": "ok"}), 200

# ---------------------------------------------------------------------------------
# Ejecutar la aplicación (modo desarrollo)
# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    # Para desarrollo (usualmente en producción se usa Gunicorn o similar, y en Kubernetes se despliega mediante Docker)
    app.run(debug=True)
