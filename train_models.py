import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
import mlflow  # NUEVO: integración con MLflow

# Configurar el experimento en MLflow. Se creará uno llamado "MovieRecommender".
mlflow.set_experiment("MovieRecommender")

# Iniciar un run de MLflow para este entrenamiento
with mlflow.start_run():
    # -----------------------------------------------------------
    # Cargar datos
    # -----------------------------------------------------------
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')

    # -----------------------------------------------------------
    # Preparar datos para modelos basados en Surprise (colaborativo)
    # -----------------------------------------------------------
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Dividir datos en entrenamiento y prueba para evaluación interna
    trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

    # -----------------------------------------------------------
    # Entrenar modelo KNN básico (Filtrado colaborativo basado en similitud)
    # -----------------------------------------------------------
    start_time = time.time()
    knn_model = KNNBasic()
    knn_model.fit(trainset)
    knn_training_time = time.time() - start_time

    # -----------------------------------------------------------
    # Entrenar modelo SVD (Filtrado colaborativo mediante descomposición matricial)
    # -----------------------------------------------------------
    start_time = time.time()
    svd_model = SVD()
    svd_model.fit(trainset)
    svd_training_time = time.time() - start_time

    # -----------------------------------------------------------
    # Preparar datos para el modelo de deep learning (Neural Collaborative Filtering)
    # Mapear IDs de usuario y película a índices consecutivos
    # -----------------------------------------------------------
    unique_users = ratings['userId'].unique()
    unique_movies = ratings['movieId'].unique()
    unique_users_sorted = np.sort(unique_users)
    unique_movies_sorted = np.sort(unique_movies)
    user2idx = {u: idx for idx, u in enumerate(unique_users_sorted)}
    movie2idx = {m: idx for idx, m in enumerate(unique_movies_sorted)}
    ratings['user_idx'] = ratings['userId'].map(user2idx)
    ratings['movie_idx'] = ratings['movieId'].map(movie2idx)

    # Dividir en conjuntos de entrenamiento y validación para la red neuronal
    train_data, val_data = train_test_split(ratings, test_size=0.2, random_state=42)
    x_train = [train_data['user_idx'].values, train_data['movie_idx'].values]
    y_train = train_data['rating'].values
    x_val = [val_data['user_idx'].values, val_data['movie_idx'].values]
    y_val = val_data['rating'].values

    # -----------------------------------------------------------
    # Definir la arquitectura del modelo neuronal (Embedding + MLP)
    # -----------------------------------------------------------
    num_users = len(user2idx)
    num_movies = len(movie2idx)
    embedding_dim = 50

    user_input = Input(shape=(1,), name='user')
    item_input = Input(shape=(1,), name='item')
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, input_length=1, name='user_emb')(user_input)
    item_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim, input_length=1, name='item_emb')(item_input)
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)
    concat = Concatenate()([user_vec, item_vec])
    dense = Dense(64, activation='relu')(concat)
    dense2 = Dense(32, activation='relu')(dense)
    output = Dense(1, activation='linear')(dense2)
    nn_model = Model([user_input, item_input], output)
    nn_model.compile(optimizer='adam', loss='mean_squared_error')

    # -----------------------------------------------------------
    # Entrenar el modelo neuronal y medir su tiempo de entrenamiento
    # -----------------------------------------------------------
    start_time = time.time()
    nn_model.fit(x_train, y_train, epochs=5, batch_size=256, validation_data=(x_val, y_val), verbose=1)
    nn_training_time = time.time() - start_time

    # -----------------------------------------------------------
    # Evaluar el desempeño del modelo neuronal en el conjunto de validación (RMSE)
    # -----------------------------------------------------------
    preds = nn_model.predict(x_val)
    nn_rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"RMSE del modelo neuronal en validación: {nn_rmse:.4f}")

    # -----------------------------------------------------------
    # Evaluación en conjunto de prueba para modelos colaborativos
    # -----------------------------------------------------------
    knn_predictions = knn_model.test(testset)
    knn_test_rmse = accuracy.rmse(knn_predictions, verbose=False)
    svd_predictions = svd_model.test(testset)
    svd_test_rmse = accuracy.rmse(svd_predictions, verbose=False)

    # Función para evaluar el modelo neuronal en el conjunto de prueba (para usuarios/películas conocidos)
    def evaluate_nn_model(nn_model, testset):
        known_user_indices = []
        known_movie_indices = []
        true_ratings = []
        for (uid, iid, true_r) in testset:
            if uid in user2idx and iid in movie2idx:
                known_user_indices.append(user2idx[uid])
                known_movie_indices.append(movie2idx[iid])
                true_ratings.append(true_r)
        if not true_ratings:
            return None
        user_arr = np.array(known_user_indices)
        movie_arr = np.array(known_movie_indices)
        preds = nn_model.predict([user_arr, movie_arr], verbose=0).reshape(-1)
        return np.sqrt(mean_squared_error(true_ratings, preds))

    nn_test_rmse = evaluate_nn_model(nn_model, testset)

    # Imprimir resultados de evaluación
    print(f"RMSE del modelo KNN en prueba: {knn_test_rmse:.4f}")
    print(f"RMSE del modelo SVD en prueba: {svd_test_rmse:.4f}")
    if nn_test_rmse is not None:
        print(f"RMSE del modelo neuronal en prueba: {nn_test_rmse:.4f}")
    else:
        print("Modelo neuronal no evaluado en prueba (usuarios/películas desconocidos en test).")

    # -----------------------------------------------------------
    # Guardar modelos entrenados y tiempos (y mapeos) en la carpeta 'models'
    # -----------------------------------------------------------
    import os
    os.makedirs('models', exist_ok=True)
    with open('models/knn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    with open('models/svd_model.pkl', 'wb') as f:
        pickle.dump(svd_model, f)
    # Guardar el modelo neuronal en formato h5
    nn_model.save('models/nn_model.h5')
    # Guardar mapeos de IDs de usuario y película
    with open('models/mapping.pkl', 'wb') as f:
        pickle.dump({'user2idx': user2idx, 'movie2idx': movie2idx}, f)
    # Guardar tiempos de entrenamiento para referencia
    with open('models/training_times.pkl', 'wb') as f:
        pickle.dump({'knn': knn_training_time, 'svd': svd_training_time, 'nn': nn_training_time}, f)

    print("Modelos entrenados y guardados en la carpeta 'models'.")

    # -----------------------------------------------------------
    # Registrar en MLflow parámetros, métricas y artefactos
    # -----------------------------------------------------------
    # Parámetros de entrenamiento
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("epochs", 5)

    # Métricas de evaluación en el set de prueba
    mlflow.log_metric("knn_rmse_test", knn_test_rmse)
    mlflow.log_metric("svd_rmse_test", svd_test_rmse)
    if nn_test_rmse is not None:
        mlflow.log_metric("nn_rmse_test", nn_test_rmse)
    mlflow.log_metric("nn_rmse_val", nn_rmse)

    # Tiempos de entrenamiento
    mlflow.log_metric("knn_train_time_sec", knn_training_time)
    mlflow.log_metric("svd_train_time_sec", svd_training_time)
    mlflow.log_metric("nn_train_time_sec", nn_training_time)

    # Registrar los artefactos (modelos y otros archivos importantes)
    mlflow.log_artifact('models/knn_model.pkl')
    mlflow.log_artifact('models/svd_model.pkl')
    mlflow.log_artifact('models/nn_model.h5')
    mlflow.log_artifact('models/mapping.pkl')
    mlflow.log_artifact('models/training_times.pkl')
