import pandas as pd
import pickle
import time
import os
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from tensorflow.keras.models import load_model
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

# Para crear un CSV con las predicciones por instancia
import csv

# Cargar datos de calificaciones
ratings = pd.read_csv('data/ratings.csv')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

# Cargar modelos entrenados
with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('models/svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)
# Cargar modelo neuronal y mapeos
nn_model = load_model('models/nn_model.h5')
with open('models/mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)
user2idx = mapping['user2idx']
movie2idx = mapping['movie2idx']

# Cargar tiempos de entrenamiento guardados
with open('models/training_times.pkl', 'rb') as f:
    training_times = pickle.load(f)
knn_training_time = training_times.get('knn', None)
svd_training_time = training_times.get('svd', None)
nn_training_time = training_times.get('nn', None)

# Función para evaluar RMSE de un modelo Surprise en el conjunto de prueba
def evaluate_surprise_model(model, testset):
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return rmse

# Función para evaluar RMSE del modelo neuronal en el conjunto de prueba
def evaluate_nn_model(nn_model, testset):
    # Filtrar solo las instancias de prueba cuyos usuarios y películas están en el modelo neuronal
    known_user_indices = []
    known_movie_indices = []
    true_ratings = []
    for (uid, iid, true_r) in testset:
        if uid in user2idx and iid in movie2idx:
            known_user_indices.append(user2idx[uid])
            known_movie_indices.append(movie2idx[iid])
            true_ratings.append(true_r)
    if len(true_ratings) == 0:
        return None  # si no hay datos utilizables
    # Predecir con el modelo neuronal
    user_arr = np.array(known_user_indices)
    movie_arr = np.array(known_movie_indices)
    preds = nn_model.predict([user_arr, movie_arr], verbose=0)
    preds = preds.reshape(-1)  # convertir a 1D
    rmse = sqrt(mean_squared_error(true_ratings, preds))
    return rmse

# Evaluar RMSE y mostrar resultados
knn_rmse = evaluate_surprise_model(knn_model, testset)
svd_rmse = evaluate_surprise_model(svd_model, testset)

def measure_inference_time(model, predict_func, user_id, item_id, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        predict_func(user_id, item_id)
    avg_time = (time.time() - start_time) / iterations
    return avg_time * 1000.0  # en milisegundos

surprise_pred = lambda model, u, i: model.predict(u, i)
nn_pred = lambda nn_model, u_idx, i_idx: nn_model.predict([np.array([u_idx]), np.array([i_idx])], verbose=0)

# Seleccionar un ejemplo de usuario-película
example_uid, example_iid = None, None
for (uid, iid, r) in testset:
    if uid in user2idx and iid in movie2idx:
        example_uid, example_iid = uid, iid
        break
if example_uid is None:
    # Si no hay data neuronal en test, se usa la primera para surprise
    example_uid, example_iid = testset[0][0], testset[0][1]

knn_infer_ms = measure_inference_time(knn_model, lambda u,i: surprise_pred(knn_model, u, i), example_uid, example_iid)
svd_infer_ms = measure_inference_time(svd_model, lambda u,i: surprise_pred(svd_model, u, i), example_uid, example_iid)
nn_rmse = None
nn_infer_ms = None

# Evaluar NN solo si el example_uid es compatible
if example_uid in user2idx and example_iid in movie2idx:
    nn_rmse = evaluate_nn_model(nn_model, testset)
    nn_infer_ms = measure_inference_time(
        nn_model,
        lambda u,i: nn_pred(nn_model, user2idx[u], movie2idx[i]),
        example_uid, example_iid
    )

knn_size = os.path.getsize('models/knn_model.pkl') / (1024 * 1024)
svd_size = os.path.getsize('models/svd_model.pkl') / (1024 * 1024)
nn_size = os.path.getsize('models/nn_model.h5') / (1024 * 1024)

print("Resultado de la evaluación de modelos:")
print("-------------------------------------")
print(f"KNNBasic -> RMSE: {knn_rmse:.4f}, Tiempo entrenamiento: {knn_training_time:.2f}s, Inferencia: {knn_infer_ms:.4f}ms, Tamaño: {knn_size:.2f}MB")
print(f"SVD -> RMSE: {svd_rmse:.4f}, Tiempo entrenamiento: {svd_training_time:.2f}s, Inferencia: {svd_infer_ms:.4f}ms, Tamaño: {svd_size:.2f}MB")

if nn_rmse is not None and nn_infer_ms is not None:
    print(f"NN -> RMSE: {nn_rmse:.4f}, Tiempo entrenamiento: {nn_training_time:.2f}s, Inferencia: {nn_infer_ms:.4f}ms, Tamaño: {nn_size:.2f}MB")
else:
    print(f"NN -> RMSE: N/A o {nn_rmse}, Tiempo entrenamiento: {nn_training_time:.2f}s, Inferencia: N/A, Tamaño: {nn_size:.2f}MB")

# -----------------------------------------------------------
# Generar CSV para Zeno: predicciones por instancia (user, item, rating, pred, model)
# -----------------------------------------------------------
# Recorremos testset y para cada modelo generamos filas.

output_file = "zeno_predictions.csv"
with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["user_id","movie_id","true_rating","pred_rating","model"])
    writer.writeheader()

    # 1) KNN
    for (uid, iid, true_r) in testset:
        pred = knn_model.predict(uid, iid)
        row = {
            "user_id": uid,
            "movie_id": iid,
            "true_rating": true_r,
            "pred_rating": round(pred.est,3),
            "model": "KNN"
        }
        writer.writerow(row)

    # 2) SVD
    for (uid, iid, true_r) in testset:
        pred = svd_model.predict(uid, iid)
        row = {
            "user_id": uid,
            "movie_id": iid,
            "true_rating": true_r,
            "pred_rating": round(pred.est,3),
            "model": "SVD"
        }
        writer.writerow(row)

    # 3) NN (solo si (uid, iid) está en user2idx, movie2idx)
    for (uid, iid, true_r) in testset:
        if uid in user2idx and iid in movie2idx:
            user_idx_val = user2idx[uid]
            movie_idx_val = movie2idx[iid]
            preds = nn_model.predict([np.array([user_idx_val]), np.array([movie_idx_val])], verbose=0)
            pred_val = float(preds[0][0])
            row = {
                "user_id": uid,
                "movie_id": iid,
                "true_rating": true_r,
                "pred_rating": round(pred_val,3),
                "model": "NN"
            }
            writer.writerow(row)

print(f"Archivo '{output_file}' generado con predicciones para Zeno.")
print("Puedes explorar este CSV con Zeno o cualquier herramienta de visualización.")
