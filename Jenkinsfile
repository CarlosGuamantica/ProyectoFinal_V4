pipeline {
    agent any
    stages {
        stage('Instalar dependencias') {
            steps {
                // Instalamos las librerías Python requeridas
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Entrenar modelos') {
            steps {
                // Ejecutamos el script de entrenamiento (entrena y guarda los modelos)
                sh 'python train_models.py'
            }
        }
        stage('Evaluar modelos') {
            steps {
                // Ejecutamos el script de evaluación (calcula RMSE, tiempos, etc.)
                sh 'python evaluate_models.py'
            }
        }
        stage('Construir imagen Docker') {
            steps {
                // Construye la imagen Docker con la aplicación y modelos (tag local "latest")
                sh 'docker build -t movies-recommender:latest .'
            }
        }
        stage('Publicar imagen') {
            steps {
                // (Opcional) Publicar la imagen en un registro de contenedores (Docker Hub, ECR, etc.)
                // Descomentar y configurar las siguientes líneas si es necesario publicar la imagen:
                // withCredentials([usernamePassword(credentialsId: 'creds-dockerhub', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                //     sh "echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin"
                //     sh 'docker tag movies-recommender:latest <usuario>/movies-recommender:latest'
                //     sh 'docker push <usuario>/movies-recommender:latest'
                // }
                echo 'Imagen Docker construida localmente (sin publicar en registro externo).'
            }
        }
        stage('Desplegar a Kubernetes') {
            steps {
                // Aplica los manifiestos de Kubernetes para desplegar la app en el cluster
                sh 'kubectl apply -f k8s/deployment.yaml'
                sh 'kubectl apply -f k8s/service.yaml'
            }
        }
    }
}
