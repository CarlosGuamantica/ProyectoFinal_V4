pipeline {
    agent {
        docker {
            image 'python:3.9'
            args '--user root'   // Opcional: asegúrate de tener permisos para instalar dependencias
        }
    }
    stages {
        stage('Build') {
            steps {
                echo 'En Python no es necesario compilar (Build paso no hace nada).'
            }
        }
        stage('Test') {
            steps {
                sh '''
                   echo "Instalando dependencias y ejecutando pruebas..."
                   pip install --no-cache-dir -r app/requirements.txt
                   pytest
                   '''
            }
        }
        stage('Deploy') {
            steps {
                echo 'Etapa de despliegue simulada completada.'
            }
        }
    }
}
