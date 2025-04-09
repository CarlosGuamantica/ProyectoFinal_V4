pipeline {
    agent any;
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
                   pip install -r app/requirements.txt
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
