pipeline {
	agent any
	    stages {
	        stage('Clone Repository') {
	        /* Cloning the repository to our workspace */
	        steps {
	        checkout scm
	        }
	   }
	   stage('Build Image') {
	        steps {
	        sh 'sudo docker build -t salarypredapp:v1 .'
	        }
	   }
	   stage('Run Image') {
	        steps {
		def x = Math.abs(new Random().nextInt())
		def y = "salarypred" + x.toString()
	        sh 'sudo docker run -d -p 8501:8501 --name y salarypredapp:v1'
	        }
	   }
	   stage('Testing'){
	        steps {
	            echo 'Testing..'
		    sh 'ifconfig enp0s3'
	            }
	   }
    }
}
