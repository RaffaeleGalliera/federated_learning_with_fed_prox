# Federated Learning with Fed Prox

This project contains a simple example of Federated Learning using the Fed Prox algorithm and MNIST as dataset.
I have built this project to learn more about Federated Learning and have some hands-on experience with it! 
It is still primitive and far from being a complete, production-ready implementation of a Federate Learning service, but it can be still used as a starting point to play with it! :)

Different equally-tiered containers can be launched, and they can either serve as a Server/HQ or a Client. 
Each container will run a Flask server and a series of routes will be exposed to allow the federation process to be managed by a third application.

A Dockerfile is provided to build an image supporting the application and the `compose.yaml` allows to run 3 containers locally.
`example.py` implements an example of workflow for the federated learning process. 


### Project Structure

The project is structured as follows:

- `app/`: This directory contains the main application code.
  - `main.py`: This is the main application file. It contains the Flask server and the routes for the application.
  - `example.py`: This file contains an example of a federated learning process.
  - `utils/`: This directory contains utility files for the application.
- `compose.yaml`: This is the Docker Compose file for the application. It defines the services that make up the application.

### Running the Application

To run the application, you need to have Docker and Docker Compose installed on your machine. Once you have these installed, you can start the application by running the following command in the root directory of the project:

```bash
docker-compose up
```