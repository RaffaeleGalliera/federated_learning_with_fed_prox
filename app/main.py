import random
import numpy as np
import os
import torch
import torchvision.datasets
from flask import Flask, request, jsonify
from torchvision import transforms
import io
from PIL import Image

from torch.utils.data import DataLoader, Dataset

from utils.datasets import CustomDatasetWithLabelsList, CustomDataset
from utils.network import CNN
from utils.utils import run_model, run_monte_carlo, train_model, save_model, \
    validate_model, train_model_with_fedprox, model_to_hex, federated_average, \
    save_and_load_hex_model


class FederatedLearningServer(Flask):
    def __init__(self, *args, **kwargs):
        super(FederatedLearningServer, self).__init__(*args, **kwargs)
        self.model = CNN()
        # Check if current_model.pth exists and load it
        if os.path.exists("current_model.pth"):
            self.model.load_state_dict(torch.load("current_model.pth"))


app = FederatedLearningServer(__name__)


@app.route('/infer', methods=['POST'])
def infer():
    image = request.files['image'].read()
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor()]
    )
    image = Image.open(io.BytesIO(image))
    image = transform(image).unsqueeze(0)
    prediction = run_model(image, app.model)
    return jsonify(
        {
            'prediction': prediction[1],
            'confidence': prediction[0]
        }
    )


@app.route('/infer_with_uncertainty', methods=['POST'])
def infer_with_uncertainty():
    image = request.files['image'].read()
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor()]
    )
    image = Image.open(io.BytesIO(image))
    image = transform(image).unsqueeze(0)
    prediction = run_model(image, app.model)
    _, _, uncertainty = run_monte_carlo(image, app.model)
    return jsonify(
        {
            'prediction': prediction[1],
            'confidence': prediction[0],
            'uncertainty': uncertainty
        }
    )


@app.route('/train-worker-directory', methods=['POST'])
def train_worker_directory():
    directory = request.json['directory']
    # Load dataset from directory
    dataset = CustomDataset(
        directory,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    trained_model = train_model(app.model, data_loader)
    save_model(trained_model, "current_model")
    loss, accuracy = validate_model(app.model)

    app.model = trained_model
    return jsonify(
        {
            'accuracy': accuracy,
            'loss': loss
        }
    )


@app.route('/train-worker-on-labels-list', methods=['POST'])
def train_worker_directory_on_labels_list():
    directory = request.json['directory']
    labels_list = request.json['labels_list']
    # Load dataset from directory
    dataset = CustomDatasetWithLabelsList(
        directory,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]),
        labels_list=labels_list
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    trained_model = train_model(app.model, data_loader)
    save_model(trained_model, "current_model")
    # Run validation
    app.model = trained_model
    loss, accuracy = validate_model(app.model, labels_list=labels_list)
    return jsonify(
        {
            'loss': loss,
            'accuracy': accuracy
        }
    )


@app.route('/fed-prox-train-worker-on-labels-list', methods=['POST'])
def fed_prox_train_worker_directory_on_labels_list():
    directory = request.json['directory']
    labels_list = request.json['labels_list']
    global_model = request.json.get('global_model', None)

    if global_model is None:
        global_model = app.model.state_dict()
    else:
        global_model = torch.load(io.BytesIO(bytes.fromhex(global_model)))

    # Load dataset from directory
    dataset = CustomDatasetWithLabelsList(
        directory,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]),
        labels_list=labels_list
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    temp_model = app.model
    trained_model = train_model_with_fedprox(
        temp_model,
        global_model,
        data_loader
    )
    loss, accuracy = validate_model(temp_model, labels_list=labels_list)
    hex_model = model_to_hex(trained_model)
    return jsonify(
        {
            'model': hex_model,
            'loss': loss,
            'accuracy': accuracy
        }
    )


@app.route('/validate-worker-on-labels-list', methods=['POST'])
def validate_worker_directory_on_labels_list():
    labels_list = request.json['labels_list']
    loss, accuracy = validate_model(app.model, labels_list=labels_list)
    return jsonify(
        {
            'loss': loss,
            'accuracy': accuracy
        }
    )


@app.route('/fed-avg', methods=['POST'])
def federated_average_route():
    data = request.json
    models_state_dict = [save_and_load_hex_model(model_hex, f"model_{i}")
                         for i, model_hex in enumerate(data['models'])]
    weights = data['weights']
    avg_state_dict = federated_average(models_state_dict, weights)
    new_model = CNN()
    new_model.load_state_dict(avg_state_dict)
    save_model(new_model, "current_model")
    app.model = new_model
    hex_model = model_to_hex(new_model)
    return jsonify({'model': hex_model})


@app.route('/set-model', methods=['POST'])
def set_model():
    model = request.json['model']
    app.model.load_state_dict(
        save_and_load_hex_model(model, "current_model")
    )
    return jsonify({'status': 'success'})


@app.route('/get-model', methods=['GET'])
def get_model():
    hex_model = model_to_hex(app.model)
    return jsonify({'model': hex_model})


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    app.run(host='0.0.0.0', port=8000)
