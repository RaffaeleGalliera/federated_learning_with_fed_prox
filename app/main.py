# Seed setting
import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import os
import torch
import torchvision.datasets
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import copy


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320,50)
        self.dropout = nn.Dropout(0.3)
        # Output layer
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        # Get images from from directory going recursively into directory/label/*.jpg
        self.images = [os.path.join(dp, f)
                       for dp, dn, fn in os.walk(os.path.expanduser(directory))
                       for f in fn]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        label = self._get_label(image_path)  # Implement this method based on your labeling strategy

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_label(self, image_path):
        return torch.tensor(int(Path(image_path).parts[-2]))


class CustomDatasetWithLabelsList(CustomDataset):
    def __init__(self, directory, labels_list, transform=None):
        super().__init__(directory, transform)
        self.labels_list = labels_list
        self.images = [os.path.join(dp, f)
                       for dp, dn, fn in os.walk(os.path.expanduser(directory))
                       for f in fn
                       if Path(dp).parts[-1] in self.labels_list]


class CustomDatasetFromPOST(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_file = self.file_list[idx]
        image = Image.open(image_file.stream).convert('L')  # Convert to grayscale
        label = self._get_label(image_file)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_label(self, image_file):
        # Implement label extraction from image file if needed
        return 0  # Placeholder


def model_to_hex(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue().hex()  # Convert bytes to hex string


def save_and_load_hex_model(serialized_model, model_name):
    with open(f"{model_name}.pth", 'wb') as f:
        f.write(bytes.fromhex(serialized_model))
    model = torch.load(f"{model_name}.pth")
    return model


def save_model(model, model_name):
    with open(f"{model_name}.pth", 'wb') as f:
        torch.save(model.state_dict(), f)


def validate_model(model,
                   labels_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                   directory='dataset/evaluation'):
    validation_dataset = CustomDatasetWithLabelsList(
        directory,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]),
        labels_list=labels_list
    )

    data_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=True
    )

    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in data_loader:
            output = model(inputs)
            test_loss += criterion(output, labels)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return test_loss.item(), accuracy.item()


def train_model(model, data_loader, epochs=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


def train_model_with_fedprox(
        local_model,
        global_model_state_dict,
        data_loader,
        mu=0.001,
        epochs=3
):
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters())
    for epoch in range(epochs):
        for data in data_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)

            # FedProx term
            fedprox_loss = 0
            for param, global_param in zip(local_model.parameters(),
                                           global_model_state_dict.values()):
                fedprox_loss += (mu / 2) * (param - global_param).norm(2)

            total_loss = loss + fedprox_loss
            total_loss.backward()
            optimizer.step()

    return local_model


def run_model(image, model):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        conf, predicted = torch.max(outputs, 1)

    return conf.item(), predicted.item()


def run_monte_carlo(image, model, num_passes=100):
    model.train()

    all_outputs = torch.zeros((num_passes, 10))
    for i in range(num_passes):
        outputs = model(image)
        all_outputs[i] = outputs

    mean_outputs = all_outputs.mean(0)
    std_dev = all_outputs.std(0)

    conf, predicted = torch.max(mean_outputs, 0)
    uncertainty = std_dev[predicted]

    return conf.item(), predicted.item(), uncertainty.item()

# Helper Functions Continued
def federated_average(models_state_dict, weights):
    """
    Implement the FedAvg algorithm.
    models: dict of serialized models
    weights: list of model weights (number of samples each worker trained on)
    """
    total_weight = sum(weights)
    avg_state_dict = CNN().state_dict()

    for key in avg_state_dict.keys():
        avg_state_dict[key] = sum(
            model[key] * weight for model, weight in
            zip(models_state_dict, weights)) / total_weight

    return avg_state_dict


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
    global_model = request.json['global_model']
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
    app.run(host='0.0.0.0', port=8000)
