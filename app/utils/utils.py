import io
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from .datasets import CustomDatasetWithLabelsList
from .network import CNN
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


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


def federated_average(models_state_dict, weights):
    total_weight = sum(weights)
    avg_state_dict = CNN().state_dict()

    for key in avg_state_dict.keys():
        avg_state_dict[key] = sum(
            model[key] * weight for model, weight in
            zip(models_state_dict, weights)) / total_weight

    return avg_state_dict
