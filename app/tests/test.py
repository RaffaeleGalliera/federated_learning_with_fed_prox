import requests
import io
import os
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from app.utils.network import CNN
import hashlib
import sys

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

server_url = "http://localhost:8000"
client_1_url = "http://localhost:9000"
client_2_url = "http://localhost:8800"

training_dir = 'dataset/trainingSet/trainingSet'


# Set blank model
def set_blank_model():
    model = CNN()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    serialized_model = buffer.getvalue().hex()
    test_model_setter(serialized_model)


def get_random_image(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    image_files = [f for f in files if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        return None

    random_image_file = random.choice(image_files)

    return os.path.join(folder_path, random_image_file)


def image_to_bytes(image):
    image = Image.open(image)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_data = img_byte_arr.getvalue()
    return img_byte_data


# Test for image inference
def test_infer(folder, number):
    # get a random image from folder
    image = get_random_image(os.path.join(folder, number))
    response = requests.post(f"{server_url}/infer", files={'image': image_to_bytes(image)})
    print("Inference Test:", response.json())
    return response.json()['prediction']


# Test for image inference
def test_infer_with_uncertainty(folder, number):
    # get a random image from folder
    image = get_random_image(os.path.join(folder, number))
    response = requests.post(f"{server_url}/infer_with_uncertainty",
                             files={'image': image_to_bytes(image)})
    print("Inference Test:", response.json())

    prediction = response.json()['prediction']
    uncertainty = response.json()['uncertainty']

    return prediction, uncertainty


def study_uncertainty(folder, number, num_inferences=1000):
    # List to store predictions, uncertainties, and confidences
    predictions, uncertainties, confidences = [], [], []

    for _ in range(num_inferences):
        image = get_random_image(os.path.join(folder, number))
        response = requests.post(f"{server_url}/infer_with_uncertainty",
                                 files={'image': image_to_bytes(image)})
        result = response.json()
        prediction, uncertainty, confidence = result['prediction'], result[
            'uncertainty'], result['confidence']

        predictions.append(prediction)
        uncertainties.append(uncertainty)
        confidences.append(confidence)

    # Determine if predictions are correct
    correct = [int(pred == int(number)) for pred in predictions]

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(len(predictions)):
        if correct[i]:
            plt.scatter(uncertainties[i], confidences[i], color='green')
        else:
            plt.scatter(uncertainties[i], confidences[i], color='red')

    plt.xlabel('Uncertainty')
    plt.ylabel('Confidence')
    plt.title('Uncertainty vs Confidence for Each Inference')
    plt.legend(['Correct', 'Incorrect'])
    plt.show()


# Test for training from a directory and labels list
def test_train_worker_directory_with_labels_list(directory_path, labels_list, url):
    response = requests.post(
        f"{url}/train-worker-on-labels-list",
        json={
            'directory': directory_path,
            'labels_list': labels_list
        }
    )
    print("Train Worker Directory Test:", response.json())
    return response.json()['accuracy']


# Test model setter
def test_model_setter(model, url):
    response = requests.post(f"{url}/set-model", json={'model': model})
    print("Model Setter Test:", response.json())


def test_validate_worker_on_labels_list(labels_list, url):
    response = requests.post(
        f"{url}/validate-worker-on-labels-list",
        json={
            'labels_list': labels_list
        }
    )
    print("Validate Worker on Labels List Test:", response.json())

    return response.json()['accuracy']


def test_fed_prox_train_worker_on_labels_list(labels_list, global_model, url):
    response = requests.post(
        f"{url}/fed-prox-train-worker-on-labels-list",
        json={
            'directory': training_dir,
            'labels_list': labels_list,
        }
    )

    print("Fed Prox Train Worker on Labels List Test:", response.json())
    return response.json()['model']


def test_fed_prox(model=None, url_1=None, url_2=None, url_3=None):
    hex_model_3 = test_fed_prox_train_worker_on_labels_list(
        ['0', '1', '2', '3'],
        model,
        url_2
    )
    print(f'Hex model 3 round {i + 1}')
    print(hashlib.sha3_256(hex_model_3.encode('UTF-8')).hexdigest())

    hex_model_4 = test_fed_prox_train_worker_on_labels_list(
        ['0', '1', '2', '4'],
        model,
        url_3
    )
    print(f'Hex model 4 round {i + 1}')
    print(hashlib.sha3_256(hex_model_4.encode('UTF-8')).hexdigest())

    mock_model_data = {
        'models': [hex_model_3, hex_model_4],
        'weights': [3481, 3258]
    }

    response = requests.post(f"{url_1}/fed-avg", json=mock_model_data)
    print("Federated Proximal Test:", response.json())
    return response.json()['model']


def test_model_getter(url):
    response = requests.get(f"{url}/get-model")
    print("Model Setter Test:", response.json())
    return response.json()['model']


if __name__ == "__main__":
    accuracy = test_train_worker_directory_with_labels_list(
        training_dir,
        ['0', '1', '2'],
        server_url
    )
    assert accuracy > 80

    global_hex_model = test_model_getter(server_url)
    print(f'Global Model')
    print(hashlib.sha3_256(global_hex_model.encode('UTF-8')).hexdigest())
    print(f'Size of model in bytes: {sys.getsizeof(global_hex_model)}')
    test_model_setter(global_hex_model, client_1_url)

    client_accuracy = test_validate_worker_on_labels_list(
        ['0', '1', '2'],
        client_1_url
    )
    assert client_accuracy == accuracy

    test_model_setter(global_hex_model, client_2_url)

    client_accuracy = test_validate_worker_on_labels_list(
        ['0', '1', '2'],
        client_2_url
    )
    assert client_accuracy == accuracy

    accuracy = test_validate_worker_on_labels_list(
        ['3', '4'],
        client_1_url
    )
    assert accuracy == 0

    accuracy = test_validate_worker_on_labels_list(
        ['3', '4'],
        client_2_url
    )
    assert accuracy == 0

    rounds = 3
    for i in range(rounds):
        print(f"Round {i + 1}")

        hex_model = test_fed_prox(
            global_hex_model,
            server_url,
            client_1_url,
            client_2_url
        )
        print(f'Hex model round {i + 1}')
        print(hashlib.sha3_256(hex_model.encode('UTF-8')).hexdigest())
        print(f'Size of model in bytes: {sys.getsizeof(hex_model)}')

        test_model_setter(hex_model, client_1_url)
        accuracy = test_validate_worker_on_labels_list(
            ['0', '1', '2', '3', '4'],
            client_1_url
        )

        test_model_setter(hex_model, client_2_url)
        accuracy = test_validate_worker_on_labels_list(
            ['0', '1', '2', '3', '4'],
            client_2_url
        )

        global_hex_model = hex_model

    assert accuracy > 80

    test_infer('../dataset/evaluation', '3')



