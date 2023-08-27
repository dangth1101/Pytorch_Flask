import os
from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
import io as io_module
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.dpn import DPN92
from models.mobilenetv2 import MobileNetV2
from models.resnet import ResNet50
from models.vgg import VGG
from torchvision.transforms.functional import to_tensor
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor



app = Flask(__name__)

# Set up
MODELS = ["vgg16", "mobilenetv2", "dpn92", "resnet50"]

# Load the pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = (device == 'cuda')

# Define the list of CIFAR-10 classes
classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')


@app.route('/')
def index():
    return render_template('index.html', active_page='home', models=MODELS)

@app.route('/predict')
def predict():
    return render_template('predict.html', active_page='predict', models=MODELS)

@app.route('/predict_result', methods=['POST']) 
def predict_result():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    model = init_model(request.form['selectedModel'])
    image = Image.open(io_module.BytesIO(file.read())).convert("RGB")
    class_probabilities = predict_image(image, model)

    return jsonify(class_probabilities)

@app.route('/check')
def check():
    return render_template('check.html', active_page='check')

@app.route('/check_result', methods=['POST'])
def check_result():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'No file part'})

    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No selected file'})
    

    image1 = Image.open(io_module.BytesIO(file1.read()))
    image2 = Image.open(io_module.BytesIO(file2.read()))

    desired_size = (256, 256)
    image1 = image1.resize(desired_size)
    image2 = image2.resize(desired_size)

    image1_tensor = to_tensor(image1).unsqueeze(0)
    image2_tensor = to_tensor(image2).unsqueeze(0)

    image1_tensor = image1_tensor.type(torch.float32) / 255.0
    image2_tensor = image2_tensor.type(torch.float32) / 255.0

    ssim_score = float(1 - F.mse_loss(image1_tensor, image2_tensor).cpu().numpy())
    l1 = float(F.l1_loss(image1_tensor,image2_tensor))
    l0 = float(torch.norm(abs(image1_tensor - image2_tensor)))

    mse = F.mse_loss(image1_tensor, image2_tensor)
    epsilon = 1e-10
    psnr = 10 * torch.log10(1.0 / (torch.sqrt(mse) + epsilon))
    l2 = float(mse.cpu().numpy())  


    return jsonify({
        'ssim': ssim_score,
        'psnr': psnr.item(),
        'l0': l0,
        'l1': l1,
        'l2': l2,
    })
    
    

def init_model(name):
    model_path = os.path.dirname(__file__)

    if name == "vgg16":
        model_path += '\pretrained_models\VGG16_ckpt.pth'
        module = VGG('VGG16')
    elif name == "mobilenetv2":
        model_path += '\pretrained_models\MobileNetV2_ckpt.pth'
        module = MobileNetV2()
    elif name == "dpn92":
        model_path += '\pretrained_models\DPN92_ckpt.pth'
        module = DPN92()
    elif name == "resnet50":
        model_path += '\pretrained_models\\resnet50_ckpt.pth'
        module = ResNet50()
    else:
        raise ValueError(f"Unsupported model name: {name}")


    model = torch.nn.DataParallel(module)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['net'])
    model.eval()

    return model

def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    probabilities = nn.Softmax(dim=1)(outputs).squeeze().tolist()  # Compute the softmax probabilities and convert to list

    # Create a dictionary of class names and their probabilities
    class_probabilities = {classes[i]: prob * 100 for i, prob in enumerate(probabilities)}

    return class_probabilities

if __name__ == "__main__":
    app.run(debug=True)
