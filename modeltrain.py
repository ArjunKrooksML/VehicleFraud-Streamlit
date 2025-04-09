import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import time
import copy
from PIL import Image
import io
import numpy as np

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = 'dataset'
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'vehicle_fraud_resnet50_pytorch.pth'
LEARNING_RATE = 0.001
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_datasets(data_dir, batch_size):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation (test) directory not found: {val_dir}")

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    if image_datasets['val'].classes != class_names:
         print(f"Warning: Class names mismatch between train ({class_names}) and val ({image_datasets['val'].classes}) directories.")

    print(f"Found classes: {class_names}")
    print(f"Dataset sizes: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}")
    return dataloaders, dataset_sizes, class_names

def build_model(num_classes):
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    if num_classes == 2:
        model_ft.fc = nn.Linear(num_ftrs, 1) # Output 1 logit for BCELossWithLogitsLoss
    else:
         model_ft.fc = nn.Linear(num_ftrs, num_classes) # For CrossEntropyLoss

    model_ft = model_ft.to(device)
    return model_ft

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, model_save_path, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Convert labels to float and reshape for BCELossWithLogitsLoss
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    labels = labels.float().unsqueeze(1) 

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    if isinstance(criterion, nn.BCEWithLogitsLoss):
                        loss = criterion(outputs, labels)
                        preds = (torch.sigmoid(outputs) > 0.5).float() # Get predictions (0 or 1)
                    else: # Assuming CrossEntropyLoss
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1) # Get predicted class index

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data if isinstance(criterion, nn.BCEWithLogitsLoss) else preds == labels)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item()) # Convert tensor to Python number
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item()) # Convert tensor to Python number

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_save_path)
                print(f'Best model saved to {model_save_path} with accuracy: {best_acc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, (train_loss_history, val_loss_history, train_acc_history, val_acc_history)


def preprocess_single_image(image_path_or_bytes, target_size=224):
    transform = data_transforms['val'] # Use validation transforms for single image
    img = None
    if isinstance(image_path_or_bytes, bytes):
        img = Image.open(io.BytesIO(image_path_or_bytes)).convert('RGB')
    elif isinstance(image_path_or_bytes, str) and os.path.exists(image_path_or_bytes):
         img = Image.open(image_path_or_bytes).convert('RGB')
    else:
        raise ValueError("Input must be a valid file path or image bytes.")

    processed_tensor = transform(img)
    processed_tensor = processed_tensor.unsqueeze(0) # Add batch dimension
    return processed_tensor.to(device)


if __name__ == '__main__':
    print(f"Using device: {device}")

    dataloaders, dataset_sizes, class_names = load_datasets(DATA_DIR, BATCH_SIZE)
    NUM_CLASSES = len(class_names)

    model = build_model(NUM_CLASSES)

    if NUM_CLASSES == 2:
        criterion = nn.BCEWithLogitsLoss() # More stable than Sigmoid + BCELoss
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE) # Only train the final layer

    model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, model_save_path, num_epochs=EPOCHS)

    print(f"\n--- Training complete ---")
    