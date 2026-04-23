# Custom Vehicle Detection CNN
# Author: Zaamin Qadeer W1906890

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image

class CustomVehicleCNN(nn.Module):
    #custom cnn architecture for vehicle detection
    #uses resnet50 backbone with custom detection head
    
    def __init__(self, num_classes=4, pretrained=True):
        super(CustomVehicleCNN, self).__init__()
        
        #load resnet50 as feature extractor backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        #remove final fc layer, keep conv layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        #custom detection head
        self.detection_head = nn.Sequential(
            #reduce channels while maintaining spatial info
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            #further reduction
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            #detection specific layers
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            #final prediction layer - simplified output
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        #classification head
        self.classifier = nn.Linear(128, num_classes)
        
        #freeze early resnet layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
    
    def forward(self, x):
        #extract features
        features = self.backbone(x)
        
        #detection head
        x = self.detection_head(features)
        x = x.view(x.size(0), -1)
        
        #classification
        x = self.classifier(x)
        
        return x


class SimpleImageDataset(Dataset):
    #loads images and labels from coco128 format
    
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        
        #get all jpg images
        self.image_files = list(self.image_dir.glob('*.jpg'))
        print(f"Found {len(self.image_files)} images")
        
        #vehicle classes in coco: car=2, motorcycle=3, bus=5, truck=7
        #map to 0-3 for our 4 classes
        self.class_map = {2: 0, 3: 1, 5: 2, 7: 3}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        #load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        #load label from corresponding txt file
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        #default to class 0 (car) if no label or no vehicle
        label = 0
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                #yolo format: class_id x_center y_center width height
                #take first vehicle found in image
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        #if it's a vehicle class, use it
                        if class_id in self.class_map:
                            label = self.class_map[class_id]
                            break
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SimplifiedCNNTrainer:
    #simplified trainer for quick validation
    
    def __init__(self, model, device='cuda', output_dir='custom_cnn_output'):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        #optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001
        )
        
        #loss function
        self.criterion = nn.CrossEntropyLoss()
        
        #lr scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.5
        )
        
        #tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_acc = 0.0
    
    def train_epoch(self, train_loader):
        #train one epoch
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            #forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            #backward
            loss.backward()
            self.optimizer.step()
            
            #stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        #validate
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=30):
        #full training
        print(f"\nCustom CNN Training Started")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}\n")
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            #lr step
            self.scheduler.step()
            
            #save best
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint('best_custom_model.pth')
                print(f"New best model saved")
            
            print()
        
        print(f"Training complete. Best accuracy: {self.best_acc:.2f}%")
        self.plot_training_curves()
        self.save_stats()
    
    def save_checkpoint(self, filename):
        #save model
        filepath = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }, filepath)
    
    def plot_training_curves(self):
        #plot loss and accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        #loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        #accuracy
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300)
        print(f"Training curves saved to {self.output_dir / 'training_curves.png'}")
    
    def save_stats(self):
        #save statistics
        stats = {
            'best_accuracy': float(self.best_acc),
            'final_train_loss': float(self.train_losses[-1]),
            'final_val_loss': float(self.val_losses[-1]),
            'final_train_acc': float(self.train_accs[-1]),
            'final_val_acc': float(self.val_accs[-1]),
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'train_accs': [float(x) for x in self.train_accs],
            'val_accs': [float(x) for x in self.val_accs]
        }
        
        with open(self.output_dir / 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Stats saved to {self.output_dir / 'training_stats.json'}")


def download_coco128():
    #downloads coco128 if needed
    import urllib.request
    import zipfile
    
    print("Checking for COCO128 dataset")
    
    if not Path('coco128').exists():
        print("Downloading COCO128")
        url = "https://ultralytics.com/assets/coco128.zip"
        urllib.request.urlretrieve(url, 'coco128.zip')
        
        print("Extracting")
        with zipfile.ZipFile('coco128.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        
        print("COCO128 ready")
    else:
        print("COCO128 found")


if __name__ == "__main__":
    # checks device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    if device == 'cpu':
        print("Training on CPU will be slower than GPU")
        epochs = 10  # reduce epochs for cpu
        batch_size = 8
    else:
        epochs = 30
        batch_size = 16
    
    # data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    #auto download coco128 if not in folder
    download_coco128()
    
    print("Creating datasets")
    
    try:
        #create dataset from images folder with real labels
        full_dataset = SimpleImageDataset(
            image_dir='coco128/images/train2017',
            label_dir='coco128/labels/train2017',
            transform=train_transform
        )
        
        #split for train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        #dataloaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"Train samples: {len(train_subset)}")
        print(f"Val samples: {len(val_subset)}\n")
        
        # create model
        print("Building custom CNN")
        model = CustomVehicleCNN(num_classes=4, pretrained=True)
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        #train
        trainer = SimplifiedCNNTrainer(model, device=device)
        trainer.train(train_loader, val_loader, epochs=epochs)
        
        print("Custom CNN training complete")
        print("This is a custom architecture designed for vehicle detection")
        
    except FileNotFoundError:
        print("ERROR: COCO128 dataset not found")
        print("To fix:")
        print("1. Download: https://ultralytics.com/assets/coco128.zip")
        print("2. Extract to current directory")
        print("3. Run this script again")