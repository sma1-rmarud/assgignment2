import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import detectors
import timm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.201))
])

trainset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

encoder = timm.create_model("resnet50_simclr_cifar10", pretrained=True)
encoder.eval()  # Freeze를 위한 준비
encoder.requires_grad_(False)  # 전체 freeze

# Linear classifier만 학습
class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.num_features, num_classes)
        
    def forward(self, x):
        with torch.no_grad():  # encoder는 forward만 사용
            feats = self.encoder.forward_features(x)
        if feats.ndim == 4:  # (B, C, H, W)
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)  # (B, C)
        return self.fc(feats)

model = LinearClassifier(encoder).cuda()
early_stop_acc = 0.94

optimizer = Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    model.train()
    correct = total = 0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    acc = correct / total
    print(f"[Epoch {epoch+1}] Train Accuracy: {acc*100:.2f}%")

    if acc >= early_stop_acc:
        print(f"Early stopping at epoch {epoch+1}: Accuracy reached {acc*100:.2f}%")
        break
    
torch.save(model.fc.state_dict(), 'linear_fc_simclr.pth')
print("분류기 가중치 저장 완료.")

model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\nFinal Test Accuracy: {correct/total * 100:.2f}%")
