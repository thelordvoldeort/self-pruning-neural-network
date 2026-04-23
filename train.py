import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Net
from utils import sparsity_loss, calculate_sparsity

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Strong lambda values (IMPORTANT)
lambda_values = [1e-3, 1e-2, 1e-1]

results = []

for lambda_val in lambda_values:
    print(f"\n🔥 Training with lambda = {lambda_val}\n")

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # More epochs for learning
    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = ce_loss + lambda_val * sp_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    sparsity = calculate_sparsity(model)

    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"🔥 Sparsity: {sparsity:.2f}%")

    results.append((lambda_val, accuracy, sparsity))

    # Plot gates
    all_gates = []

    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title(f"Gate Distribution (lambda={lambda_val})")
    plt.savefig(f"gate_distribution_{lambda_val}.png")
    plt.close()

# Final results
print("\n📊 FINAL RESULTS:")
for res in results:
    print(f"Lambda: {res[0]} | Accuracy: {res[1]:.2f}% | Sparsity: {res[2]:.2f}%")