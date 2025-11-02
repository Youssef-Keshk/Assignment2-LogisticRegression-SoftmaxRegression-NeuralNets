import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import os, time, csv
from typing import List, Type, Optional

class CustomNeuralNet(nn.Module):
    def __init__(self, sizes: List[int], activation: Type[nn.Module] = nn.Tanh, weight_init: str = "xavier"):
        """
        sizes: list like [input_dim, hidden1, hidden2, ..., output_dim] (minimum 3 layers)
        activation: activation class for hidden layers (default ReLU)
        weight_init: "xavier" or "he" (defualt xavier)
        """

        super().__init__()
        assert len(sizes) >= 3
        self.sizes = sizes
        self.activation_cls = activation
        self.weight_init = weight_init.lower()
        layers = []
        for i in range(1, len(sizes)):
            # Linearity
            layers.append(nn.Linear(sizes[i-1], sizes[i]))
            # Activation
            if i < len(sizes) - 1:
                layers.append(activation())
            self.net = nn.Sequential(*layers)
            self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                if self.weight_init == "he":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> torch.Tensor:
        """
        x: input x of shape (batch_size, input_dim).
        Returns logits of shape (batch_size, output_dim).
        """
        return self.net(x)
    

def create_model_folder(folder=".", base="model"):
    folder = os.path.join(folder, f"{base}1")
    i = 1
    while os.path.exists(folder):
        i += 1
        folder = f"{base}{i}"
    os.makedirs(folder)
    return folder


def setup_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilizing device: {device}")
    return device


def split_dataset(X, y, val_fraction, batch, seed):
    dataset = TensorDataset(X, y)
    n_total = len(dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, valid_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False)
    return train_loader, val_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    batch_losses = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)               # logits of this batch
            loss = criterion(logits, yb)     # compute loss

            batch_losses.append(loss.item())
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

    train_loss = total_loss / total_samples if total_samples > 0 else float("nan")
    train_acc = total_correct / total_samples if total_samples > 0 else float("nan")
    train_std = torch.tensor(batch_losses).std().item() 

    return train_loss, train_acc, train_std


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    batch_losses = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()            # clear old gradients
        logits = model(xb)               # logits of this batch
        loss = criterion(logits, yb)     # compute loss
        loss.backward()                  # compute gradients
        optimizer.step()                 # update weights

        batch_losses.append(loss.item())
        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += xb.size(0)
    
    train_loss = total_loss / total_samples if total_samples > 0 else float("nan")
    train_acc = total_correct / total_samples if total_samples > 0 else float("nan")
    train_std = torch.tensor(batch_losses).std().item() 

    return train_loss, train_acc, train_std


def train_model(model: nn.Module,
                X: torch.Tensor,
                y: torch.Tensor,
                epochs: int = 30,
                batch: int = 64,
                lr: float = 0.01,
                val_fraction: float = 0.2,
                tolerance: float = 10e-6,
                patience: int = 5,
                destination: str = ".",
                seed: int = 42):
    
    model_dir = create_model_folder(folder=destination, base="model")
    os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)

    # Create cvs file to log training process data
    csv_path = os.path.join(model_dir, "results.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_std", "val_std"])

    # GPU acceleration
    device = setup_device()
    model = model.to(device)

    # Data split and division into batches
    train_loader, val_loader = split_dataset(X, y, val_fraction, batch, seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()

    try:
        for epoch in range(1, epochs+1):
            # Training phase
            train_loss, train_acc, train_std = train_epoch(model, train_loader, criterion, optimizer, device)

            # Validation phase
            val_loss, val_acc, val_std = evaluate(model, val_loader, criterion, device)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} | "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
                f"time: {elapsed:.1f}s")
            
            # Save epoch data
            with open(os.path.join(model_dir, "results.csv"), mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, train_std, val_std])
            
            if not (val_loss != val_loss):
                if val_loss < best_val_loss - tolerance:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_loss': val_loss
                    }, os.path.join(model_dir, "weights", "best.pt"))
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("\n" + "─" * 60)
                        print(f"   ⚠ No improvement for {epochs_no_improve} epochs (Interupt to exit safely)")
                        print("─" * 60 + "\n")
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
        
    finally:
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_loss
        }, os.path.join(model_dir, "weights", "last.pt"))
        print(f"Training completed. Saved best.pt and last.pt in {model_dir}")




        
