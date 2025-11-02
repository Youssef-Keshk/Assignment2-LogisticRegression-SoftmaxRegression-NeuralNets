import torch
import torch.nn as nn
import numpy as np


class BinaryLogisticRegression(nn.Module):
    def __init__(self, input_size=784):
        super(BinaryLogisticRegression, self).__init__()
        self.W_bin = nn.Parameter(torch.zeros(input_size, 1))
        self.b_bin = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = x @ self.W_bin + self.b_bin      # linear transformation
        return self.sigmoid(logits)               # sigmoid output

class MulticlassLogisticRegression(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(MulticlassLogisticRegression, self).__init__()
        self.W_multi = nn.Parameter(torch.zeros(input_size, num_classes))
        self.b_multi = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x @ self.W_multi + self.b_multi     # linear transformation
    

def evaluate_model(model, X, y, device, model_type='multiclass'):
    model.eval()
    model = model.to(device)

    # Prepare data
    X_tensor = X.to(device)
    y_tensor = y.to(device)

    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0

    # Use appropriate loss function
    if model_type == 'binary':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]

            outputs = model(batch_X)

            if model_type == 'binary':
                # For binary classification
                predicted = (outputs > 0.5).float().squeeze()
                loss = criterion(outputs.squeeze(), batch_y.float())
            else:
                # For multiclass classification
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = total_loss / total

    return accuracy, avg_loss, np.array(all_preds), np.array(all_labels)
