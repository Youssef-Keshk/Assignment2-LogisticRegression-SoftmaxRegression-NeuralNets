import torch
import numpy as np

class LogisticRegression:
    def __init__(self, input_dim):
        self.W_bin = torch.zeros((input_dim, 1), requires_grad=False) * 0.01
        self.b_bin = torch.zeros(1, requires_grad=False)

    def _sigmoid(self, z):
        return torch.sigmoid(z)


    def _predict_binary(self, X):
        z = X @ self.W_bin + self.b_bin
        return self._sigmoid(z)


    def _binary_cross_entropy(self, y_pred, y_true):
        eps = 1e-9
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))


    def _update_weights_binary(self, X, y_true, y_pred, lr=0.01):
        m = X.shape[0]
        error = y_pred - y_true

        dW = (X.t() @ error) / m
        db = torch.mean(error)

        self.W_bin -= lr * dW
        self.b_bin -= lr * db


    def _accuracy_binary(self, pred, labels):
        pred_labels = (pred >= 0.5).float()
        return (pred_labels == labels).float().mean().item()
    

    def train(self, train_loader, val_loader, epochs=20, lr=0.01):

        self.train_losses_bin = []
        self.val_losses_bin = []
        self.train_accs_bin = []
        self.val_accs_bin = []

        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            total_train = 0

            for X_batch, y_batch in train_loader:
                y_batch = y_batch.view(-1, 1)
                preds = self._predict_binary(X_batch)
                loss = self._binary_cross_entropy(preds, y_batch)

                self._update_weights_binary(X_batch, y_batch, preds, lr=lr)

                train_loss += loss.item() * X_batch.size(0)
                train_acc += self._accuracy_binary(preds, y_batch) * X_batch.size(0)
                total_train += X_batch.size(0)

            train_loss /= total_train
            train_acc /= total_train
            val_loss = 0.0
            val_acc = 0.0
            total_val = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_batch = y_batch.view(-1, 1)
                    preds = self._predict_binary(X_batch)
                    loss = self._binary_cross_entropy(preds, y_batch)

                    val_loss += loss.item() * X_batch.size(0)
                    val_acc += self._accuracy_binary(preds, y_batch) * X_batch.size(0)
                    total_val += X_batch.size(0)

            val_loss /= total_val
            val_acc /= total_val

            self.train_losses_bin.append(train_loss)
            self.val_losses_bin.append(val_loss)
            self.train_accs_bin.append(train_acc)
            self.val_accs_bin.append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def evaluate_binary(self, test_loader):
        all_preds = []
        all_labels = []
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_batch = y_batch.view(-1, 1)
                preds = self._predict_binary(X_batch)
                pred_labels = (preds >= 0.5).float()

                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

                test_correct += (pred_labels == y_batch).sum().item()
                test_total += y_batch.size(0)

        test_accuracy = test_correct / test_total
        return test_accuracy, np.array(all_preds), np.array(all_labels)