import torch
import numpy as np

class SoftmaxRegression:
    def __init__(self, input_dim, n_classes) -> None:
        self.W_multi = torch.randn((input_dim, n_classes), requires_grad=False) * 0.01
        self.b_multi = torch.zeros(n_classes, requires_grad=False)
        self.n_classes = n_classes

    def _softmax(self, z):
        z_shifted = z - torch.max(z, dim=1, keepdim=True)[0]
        exp_z = torch.exp(z_shifted)
        return exp_z / torch.sum(exp_z, dim=1, keepdim=True)

    def _predict_multiclass(self, X):
        z = X @ self.W_multi + self.b_multi
        return self._softmax(z)

    def _cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-9
        y_pred = torch.clamp(y_pred, eps, 1 - eps)

        batch_size = y_true.size(0)
        y_one_hot = torch.zeros(batch_size, self.n_classes)
        y_one_hot[torch.arange(batch_size), y_true.long()] = 1

        loss = -torch.sum(y_one_hot * torch.log(y_pred)) / batch_size
        return loss

    def _update_weights_multiclass(self, X, y_true, y_pred, lr=0.01):
        m = X.shape[0]

        y_one_hot = torch.zeros(m, self.n_classes)
        y_one_hot[torch.arange(m), y_true.long()] = 1

        error = y_pred - y_one_hot

        dW = (X.t() @ error) / m
        db = torch.mean(error, dim=0)

        self.W_multi -= lr * dW
        self.b_multi -= lr * db

    def _accuracy_multiclass(self, pred, labels):
        pred_labels = torch.argmax(pred, dim=1)
        return (pred_labels == labels).float().mean().item()
    
    def train(self, train_loader, val_loader, epochs=20, lr=0.01):
        self.train_losses_multi = []
        self.val_losses_multi = []
        self.train_accs_multi = []
        self.val_accs_multi = []

        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            total_train = 0

            for X_batch, y_batch in train_loader:
                preds = self._predict_multiclass(X_batch)
                loss = self._cross_entropy_loss(preds, y_batch)

                self._update_weights_multiclass(X_batch, y_batch, preds, lr=lr)

                train_loss += loss.item() * X_batch.size(0)
                train_acc += self._accuracy_multiclass(preds, y_batch) * X_batch.size(0)
                total_train += X_batch.size(0)

            train_loss /= total_train
            train_acc /= total_train

            val_loss = 0.0
            val_acc = 0.0
            total_val = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    preds = self._predict_multiclass(X_batch)
                    loss = self._cross_entropy_loss(preds, y_batch)

                    val_loss += loss.item() * X_batch.size(0)
                    val_acc += self._accuracy_multiclass(preds, y_batch) * X_batch.size(0)
                    total_val += X_batch.size(0)

            val_loss /= total_val
            val_acc /= total_val

            self.train_losses_multi.append(train_loss)
            self.val_losses_multi.append(val_loss)
            self.train_accs_multi.append(train_acc)
            self.val_accs_multi.append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                

    def evaluate_multiclass(self, test_loader):
        all_preds = []
        all_labels = []
        test_correct = 0
        test_total = 0
        per_class_correct = torch.zeros(self.n_classes)
        per_class_total = torch.zeros(self.n_classes)

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds = self._predict_multiclass(X_batch)
                pred_labels = torch.argmax(preds, dim=1)

                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

                test_correct += (pred_labels == y_batch).sum().item()
                test_total += y_batch.size(0)

                for i in range(self.n_classes):
                    mask = y_batch == i
                    per_class_correct[i] += (pred_labels[mask] == y_batch[mask]).sum().item()
                    per_class_total[i] += mask.sum().item()

        test_accuracy = test_correct / test_total
        per_class_accuracy = per_class_correct / per_class_total
        return test_accuracy, np.array(all_preds), np.array(all_labels), per_class_accuracy