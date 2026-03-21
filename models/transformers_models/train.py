import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Device
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1).cuda() + 1
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")

DEVICE = _get_device()
print(f"[transformers_models] Using device: {DEVICE}")


def _is_bert_model(model: nn.Module) -> bool:
    """Verifica se o modelo é um BERTClassifier (recebe dicts em vez de tensors)."""
    return hasattr(model, "bert")


# Treino

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module = None,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Treino com early stopping e learning rate scheduler.
    Suporta tanto TransformerClassifier (input: tensor) como BERTClassifier (input: dict).
    """

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    is_bert = _is_bert_model(model)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state_dict = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # Training step
        model.train()
        for batch in train_loader:
            if is_bert:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                y_batch = batch["label"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(x_batch)

            optimizer.zero_grad()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluation
        train_loss, train_acc = evaluate(model, train_loader, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
            )

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(
                        f"\n[Early Stopping] Stopped at epoch {epoch}. "
                        f"Best val_loss: {best_val_loss:.4f}"
                    )
                break

    # Restaurar os melhores pesos
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(DEVICE)

    return history


# Avaliação

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module = None,
) -> tuple[float, float]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    is_bert = _is_bert_model(model)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if is_bert:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                y_batch = batch["label"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(x_batch)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Prediction

def predict(
    model: nn.Module,
    loader: DataLoader,
) -> torch.Tensor:

    is_bert = _is_bert_model(model)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            if is_bert:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                x_batch, _ = batch
                x_batch = x_batch.to(DEVICE)
                outputs = model(x_batch)

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds)
