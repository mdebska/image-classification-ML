import timm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from io import StringIO

# import wandb
wandb.init(
    project="cifar10-HP-tuning-final",
    tags=["cifar10", "resnet18", "optimizer", "lr", "batch_size", "dropout"])
config = wandb.config

# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

set_seed(config.seed)

# mean & std
mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

# augmentation
aug = config.augmentation if hasattr(config, "augmentation") else "none"
if config.augmentation == "basic":
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
elif config.augmentation == "color":
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
elif config.augmentation == "affine":
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
else:
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # std of CIFAR-10
])


# Use seeded generator for reproducibility
generator = torch.Generator().manual_seed(config.seed)

# Load cifar10 training set (50,000 images)
train_val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

# Split the training set into 80% train and 20% val 
train_size = int(0.8 * len(train_val_dataset))  # 40,000
val_size = len(train_val_dataset) - train_size   # 10,000
train_set, val_set = random_split(train_val_dataset, [train_size, val_size], generator = generator)

# Apply non-augmented transform to val_set
val_set.dataset.transform = transform  # Override with clean transform

# Load the official MNIST test set (10,000 images)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator = generator)
val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, generator = generator)
test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, generator = generator)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Definition
dropout = config.dropout if hasattr(config, "dropout") else 0.0
model = timm.create_model("resnet18", pretrained=False, num_classes=10, in_chans=3)

# add dropout layer
in_features = model.get_classifier().in_features
model.fc = nn.Sequential(
    nn.Dropout(p=dropout),
    nn.Linear(in_features, 10)
)
model.to(device)

# Optional config fallback
weight_decay = config.weight_decay if hasattr(config, "weight_decay") else 0.0

# Loss
criterion = nn.CrossEntropyLoss()

if config.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
elif config.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=weight_decay)
elif config.optimizer == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
else:
    raise ValueError(f"Unknown optimizer: {config.optimizer}")

# Learning Rate Scheduler

if hasattr(config, "lr_scheduler"):
    if config.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif config.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    elif config.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    else:
        scheduler = None
else:
    scheduler = None

wandb.watch(model)

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# define evaluation function for val and test

def evaluate_model(
    model,
    data_loader,
    criterion,
    class_names,
    mode="val",        # "val" or "test"
    epoch=None,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.eval()
    total_loss, total_correct = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_mat = confusion_matrix(all_labels, all_preds)
    per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)

    # === Classification report ===
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\n {mode.upper()} Classification Report:\n")
    #print(report_text)

    if epoch is not None:
        report_path = f"classification_report_{mode}_epoch{epoch+1}.txt"
    else:
        report_path = f"classification_report_{mode}.txt"

    with open(report_path, "w") as f:
        f.write(report_text)

    # === Confusion matrix plot ===
    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"{mode.capitalize()} Confusion Matrix")
    plt.tight_layout()

    # === Per-class accuracy table ===
    acc_table = wandb.Table(columns=["Class", "Accuracy"])
    for cls, acc in zip(class_names, per_class_acc):
        acc_table.add_data(cls, float(acc))

    # === Classification report as W&B Table with accuracy column ===
    report_df = pd.read_fwf(StringIO(report_text), index_col=0)
    report_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-Score", "Support", "Accuracy"])

    rows_to_log = []
    accuracy_row = None

    for idx, row in report_df.iterrows():
        precision = row.get("precision", None)
        recall = row.get("recall", None)
        f1_ = row.get("f1-score", None)
        support = row.get("support", None)

        try:
            precision = float(precision) if precision != "-" else None
            recall = float(recall) if recall != "-" else None
            f1_ = float(f1_) if f1_ != "-" else None
            support = int(support) if pd.notna(support) else None
        except:
            continue

        if idx in class_names:  # <- fixed condition
            acc = float(per_class_acc[class_names.index(idx)])
            rows_to_log.append([idx, precision, recall, f1_, support, acc])
        elif idx.lower() == "accuracy":
            accuracy_row = [idx, None, None, None, support, accuracy]
        else:
            rows_to_log.append([idx, precision, recall, f1_, support, None])

    for row in rows_to_log:
        report_table.add_data(*row)
    if accuracy_row:
        report_table.add_data(*accuracy_row)

    # preprint updated classification report table with accuracy
    columns = ["Class", "Precision", "Recall", "F1-Score", "Support", "Accuracy"]
    df = pd.DataFrame(rows_to_log + ([accuracy_row] if accuracy_row else []), columns=columns)

    print("\nClassification Report Table (with Accuracy):")
    print(df.to_string(index=False))


    # === W&B logging ===
    log_data = {
        f"{mode}_loss": avg_loss,
        f"{mode}_accuracy": accuracy,
        f"{mode}_precision_macro": precision,
        f"{mode}_recall_macro": recall,
        f"{mode}_f1_score_macro": f1,
        f"{mode}_classification_report_path": report_path
    }

    if epoch is not None:
        log_data["epoch"] = epoch + 1
        log_data[f"{mode}_confusion_matrix_image_epoch_{epoch+1}"] = wandb.Image(fig)
        log_data[f"{mode}_per_class_accuracy_table_epoch_{epoch+1}"] = acc_table
        log_data[f"{mode}_classification_report_table_epoch_{epoch+1}"] = report_table
    else:
        log_data[f"{mode}_confusion_matrix_image"] = wandb.Image(fig)
        log_data[f"{mode}_per_class_accuracy_table"] = acc_table
        log_data[f"{mode}_classification_report_table"] = report_table

    wandb.log(log_data)
    plt.close(fig)

    return avg_loss, accuracy

# Early stopping
best_loss = float('inf')
patience = 5
counter = 0
#min_delta = 1e-3  # adjust as needed


# Training + Evaluation loop
best_val_accuracy = 0.0
for epoch in range(config.max_epochs):
    model.train()
    total_loss, correct = 0.0, 0
    train_preds, train_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

        preds = outputs.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_accuracy = correct / len(train_loader.dataset)
    avg_train_loss = total_loss / len(train_loader)
    train_precision = precision_score(train_labels, train_preds, average='macro')
    train_recall = recall_score(train_labels, train_preds, average='macro')
    train_f1 = f1_score(train_labels, train_preds, average='macro')

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
        f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")
    
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "train_precision_macro": train_precision,
        "train_recall_macro":train_recall,
        "train_f1_score_macro": train_f1
    })

    # validation
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, class_names, mode="val", epoch=epoch)
        # Step the learning rate scheduler (if defined)
    if scheduler:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Optional: log current learning rate to W&B
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}] Learning rate: {current_lr:.6f}")
        wandb.log({"learning_rate": current_lr})

    # save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model_path = "best_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved with val accuracy: {val_accuracy:.4f}")

        # Log to wandb
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    # apply early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


# final test
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
evaluate_model(model, test_loader, criterion, class_names, mode="test", epoch=None)

wandb.finish()