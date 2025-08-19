import os
import joblib
import numpy as np
import json
import random
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, 
    confusion_matrix, classification_report, precision_score, recall_score, f1_score
)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import argparse
import yaml



def loading_pkl_file(file_pkl):
    protein_uid_list = []
    data = joblib.load(file_pkl)
    protein_uid_embedding = {}
    for protein_uid,embedding in data:
        protein_uid_list.append(protein_uid)
        protein_uid_embedding[protein_uid] = embedding


    org_list = {}
    orgs = []
    for protein_uid in protein_uid_list:

        org,rank = protein_uid.split(f'_')
        orgs.append(org)
        if org not in org_list:
            org_list[org] = []

        org_list[org].append(protein_uid)
        
    return orgs,org_list,protein_uid_embedding
    

def sample_num_seq_from_pkl(orgs,org_list,protein_uid_embedding,sample_num=50):

    sample_org_protein = {}
    for org in orgs:
        try:
            sample_org_protein[org] = random.sample(org_list[org],min(sample_num,len(org_list[org])))
        except:
            pass
        
    print(f'samplling number : {len(sample_org_protein[org])}')
        
    org_embedding = {}
    for org,protein_list in sample_org_protein.items():
        org_embedding[org] = []
        for pro in protein_list:
            org_embedding[org].append(protein_uid_embedding[pro])
        org_embedding[org]  = np.mean(org_embedding[org],axis=0)
        
    return org_embedding


def obtain_X_y(org_embedding,cls_org):
    
    cls_org = json.load(open(cls_org))

    org_embedding_cls = {}
    for org,embedding in org_embedding.items():
        for cls,org_list in cls_org.items():
            if org in org_list:
                org_embedding_cls[org] = [(embedding,cls)]


    X = []
    y = []

    for org,embedding_cls in org_embedding_cls.items():

        embedding,cls = embedding_cls[0]


        if cls == f'PAc':
            cls_num = 0
        elif cls == f'PAd':
            cls_num = 2
        elif cls == f'NPA':
            cls_num = 1

        if cls_num == 2:
            continue
        
        X.append(embedding)
        y.append(cls_num)

    X = np.array(X)
    y = np.array(y)
    
    
    return X,y


 
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.5):
        super(TwoLayerNN, self).__init__()
        # Add a deeper network with batch normalization
        hidden_size = int(input_size // 2)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization for the first layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)  # Batch normalization for the second layer
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # Using ReLU with batch normalization
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))  # Using ReLU with batch normalization
        x = self.dropout(x)
        return self.fc3(x)


def prepare_data(X, y, n_test, m_val, device):
    # Split data by class
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    if len(class_0) < (n_test + m_val) // 2 or len(class_1) < (n_test + m_val) // 2:
        raise ValueError("Not enough samples in each class.")
    
    # Test set
    X_test = np.vstack((class_0[:n_test//2], class_1[:n_test//2]))
    y_test = np.concatenate((y[y==0][:n_test//2], y[y==1][:n_test//2]))
    
    # Validation set
    X_val = np.vstack((class_0[n_test//2:n_test//2 + m_val//2], class_1[n_test//2:n_test//2 + m_val//2]))
    y_val = np.concatenate((y[y==0][n_test//2:n_test//2 + m_val//2], y[y==1][n_test//2:n_test//2 + m_val//2]))
    
    # Training set
    X_train = np.vstack((class_0[n_test//2 + m_val//2:], class_1[n_test//2 + m_val//2:]))
    y_train = np.concatenate((y[y==0][n_test//2 + m_val//2:], y[y==1][n_test//2 + m_val//2:]))
    
    # Shuffle datasets
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    

    print("\nDataset Summary:")
    print(f"{'Set':<10}{'Total':<10}{'Class 0':<10}{'Class 1':<10}")
    print("-" * 40)
    for name, labels in [
        ("Train", y_train),
        ("Val",   y_val),
        ("Test",  y_test),
    ]:
        c0 = np.sum(labels == 0)
        c1 = np.sum(labels == 1)
        total = len(labels)
        print(f"{name:<10}{total:<10}{c0:<10}{c1:<10}")
    print('\n')
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor



def initialize_model(input_size, output_size, y_train_tensor, device, learning_rate=1e-6):

    model = TwoLayerNN(input_size, output_size).to(device)
    
    # Compute class weights
    class_counts = np.bincount(y_train_tensor.cpu().numpy())
    class_weights = torch.tensor([1 / count for count in class_counts], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer


def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, early_stopping_patience=5000):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        
        # Metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            y_val_np = y_val.cpu().numpy()
            val_pred_binary = (F.softmax(val_outputs, dim=1)[:, 1].cpu().numpy() >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val_np, val_pred_binary).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else None
            print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Specificity: {specificity:.4f}',end='\r')
    
    return model, train_losses, val_losses


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        probs = F.softmax(test_outputs, dim=1)
        preds = torch.argmax(test_outputs, dim=1)
    
    y_true = y_test.cpu().numpy()
    y_pred = preds.cpu().numpy()
    probs_np = probs.cpu().numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, probs_np[:, 1])
    
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, probs_np[:, 1])
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label=f'Positive Class AUC={auc:.4f}')
    plt.plot([0,1],[0,1],'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.show()
    
    return auc, fpr, tpr


def get_model_eval(X, y, num_epochs, n_test, m_val, learning_rate=1e-6, early_stopping_patience=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(X, y, n_test, m_val, device)
    
    input_size = X_train.shape[1]
    output_size = len(np.unique(y))
    model, criterion, optimizer = initialize_model(input_size, output_size, y_train, device, learning_rate)
    
    model, train_losses, val_losses = train_model(
        model, criterion, optimizer, 
        X_train, y_train, X_val, y_val, 
        num_epochs, early_stopping_patience
    )
    
    auc, fpr, tpr = evaluate_model(model, X_test, y_test)
    
    return auc, fpr, tpr, model




def main(config):
    file_pkl = config["file_pkl"]
    cls_org = config["cls_org"]
    num_list = config["sample_num_list"]
    model_output = config["model_output"]
    results_file = config.get("results_file", "model_results.json")
    roc_curve_plot = config.get("roc_curve_plot", "roc_curve_plot_with_auc.png")
    training_cfg = config.get("training", {})
    num_epochs = training_cfg.get("num_epochs", 30000)
    learning_rate = float(training_cfg.get("learning_rate", 1e-6))
    early_stopping_patience = training_cfg.get("early_stopping_patience", 5000)

    os.makedirs(model_output, exist_ok=True)

    # Load data
    orgs, org_list, protein_uid_embedding = loading_pkl_file(file_pkl)

    # Initialize lists to store ROC curve data and AUC scores
    fpr_all, tpr_all, labels, auc_scores = [], [], [], []
    results = {}

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)

    for num in num_list:
        org_embedding = sample_num_seq_from_pkl(orgs, org_list, protein_uid_embedding, sample_num=num)
        X, y = obtain_X_y(org_embedding, cls_org)

        # Shuffle data
        X_resampled, y_resampled = shuffle(X, y, random_state=42)

        # Compute evaluation sample number
        t_num = int(len(X_resampled[y_resampled == 0]) * 0.1)

        pre_auc = 0
        if results and str(num) in results:
            pre_auc = results[str(num)]["auc_pos"]
            print(f"[INFO] Pre-trained AUC for num={num}: {pre_auc:.3f}")

        # Train model
        auc_pos, fpr_pos, tpr_pos, model = auc_pos, fpr_pos, tpr_pos, model = get_model_eval(
                X_resampled, y_resampled,
                num_epochs, t_num, t_num,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience
            )

        if pre_auc > auc_pos:
            print(f"[INFO] Previous model better for num={num}, keeping old results.")
            auc_pos = results[str(num)]["auc_pos"]
            fpr_pos = results[str(num)]["fpr_pos"]
            tpr_pos = results[str(num)]["tpr_pos"]
            model_path = results[str(num)]["model"]
        else:
            model_path = os.path.join(model_output, f"best_model_{num}.pth")
            torch.save(model.state_dict(), model_path)
            results[str(num)] = {
                "auc_pos": auc_pos,
                "fpr_pos": list(fpr_pos),
                "tpr_pos": list(tpr_pos),
                "model": model_path,
            }

        # Store ROC curve data
        fpr_all.append(fpr_pos)
        tpr_all.append(tpr_pos)
        labels.append(f"num={num}")
        auc_scores.append(auc_pos)

    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    for fpr, tpr, label, auc_score in zip(fpr_all, tpr_all, labels, auc_scores):
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.title("ROC Curves for Different num Values")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(roc_curve_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Training finished. Results saved to {results_file}, ROC plot saved to {roc_curve_plot}")


    # 打印所有评估结果表格
    print("\nEvaluation Results (AUC)")
    print("=" * 40)
    print(f"{'sample num':<10}{'AUC':<10}")
    print("-" * 40)
    for num, auc_score in zip(num_list, auc_scores):
        print(f"{num:<10}{auc_score:<10.3f}")
    print("=" * 40)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models with different sample numbers")
    parser.add_argument("-c", "--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)