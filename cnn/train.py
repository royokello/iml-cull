import argparse
import torch
import torch.nn as nn
from dataset import get_loaders
from model import create_model
import os
import json
import csv


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            preds = model(x.to(device)).argmax(1)
            correct += (preds == y.to(device)).sum().item()
    return correct / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Root directory containing the source folder with images')
    p.add_argument('--source', default='src', help='Name of the source directory containing images (default: src)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--loss_window', type=int, default=3, help='Number of epochs to analyze loss trend')
    args = p.parse_args()

    # Define paths
    model_path = os.path.join(args.input, 'cull_model.pth')
    info_path = os.path.join(args.input, 'cull_model.json')
    log_path = os.path.join(args.input, 'cull_epoch_log.csv')

    # Remove existing files if they exist
    for file_path in [model_path, info_path, log_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed existing file: {file_path}")

    train_loader, val_loader = get_loaders(os.path.join(args.input, 'cull_labels.csv'), 
                                         os.path.join(args.input, args.source), 
                                         args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1
    loss_history = []
    
    # Reset or create the log file with headers
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_accuracy'])

    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = eval_model(model, val_loader, device)
        loss_history.append(train_loss)
        print(f"Epoch {epoch}: Train Loss={train_loss:.8f}, Val Acc={val_acc:.8f}")

        # Log metrics to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_acc:.8f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(model.state_dict(), model_path)
                    
            model_info = {
                'base_model': model.__class__.__name__,
                'epochs': epoch,
                'val_accuracy': val_acc,
                'train_loss': train_loss,
                'learning_rate': args.lr,
                'batch_size': args.batch_size
            }
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            print(f"Saved model with train_loss={train_loss:.8f}, val_acc={val_acc:.8f}")

        # Suggested replacement
        if len(loss_history) >= args.loss_window:
            recent = loss_history[-args.loss_window:]
            diff = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            avg_change = sum(diff) / len(diff)
            mean_recent = sum(recent) / len(recent)
            
            # δ = minimum relative improvement we still care about
            delta = 1e-3
            
            if avg_change / mean_recent > -delta:

                # Save model if current validation accuracy is at least as good as best
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    
                    model_info = {
                        'base_model': model.__class__.__name__,
                        'epochs': epoch,
                        'val_accuracy': val_acc,
                        'train_loss': train_loss,
                        'learning_rate': args.lr,
                        'batch_size': args.batch_size
                    }
                    with open(info_path, 'w') as f:
                        json.dump(model_info, f, indent=4)
                    print(f"Saved final model with train_loss={train_loss:.8f}, val_acc={val_acc:.8f}")
                
                print(f"\nStopping training early at epoch {epoch} due to loss plateau")
                break

if __name__ == '__main__':
    main()