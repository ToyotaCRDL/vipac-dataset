"""Train RSS-CNN model on VIPAC pairwise comparison data."""

import argparse
import joblib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from vipac_analysis.config import KWORDS
from vipac_analysis.dataset import PairedComparisonDataset
from vipac_analysis.model import RSSCNN, ranking_loss
from vipac_analysis.transforms import train_transform


def train_one_repetition(train_loader, val_loader, test_loader,
                         model, optimizer, criterion, lmd, device,
                         nepoch=20, patience=3):
    """Train for up to nepoch with early stopping on validation loss."""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    tr_losses, tr_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(nepoch):
        # Train
        model.train()
        epoch_losses = []
        for left_imgs, right_imgs, labels, *_ in train_loader:
            left_imgs, right_imgs, labels = (
                left_imgs.to(device), right_imgs.to(device), labels.to(device))

            optimizer.zero_grad()
            outputs, outputs_lr, outputs_rr = model(left_imgs, right_imgs)
            loss = criterion(outputs, labels)
            loss_r = ranking_loss(outputs_lr, outputs_rr, labels, lmd, device)
            (loss + loss_r).backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            sys.stderr.write(
                f'\r  train iter {train_loader.__len__()} '
                f'loss {loss.item():.3f} loss_r {loss_r.item():.3f}')
            sys.stderr.flush()

        avg_tr_loss = sum(epoch_losses) / len(epoch_losses)
        tr_losses.append(avg_tr_loss)

        # Validate
        model.eval()
        val_loss, val_loss_r, correct, total = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for left_imgs, right_imgs, labels, *_ in val_loader:
                left_imgs, right_imgs, labels = (
                    left_imgs.to(device), right_imgs.to(device), labels.to(device))

                outputs, outputs_lr, outputs_rr = model(left_imgs, right_imgs)
                val_loss += criterion(outputs, labels).item()
                val_loss_r += ranking_loss(
                    outputs_lr, outputs_rr, labels, lmd, device).item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        n_val = len(val_loader)
        val_loss /= n_val
        val_loss_r /= n_val
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'(Epoch {epoch + 1}/{nepoch}) | '
              f'Train loss {avg_tr_loss:.4f} | '
              f'Val loss {val_loss:.4f} rank {val_loss_r:.4f} | '
              f'Val acc {val_acc:.4%}')

        # Early stopping: save best model
        total_val = val_loss + val_loss_r
        if total_val < best_val_loss:
            best_val_loss = total_val
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # Restore best model
    model.load_state_dict(best_state)
    return model, tr_losses, tr_accs, val_losses, val_accs


def evaluate(test_loader, model, criterion, lmd, device):
    """Run test set evaluation."""
    model.eval()
    test_loss, test_loss_r, correct, correct_r, total = 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for left_imgs, right_imgs, labels, *_ in test_loader:
            left_imgs, right_imgs, labels = (
                left_imgs.to(device), right_imgs.to(device), labels.to(device))

            outputs, outputs_lr, outputs_rr = model(left_imgs, right_imgs)
            test_loss += criterion(outputs, labels).item()
            test_loss_r += ranking_loss(
                outputs_lr, outputs_rr, labels, lmd, device).item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Ranking accuracy
            direction = (0.5 - labels.float()) * 2
            diff = (outputs_lr - outputs_rr).squeeze(-1)
            correct_r += sum(torch.max(
                torch.tensor([0], dtype=torch.float32, device=device),
                diff * direction) > 0).item()

    n_test = len(test_loader)
    return test_loss / n_test, test_loss_r / n_test, correct / total, correct_r / total


def main():
    parser = argparse.ArgumentParser(description='Train RSS-CNN on VIPAC data')
    parser.add_argument('--country', required=True, choices=['us', 'jp'],
                        help='Country dataset')
    parser.add_argument('--attribute', required=True, choices=KWORDS,
                        help='Attribute to train')
    parser.add_argument('--responses', required=True,
                        help='Path to pairwise-responses-{us|jp}.csv')
    parser.add_argument('--images', required=True,
                        help='Path to VIPAC/images/ directory')
    parser.add_argument('--output', required=True,
                        help='Output directory for model weights')
    parser.add_argument('--repetitions', type=int, default=10,
                        help='Number of training repetitions')
    parser.add_argument('--fdb', default=None,
                        help='Path to FractalDB pre-trained weights')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lmd', type=float, default=0.01,
                        help='Ranking loss coefficient')
    parser.add_argument('--cnn', default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--activation', default='sigmoid',
                        choices=['softrelu', 'sigmoid'])
    parser.add_argument('--catdim', type=int, default=2,
                        choices=[1, 2])
    parser.add_argument('--ncorrect-thre', type=int, default=0,
                        help='Dummy-question correctness threshold '
                        '(0=disable filtering)')
    parser.add_argument('--participants', default=None,
                        help='Path to participants CSV for age filtering '
                        '(default: disabled)')
    parser.add_argument('--age-min', type=int, default=0,
                        help='Minimum participant age (inclusive)')
    parser.add_argument('--age-max', type=int, default=130,
                        help='Maximum participant age (inclusive)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Dataset
    print(f'Loading dataset: {args.responses} ({args.attribute})')
    dataset = PairedComparisonDataset(
        args.responses, args.images, attribute=args.attribute,
        transform=train_transform, ncorrect_thre=args.ncorrect_thre,
        participants_path=args.participants,
        age_min=args.age_min, age_max=args.age_max)
    print(f'Dataset size: {len(dataset)} pairs')

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=2, shuffle=False)

    # Train repetitions
    test_accs = []
    test_accs_r = []
    min_val_losses = []

    for rep in range(args.repetitions):
        torch.manual_seed(args.seed + rep)
        print(f'\n=== Repetition {rep} ===')

        model = RSSCNN(
            cnn_name=args.cnn, fdb_path=args.fdb, gap=True,
            activation=args.activation, catdim=args.catdim)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model, tr_losses, tr_accs, val_losses, val_accs = train_one_repetition(
            train_loader, val_loader, test_loader,
            model, optimizer, criterion, args.lmd, device)

        # Save model
        model_fn = os.path.join(
            args.output,
            f'rsscnn_{args.cnn}_gap_{args.attribute}_r{rep}.pth')
        torch.save(model.state_dict(), model_fn)

        # Test evaluation
        test_loss, test_loss_r, test_acc, test_acc_r = evaluate(
            test_loader, model, criterion, args.lmd, device)
        test_accs.append(test_acc)
        test_accs_r.append(test_acc_r)
        if val_losses:
            min_val_losses.append(min(val_losses))

        print(f'Test acc {test_acc:.4%}, loss {test_loss:.4f}, rank {test_loss_r:.4f}')

        # Save metadata
        vard_fn = model_fn.replace('.pth', '.joblib')
        vard = {
            'args': vars(args),
            'repetition': rep,
            'train_losses': tr_losses,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'test_loss': test_loss,
            'test_loss_r': test_loss_r,
            'test_acc': test_acc,
            'test_acc_r': test_acc_r,
        }
        joblib.dump(vard, vard_fn, compress=3)

    print(f'\nDone. Models saved to {args.output}/')
    print(f'Test accuracy: mean={np.mean(test_accs):.2%}, std={np.std(test_accs):.2%}')

    # Save summary
    summary = {
        'test_accs': test_accs,
        'test_accs_r': test_accs_r,
        'min_val_losses': min_val_losses,
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'test_acc_r_mean': float(np.mean(test_accs_r)),
        'test_acc_r_std': float(np.std(test_accs_r)),
    }
    summary_fn = os.path.join(
        args.output, f'rsscnn_{args.cnn}_gap_{args.attribute}_summary.joblib')
    joblib.dump(summary, summary_fn, compress=3)
    print(f'Summary saved to {summary_fn}')


if __name__ == '__main__':
    main()
