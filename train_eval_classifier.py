import os
os.environ["XFORMERS_DISABLED"] = "1"
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MCCDDataset
from eval_models import MultiTaskCalligraphyEvalNet

# Limit PyTorch CPU threads to prevent high CPU load
torch.set_num_threads(2)

def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def evaluate(model, val_loader, device, max_eval_steps=-1):
    """Evaluate model on validation set (strictly no gradients)"""
    model.eval()
    total_loss = 0.0
    char_top1_acc = 0.0
    char_top5_acc = 0.0
    callig_acc = 0.0
    script_acc = 0.0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, batch in enumerate(val_loader, 1):
            x = batch['image'].to(device, non_blocking=True)
            y_char = batch['y_char'].to(device, non_blocking=True)
            y_callig = batch['y_callig'].to(device, non_blocking=True)
            y_script = batch['y_script'].to(device, non_blocking=True)
            
            b = x.size(0)
            
            with torch.amp.autocast('cuda'):
                logits_char, logits_callig, logits_script = model(x)
                l_char = criterion(logits_char, y_char)
                l_callig = criterion(logits_callig, y_callig)
                l_script = criterion(logits_script, y_script)
                loss = l_char + l_callig + l_script
                
            total_loss += loss.item() * b
            
            c_top1, c_top5 = compute_accuracy(logits_char, y_char, topk=(1, 5))
            cal_acc, = compute_accuracy(logits_callig, y_callig, topk=(1,))
            sc_acc, = compute_accuracy(logits_script, y_script, topk=(1,))
            
            char_top1_acc += c_top1 * b
            char_top5_acc += c_top5 * b
            callig_acc += cal_acc * b
            script_acc += sc_acc * b
            total_samples += b

            if max_eval_steps > 0 and step >= max_eval_steps:
                break

    avg_loss = total_loss / total_samples
    avg_char_top1 = char_top1_acc / total_samples
    avg_char_top5 = char_top5_acc / total_samples
    avg_callig = callig_acc / total_samples
    avg_script = script_acc / total_samples

    return avg_loss, avg_char_top1, avg_char_top5, avg_callig, avg_script

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Evaluation Model Training on Device: {device} ===")
    print(f"Dataset CSV: Train='{args.train_csv}', Val='{args.val_csv}'")
    print(f"CPU Optimizations: PyTorch Threads=2, load_maps=False (3x I/O speedup)")
    print(f"VRAM Optimization: AMP=True, Batch Size={args.batch_size}")

    # 1. Datasets (Strict separation & load_maps=False for 3x speedup!)
    train_ds = MCCDDataset(csv_file=args.train_csv, root_dir=args.data_dir, image_size=args.image_size, load_maps=False, is_train=True)
    val_ds = MCCDDataset(csv_file=args.val_csv, root_dir=args.data_dir, image_size=args.image_size, load_maps=False, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    # 2. Model
    model = MultiTaskCalligraphyEvalNet(
        num_characters=args.num_characters,
        num_calligraphers=args.num_calligraphers,
        num_scripts=args.num_scripts,
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        unfreeze_blocks=args.unfreeze_blocks
    ).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_val_acc = 0.0

    # 3. Training Loop
    if getattr(args, 'eval_only', False):
        print("\n--- Running Evaluation on Test Set ---")
        # Load best weights
        if os.path.exists(args.save_path):
            checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint from {args.save_path} (OCR Top-1: {checkpoint.get('val_c_top1', 0.0):.2f}%)")
        else:
            print(f"Warning: Checkpoint not found at {args.save_path}")
            
        test_loss, test_c_top1, test_c_top5, test_callig, test_script = evaluate(model, val_loader, device)
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"OCR Top-1 Acc: {test_c_top1:.2f}% (Top-5: {test_c_top5:.2f}%)")
        print(f"Calligrapher Acc: {test_callig:.2f}%")
        print(f"Script Acc: {test_script:.2f}%\n")
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(train_loader, 1):
            x = batch['image'].to(device, non_blocking=True)
            y_char = batch['y_char'].to(device, non_blocking=True)
            y_callig = batch['y_callig'].to(device, non_blocking=True)
            y_script = batch['y_script'].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits_char, logits_callig, logits_script = model(x)
                l_char = criterion(logits_char, y_char)
                l_callig = criterion(logits_callig, y_callig)
                l_script = criterion(logits_script, y_script)
                loss = 1.0 * l_char + 1.0 * l_callig + 0.2 * l_script

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] LR: {current_lr:.6f} | Train Loss: {loss.item():.4f} (Char: {l_char.item():.4f}, Callig: {l_callig.item():.4f}, Script: {l_script.item():.4f})", flush=True)

            if args.max_steps > 0 and step >= args.max_steps:
                print(f"Reached max_steps={args.max_steps} for dry-run testing.")
                break

        elapsed = time.time() - start_time
        print(f"\n--- Epoch {epoch} Complete in {elapsed:.1f}s ---")
        
        # Step learning rate scheduler at end of epoch
        scheduler.step()

        # Evaluate on validation set
        val_loss, val_c_top1, val_c_top5, val_callig, val_script = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f} | OCR Top-1 Acc: {val_c_top1:.2f}% (Top-5: {val_c_top5:.2f}%) | Calligrapher Acc: {val_callig:.2f}% | Script Acc: {val_script:.2f}%\n", flush=True)

        # Save best model based on OCR Top-1 Accuracy
        if val_c_top1 > best_val_acc:
            best_val_acc = val_c_top1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_c_top1': val_c_top1,
                'val_callig': val_callig,
                'val_script': val_script,
                'args': args
            }
            torch.save(checkpoint, args.save_path)
            print(f"Saved new best model checkpoint to {args.save_path} (OCR Top-1: {val_c_top1:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Calligraphy Evaluation Classifiers & OCR Model")
    parser.add_argument("--train-csv", type=str, default="train.csv", help="Path to training set CSV")
    parser.add_argument("--val-csv", type=str, default="val.csv", help="Path to validation set CSV")
    parser.add_argument("--data-dir", type=str, default="", help="Root dataset directory if CSV has relative paths")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14", "resnet18", "resnet34"])
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone weights for fast linear probe")
    parser.add_argument("--unfreeze-blocks", type=int, default=0, help="If freeze_backbone is True, unfreeze the last N blocks of the ViT backbone")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on val-csv (test set) using saved checkpoint")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-characters", type=int, default=7765)
    parser.add_argument("--num-calligraphers", type=int, default=2243)
    parser.add_argument("--num-scripts", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=-1, help="Max steps per epoch for quick dry-run testing (-1 for full epoch)")
    parser.add_argument("--save-path", type=str, default="pretrained_models/eval_classifier.pt")
    
    args = parser.parse_args()
    train(args)
