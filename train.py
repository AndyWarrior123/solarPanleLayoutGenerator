import time
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import make_datasets
from src.model   import SolarUNet
from src.loss    import CombinedLoss
from src.utils   import load_config, EarlyStopping, save_checkpoint


def main():
    # ── 1. Config & device ───────────────────────────────────────────────────
    print("=" * 60)
    print("  Solar Panel Layout Designer — Training")
    print("=" * 60)

    cfg    = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/5] Device : {device}")
    if device.type == "cuda":
        print(f"       GPU    : {torch.cuda.get_device_name(0)}")

    # ── 2. Datasets ──────────────────────────────────────────────────────────
    print("\n[2/5] Loading dataset...")
    train_ds, val_ds = make_datasets(cfg)
    print(f"       Train samples : {len(train_ds)}")
    print(f"       Val samples   : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True,  num_workers=cfg.data.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=device.type == "cuda",
    )

    # ── 3. Model, loss, optimiser ─────────────────────────────────────────────
    print("\n[3/5] Building model...")
    model     = SolarUNet(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.early_stopping_patience * 2
    )
    stopper = EarlyStopping(cfg.training.early_stopping_patience)
    scaler  = GradScaler(device=device.type, enabled=device.type == "cuda")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Encoder         : {cfg.model.encoder} (ImageNet pretrained)")
    print(f"       Trainable params: {total_params:,}")
    print(f"       Epochs          : {cfg.training.epochs}  |  batch: {cfg.training.batch_size}  |  lr: {cfg.training.lr}")

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print("\n[4/5] Training...\n")
    best_val   = float("inf")
    start_time = time.time()

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.time()

        # — train —
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{cfg.training.epochs} [train]",
                    leave=False, ncols=72)
        for images, masks, meta in loop:
            images, masks, meta = images.to(device), masks.to(device), meta.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images, meta), masks, meta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # — validate —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, meta in tqdm(val_loader,
                                            desc=f"Epoch {epoch:03d}/{cfg.training.epochs} [val]  ",
                                            leave=False, ncols=72):
                images, masks, meta = images.to(device), masks.to(device), meta.to(device)
                val_loss += criterion(model(images, meta), masks, meta).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step()
        lr_now      = scheduler.get_last_lr()[0]
        elapsed     = time.time() - epoch_start

        improved = "  ◀ best" if val_loss < best_val else ""
        print(f"Epoch {epoch:03d}/{cfg.training.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  ({elapsed:.0f}s){improved}")

        # — checkpoints —
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            f"{cfg.training.checkpoint_dir}/best.pt")

        if stopper(val_loss):
            print(f"\nEarly stopping triggered (no improvement for "
                  f"{cfg.training.early_stopping_patience} epochs).")
            break

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n[5/5] Done.")
    print(f"       Best val loss  : {best_val:.4f}")
    print(f"       Total time     : {total_time / 60:.1f} min")
    print(f"       Best checkpoint: {cfg.training.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
