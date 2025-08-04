
import torch
import wandb 
import time 

EPOCHS = 150  #@param {type:"integer"}
LR = 1e-3     #@param {type:"number"}
BASE_CH = 32  #@param {type:"integer"}
COND_METHOD = "film"  #@param ["film", "concat_rgb", "concat_idx"]
WANDB_PROJECT = "ayna-conditional-unet"  #@param {type:"string"}
USE_WANDB = True  #@param {type:"boolean"}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


num_colors = len(train_ds.color_to_idx)
model = UNetCond(in_ch=3, out_ch=3, base_ch=BASE_CH, cond_method=COND_METHOD, num_colors=num_colors, cond_dim=32).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

if USE_WANDB and WANDB_AVAILABLE:

    wandb.init(project=WANDB_PROJECT, config={
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "base_ch": BASE_CH,
        "cond_method": COND_METHOD,
        "augment": True,
        "out_size": OUT_SIZE,
    })
    wandb.watch(model, log="gradients", log_freq=50)
else:
    print("W&B disabled or not available.")

best_val = float('inf')
save_path = f"/content/cond_unet_{COND_METHOD}.pt"

for epoch in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    running = {"loss":0.0, "mse":0.0, "l1":0.0}
    for batch in train_loader:
        img = batch["inp"].to(device)
        tgt = batch["target"].to(device)
        color_idx = batch["color_idx"].to(device)
        color_rgb = batch["color_rgb"].to(device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(img, color_idx, color_rgb)
            loss = F.l1_loss(pred, tgt) * 0.7 + F.mse_loss(pred, tgt) * 0.3
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        m = compute_metrics(pred.detach(), tgt)
        running["loss"] += loss.item()
        running["mse"]  += m["mse"]
        running["l1"]   += m["l1"]

    n_batches = max(1, len(train_loader))
    train_log = {k: v/n_batches for k,v in running.items()}

    model.eval()
    with torch.no_grad():
        val_running = {"loss":0.0, "mse":0.0, "l1":0.0}
        all_preds = []
        for batch in val_loader:
            img = batch["inp"].to(device)
            tgt = batch["target"].to(device)
            color_idx = batch["color_idx"].to(device)
            color_rgb = batch["color_rgb"].to(device)
            pred = model(img, color_idx, color_rgb)
            loss = F.l1_loss(pred, tgt) * 0.7 + F.mse_loss(pred, tgt) * 0.3
            m = compute_metrics(pred, tgt)
            val_running["loss"] += loss.item()
            val_running["mse"]  += m["mse"]
            val_running["l1"]   += m["l1"]
            all_preds.append(pred.cpu())
        n_val_batches = max(1, len(val_loader))
        val_log = {f"val_{k}": v/n_val_batches for k,v in val_running.items()}


    train_log["psnr"] = psnr(torch.tensor(train_log["mse"])).item()
    val_log["val_psnr"] = psnr(torch.tensor(val_log["val_mse"])).item()
    elapsed = time.time() - t0

    log = {"epoch": epoch, "time_s": elapsed, **train_log, **val_log}
    print(log)

    if USE_WANDB and WANDB_AVAILABLE:
        wandb.log(log)

    if val_log["val_loss"] < best_val:
        best_val = val_log["val_loss"]
        torch.save({"model": model.state_dict(),
                    "color_to_idx": train_ds.color_to_idx,
                    "idx_to_color": train_ds.idx_to_color,
                    "color_to_rgb": color_to_rgb,
                    "hparams": {"base_ch": BASE_CH, "cond_method": COND_METHOD, "out_size": OUT_SIZE}}, save_path)

batch = next(iter(val_loader))
model.eval()
with torch.no_grad():
    preds = model(batch["inp"].to(device), batch["color_idx"].to(device), batch["color_rgb"].to(device))
show_batch(batch, preds, max_n=min(6, batch["inp"].size(0)))
print("Best checkpoint saved at:", save_path)