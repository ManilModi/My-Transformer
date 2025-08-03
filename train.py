import torch
import torch.nn as nn

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, max_grad_norm=0.5):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()

        logits = model(src, tgt_inp)

        # Flatten logits and targets for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        tgt_out_flat = tgt_out.reshape(-1)

        loss = loss_fn(logits_flat, tgt_out_flat)
        loss.backward()

        # 🚨 Gradient clipping to prevent NaNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
