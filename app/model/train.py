from utils import get_device_torch
from transformers import AdamW, get_linear_schedule_with_warmup
from .EmbeddingLSTM import EmbeddingLSTM
import time
import torch.nn as nn
import torch

device = get_device_torch()
loss_fn = nn.CrossEntropyLoss()

def train_bilstm(epochs, dropout, input_dim, hidden_dim, output_dim, bert_model, adam_eps, adam_lr, train_dataloader,
                  model_route):

    model = EmbeddingLSTM(dropout, input_dim, hidden_dim, output_dim, bert_config=bert_model)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=adam_lr, eps=adam_eps)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    print("Start training...\n")
    for epoch_i in range(epochs):

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step % 200 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        print("-" * 70)
        print("\n")

    model_route = f"{model_route}"

    torch.save(model, model_route)
    print("Training screening!")
    return model