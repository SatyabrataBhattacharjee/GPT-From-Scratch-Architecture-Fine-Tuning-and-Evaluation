import torch
from training.losses import calc_loss_batch
from training.eval import calc_loss_loader
from generation.sample import generate_and_print_sample

def train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter, start_context, tokenizer
):
    step = 0

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()

            if step % eval_freq == 0:
                train_loss, train_ppl = evaluate_model(
                    train_loader, model, device, eval_iter
                )
                val_loss, val_ppl = evaluate_model(
                    val_loader, model, device, eval_iter
                )

                print(
                    f"Epoch {epoch+1} Step {step} | "
                    f"Train loss {train_loss:.3f}, ppl {train_ppl:.2f} | "
                    f"Val loss {val_loss:.3f}, ppl {val_ppl:.2f}"
                )

            step += 1

        generate_and_print_sample(model, tokenizer, device, start_context)
