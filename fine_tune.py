
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model import MiniLLM, TextDataset

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def log_metrics(log_dir, step, metrics):
    log_file = os.path.join(log_dir, "metrics.jsonl")
    log_entry = {"step": step, **metrics}
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--train_steps", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default="logs/fine_tune")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--freeze_layers", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Create directories
    os.makedirs(os.path.join(args.model_name, "checkpoints_finetuned"), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load config
    with open(os.path.join(args.model_name, "config.json"), "r") as f:
        config = json.load(f)

    # Initialize model
    model = MiniLLM(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        activation_function=config["activation_function"],
    )
    model.load_state_dict(torch.load(args.checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Causal mask
    mask = torch.nn.Transformer.generate_square_subsequent_mask(config["seq_len"]).to(device)

    # Freeze layers if specified
    if args.freeze_layers:
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False

    # Load vocabulary
    with open("vocab.json", "r") as f:
        vocab = json.load(f)
    char_to_idx = vocab["char_to_idx"]

    # Setup data loaders
    train_dataset = TextDataset(args.train_data, config["seq_len"], char_to_idx)
    valid_dataset = TextDataset(args.valid_data, config["seq_len"], char_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"])

    # Setup optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.train_steps)

    # Training loop
    step = 0
    while step < args.train_steps:
        for batch in train_loader:
            model.train()
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            outputs = model(inputs, mask=mask[:inputs.size(1), :inputs.size(1)])
            loss = torch.nn.functional.cross_entropy(outputs.reshape(-1, config["vocab_size"]), targets.reshape(-1))
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % args.checkpoint_interval == 0:
                # Save checkpoint
                checkpoint_path = os.path.join(args.model_name, "checkpoints_finetuned", f"step_{step + 1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                # Validation
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for valid_batch in valid_loader:
                        inputs = valid_batch[:, :-1].to(device)
                        targets = valid_batch[:, 1:].to(device)
                        valid_outputs = model(inputs, mask=mask[:inputs.size(1), :inputs.size(1)])
                        valid_loss = torch.nn.functional.cross_entropy(valid_outputs.reshape(-1, config["vocab_size"]), targets.reshape(-1))
                        total_loss += valid_loss.item()
                avg_loss = total_loss / len(valid_loader)
                print(f"Step {step + 1}, Validation Loss: {avg_loss}")
                log_metrics(args.log_dir, step + 1, {"validation_loss": avg_loss})

            step += 1
            if step >= args.train_steps:
                break

if __name__ == "__main__":
    main()
