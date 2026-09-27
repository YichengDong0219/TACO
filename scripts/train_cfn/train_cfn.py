import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cfn.cfn_net import CFN
from cfn.feature_dataset import cfn_feature_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFN with feature dataset")

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--save_freq", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)

    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=2)

    parser.add_argument("--feature_dir", type=str,
                        default="")
    parser.add_argument("--multi_feature_file", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--task_balanced", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--cfn_output_dim", type=int, default=20)
    parser.add_argument("--cfn_hidden_dim", type=int, default=1536)

    args = parser.parse_args()
    return args


def train(args):
    overall_start = time.time()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- dataset ----------------
    t0 = time.time()
    dataset = cfn_feature_dataset(
        feature_dir=args.feature_dir,
        multi_feature_file=args.multi_feature_file,
        output_dimensions=args.cfn_output_dim,
    )
    t1 = time.time()
    print(f"📦 dataset loading time: {t1 - t0:.2f}s")

    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.task_balanced:
        sampler = torch.utils.data.WeightedRandomSampler(
            dataset.sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type != "cpu"),
        drop_last=False,
        generator=generator,
    )

    # ---------------- model ----------------
    t0 = time.time()
    model = CFN(
        input_dim=args.input_dim,
        cfn_output_dim=args.cfn_output_dim,
        cfn_hidden_dim=args.cfn_hidden_dim,
    ).to(device)
    t1 = time.time()
    print(f"🧠 model init time: {t1 - t0:.2f}s")

    num_batches_per_epoch = len(dataloader)
    total_epochs = args.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        steps_per_epoch=math.ceil(len(dataloader) / args.grad_accum_steps),
        epochs=total_epochs,
        anneal_strategy="cos", 
    )

    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    config = vars(args).copy()
    config["task_counts"] = dataset.task_counts
    config["num_samples"] = len(dataset)
    config["sampling_strategy"] = (
        "task_balanced_replacement" if args.task_balanced else "shuffled_concatenation"
    )
    (output_dir / "train_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"

    model.train()

    grad_accum_steps = args.grad_accum_steps
    log_interval = args.log_interval

    for epoch in range(total_epochs):
        print(f"\n📘 Epoch {epoch + 1} start")
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        accum_batch_time = 0.0
        accum_data_time = 0.0
        accum_forward_time = 0.0
        accum_backward_time = 0.0
        batch_counter = 0

        batch_start = time.time()
        data_start = time.time()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            data_end = time.time()

            forward_start = time.time()
            loss, model_output_train = model.compute_loss_feature(batch)
            forward_end = time.time()

            loss = loss / grad_accum_steps 
            backward_start = time.time()
            loss.backward()
            backward_end = time.time()

            batch_end = time.time()

            total_loss += loss.item() * grad_accum_steps
            batch_counter += 1

            train_norm = model_output_train.norm(p=2, dim=1).mean()
            prior_o_norm = model.cfn.prior_outputs.norm(p=2, dim=1).mean()
            prior_norm = model.cfn.prior.norm(p=2, dim=1).mean()
            global_step = epoch * num_batches_per_epoch + batch_counter
            writer.add_scalar("Norm/train", train_norm.item(), global_step)
            writer.add_scalar("Norm/prior_o", prior_o_norm.item(), global_step)
            writer.add_scalar("Norm/prior", prior_norm.item(), global_step)
            writer.add_scalar("cfn/prior_mean", model.cfn.prior_mean.mean().item(), global_step)
            writer.add_scalar("cfn/prior_var", model.cfn.prior_var.mean().item(), global_step)

            accum_batch_time += batch_end - batch_start
            accum_data_time += data_end - data_start
            accum_forward_time += forward_end - forward_start
            accum_backward_time += backward_end - backward_start

            if batch_counter % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if batch_counter % log_interval == 0:
                step = epoch * num_batches_per_epoch + batch_counter
                avg_loss = total_loss / batch_counter
                writer.add_scalar("Train/loss", avg_loss, step)
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Train/lr", current_lr, step)

            if batch_counter % 200 == 0:
                print(
                    f"⏱️ avg 200 batch time: "
                    f"batch={accum_batch_time:.2f}s | "
                    f"data={accum_data_time:.2f}s | "
                    f"forward={accum_forward_time:.2f}s | "
                    f"backward={accum_backward_time:.2f}s"
                )
                accum_batch_time = 0.0
                accum_data_time = 0.0
                accum_forward_time = 0.0
                accum_backward_time = 0.0

            if batch_counter >= num_batches_per_epoch:
                break

            batch_start = time.time()
            data_start = time.time()

        if batch_counter % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss = total_loss / batch_counter
        print(f"✅ Epoch {epoch + 1} avg Loss: {avg_loss:.4f}")
        print(f"🕒 Epoch time: {time.time() - epoch_start:.2f}s")
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }) + "\n")

        if (epoch + 1) % args.save_freq == 0:
            torch.save(model.cfn.state_dict(), output_dir / f"model_epoch{epoch + 1}.pt")

    writer.close()
    print(f"✅ all time: {time.time() - overall_start:.2f}s")


if __name__ == "__main__":
    args = parse_args()
    train(args)
