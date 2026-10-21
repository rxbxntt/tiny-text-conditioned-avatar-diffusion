import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from pathlib import Path

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import create_dataset_splits

from text.caption_generator import generate_caption
from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection

from diffusion.diffuser import DiffusionScheduler
from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet

NUM_TIMESTEPS = 1000
SCHEDULE_TYPE = "linear"
IMAGE_SIZE = 32
TIME_EMBEDDING_DIM = 64
CONTEXT_DIM = 128

# ============================================================
# DATASET
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\utente\...\cartoonset100k\0"
)

dataset = CartoonSetDataset(
    root_dir=DATASET_ROOT,
    expected_samples=10_000,
)

(
    train_indices,
    val_indices,
    test_indices,
    ood_indices,
) = create_dataset_splits(
    dataset=dataset,
    ood_hair_color=4,
    ood_eye_color=1,
    seed=42,
)

print("Dataset:", len(dataset))
print("Train:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))
print("OOD:", len(ood_indices))


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

scheduler = DiffusionScheduler(
    num_timesteps=NUM_TIMESTEPS,
    schedule_type=SCHEDULE_TYPE,
)

scheduler.betas = scheduler.betas.to(device)
scheduler.alphas = scheduler.alphas.to(device)
scheduler.alpha_bars = scheduler.alpha_bars.to(device)
scheduler.sqrt_alpha_bars = scheduler.sqrt_alpha_bars.to(device)
scheduler.sqrt_one_minus_alpha_bars = (
    scheduler.sqrt_one_minus_alpha_bars.to(device)
)

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = CaptionTokenizer()

train_captions = []

for index in train_indices:
    sample = dataset.samples[index]
    metadata = dataset._load_metadata(sample["metadata"])
    train_captions.append(
        generate_caption(metadata)
    )

tokenizer.fit(train_captions)

print("Vocabulary size:", tokenizer.vocab_size)

# ============================================================
# MODELS
# ============================================================

text_encoder = TextEncoder(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    max_length=128,
    dropout=0.0,
).to(device)

conditioning_projection = TextConditioningProjection(
    input_dim=64,
    output_dim=128,
).to(device)

timestep_embedding = TimestepEmbedding(
    num_timesteps=NUM_TIMESTEPS,
    embedding_dim=TIME_EMBEDDING_DIM,
).to(device)

unet = DiffusionUNet(
    time_dim=TIME_EMBEDDING_DIM,
    context_dim=CONTEXT_DIM,
    num_heads=4,
).to(device)

print("Device:", device)


# ============================================================
# CONFIG
# ============================================================

NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

CFG_DROPOUT_PROB = 0.10

CHECKPOINT_DIR = "checkpoints_conditioned_cfg"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# TRAIN / VALIDATION DATA
# ============================================================

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)


def collate_fn(batch):
    return {
        "image": [item["image"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


# ============================================================
# MODELS
# ============================================================

text_encoder.train()
conditioning_projection.train()
timestep_embedding.train()
unet.train()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    list(text_encoder.parameters())
    + list(conditioning_projection.parameters())
    + list(timestep_embedding.parameters())
    + list(unet.parameters()),
    lr=LEARNING_RATE,
)


# ============================================================
# HELPER: PREPARE BATCH
# ============================================================

def prepare_batch(batch):

    images = batch["image"]

    x0 = torch.stack([
        torch.from_numpy(
            np.array(image.resize((32, 32)))
        ).permute(2, 0, 1).float() / 255.0
        for image in images
    ]).to(device)

    captions = [
        generate_caption(metadata)
        for metadata in batch["metadata"]
    ]

    input_ids = torch.tensor(
        [
            tokenizer.encode(
                caption,
                max_length=18,
                padding=True,
                truncation=False,
            )
            for caption in captions
        ],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        [
            tokenizer.attention_mask(ids)
            for ids in input_ids.cpu().tolist()
        ],
        dtype=torch.bool,
        device=device,
    )

    return x0, input_ids, attention_mask


# ============================================================
# TRAINING STEP
# ============================================================

def training_step(
    x0,
    input_ids,
    attention_mask,
    apply_cfg_dropout=True,
):

    # --------------------------------------------------------
    # Text conditioning
    # --------------------------------------------------------

    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    context = conditioning_projection(text_features)

    # --------------------------------------------------------
    # CFG conditioning dropout
    # --------------------------------------------------------

    if apply_cfg_dropout:

        batch_size = x0.size(0)

        dropout_mask = (
            torch.rand(
                batch_size,
                device=device,
            )
            < CFG_DROPOUT_PROB
        )

        # Unconditional samples receive zero text context
        context = context.clone()
        context[dropout_mask] = 0.0

    # --------------------------------------------------------
    # Diffusion timestep
    # --------------------------------------------------------

    batch_size = x0.size(0)

    t = scheduler.sample_timesteps(
        batch_size=batch_size,
        device=device,
    )

    # --------------------------------------------------------
    # Gaussian noise
    # --------------------------------------------------------

    noise = torch.randn_like(x0)

    # --------------------------------------------------------
    # Forward diffusion
    # --------------------------------------------------------

    xt = (
        scheduler.sqrt_alpha_bar(t, x0) * x0
        + scheduler.sqrt_one_minus_alpha_bar(t, x0) * noise
    )

    # --------------------------------------------------------
    # Time conditioning
    # --------------------------------------------------------

    time_embedding = timestep_embedding(t)

    # --------------------------------------------------------
    # Noise prediction
    # --------------------------------------------------------

    predicted_noise = unet(
        x=xt,
        time_embedding=time_embedding,
        context=context,
    )

    # --------------------------------------------------------
    # Epsilon prediction objective
    # --------------------------------------------------------

    loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    return loss


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    text_encoder.eval()
    conditioning_projection.eval()
    timestep_embedding.eval()
    unet.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:

        x0, input_ids, attention_mask = prepare_batch(batch)

        loss = training_step(
            x0,
            input_ids,
            attention_mask,
            apply_cfg_dropout=False,
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ============================================================
# TRAINING
# ============================================================

training_start = datetime.now().isoformat(timespec="seconds")
start_time = time.time()

num_train_batches = len(train_loader)

print("=" * 70)
print("CONDITIONAL DIFFUSION TRAINING WITH CFG")
print("=" * 70)
print("Start:", training_start)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Train batches / epoch:", num_train_batches)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("CFG dropout probability:", CFG_DROPOUT_PROB)
print("=" * 70)


for epoch in range(1, NUM_EPOCHS + 1):

    text_encoder.train()
    conditioning_projection.train()
    timestep_embedding.train()
    unet.train()

    total_train_loss = 0.0
    num_batches = 0

    epoch_start = time.time()

    for batch_idx, batch in enumerate(train_loader, start=1):

        x0, input_ids, attention_mask = prepare_batch(batch)

        optimizer.zero_grad()

        loss = training_step(
            x0,
            input_ids,
            attention_mask,
            apply_cfg_dropout=True,
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

        # ----------------------------------------------------
        # Batch progress
        # ----------------------------------------------------

        elapsed_batch = time.time() - epoch_start
        avg_batch_time = elapsed_batch / batch_idx
        remaining_batches = num_train_batches - batch_idx
        estimated_remaining = avg_batch_time * remaining_batches

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}] "
            f"Batch [{batch_idx:04d}/{num_train_batches}] | "
            f"Loss: {loss.item():.6f} | "
            f"ETA epoch: {estimated_remaining:.1f}s"
        )

    train_loss = total_train_loss / num_batches

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss = validate()

    epoch_time = time.time() - epoch_start

    timestamp = datetime.now().isoformat(timespec="seconds")

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"checkpoint_epoch_{epoch}.pt",
    )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict": {
                "text_encoder":
                    text_encoder.state_dict(),

                "conditioning_projection":
                    conditioning_projection.state_dict(),

                "timestep_embedding":
                    timestep_embedding.state_dict(),

                "unet":
                    unet.state_dict(),
            },

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss": train_loss,
            "val_loss": val_loss,

            "timestamp": timestamp,

            "config": {
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,

                "num_timesteps": 1000,
                "image_size": 32,

                "text_hidden_size": 64,
                "text_layers": 2,
                "context_dim": 128,

                "cfg_enabled": True,
                "cfg_dropout_probability": CFG_DROPOUT_PROB,
            },
        },
        checkpoint_path,
    )

    print()
    print(
        f"Epoch [{epoch:03d}/{NUM_EPOCHS}] COMPLETE | "
        f"Train: {train_loss:.6f} | "
        f"Val: {val_loss:.6f} | "
        f"Time: {epoch_time:.1f}s | "
        f"Checkpoint: {checkpoint_path}"
    )
    print()


# ============================================================
# TRAINING END
# ============================================================

training_end = datetime.now().isoformat(timespec="seconds")
elapsed = time.time() - start_time

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print("Start:", training_start)
print("End:", training_end)
print(f"Total duration: {elapsed / 60:.2f} minutes")
print("Final checkpoint:", checkpoint_path)
print("=" * 70)