import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from src.model import SimCLR
from src.loss import NTXentLoss


# Mixed Precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# XLA (Optional but faster)
tf.config.optimizer.set_jit(True)

#########################################################
# Paths
#########################################################

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MODEL_DIR = Path("saved_models")
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

#########################################################
# Hyperparameters
#########################################################

BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.5
PROJECTION_DIM = 128

#########################################################
# Model
#########################################################

model = SimCLR(projection_dim=PROJECTION_DIM)
dummy = tf.random.normal((1, 32, 32, 3))
_ = model(dummy, training=False)

# Loss
loss_fn = NTXentLoss(temperature=TEMPERATURE)

#########################################################
# Optimizer
#########################################################

STEPS_PER_EPOCH = 50000 // BATCH_SIZE
TOTAL_STEPS = (STEPS_PER_EPOCH * EPOCHS)

WARMUP_EPOCHS = 10
WARMUP_STEPS = WARMUP_EPOCHS * STEPS_PER_EPOCH

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, base_lr):

        super().__init__()
        self.base_lr = base_lr
        self.cosine = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=base_lr, decay_steps=TOTAL_STEPS - WARMUP_STEPS, alpha=1e-2)

    def __call__(self, step):

        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / tf.cast(WARMUP_STEPS, tf.float32))
        cosine_lr = self.cosine(step - WARMUP_STEPS)

        return tf.where(step < WARMUP_STEPS, warmup_lr, cosine_lr)


# lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=LEARNING_RATE,
#     decay_steps=TOTAL_STEPS,
#     alpha=1e-2
# )

lr_schedule = WarmUpCosine(LEARNING_RATE)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, beta_1=0.9, beta_2=0.999)

#########################################################
# Checkpoint
#########################################################

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

#########################################################
# Resume Training
#########################################################

if manager.latest_checkpoint:

    checkpoint.restore(manager.latest_checkpoint)
    print(f"Checkpoint restored from {manager.latest_checkpoint}")

else:

    print("Training from scratch.")

#########################################################
# Metrics
#########################################################

train_loss = tf.keras.metrics.Mean(name="train_loss")

#########################################################
# Train Step
#########################################################

@tf.function(jit_compile=True)
def train_step(view1, view2):

    with tf.GradientTape() as tape:

        z1 = model(view1, training=True)
        z2 = model(view2, training=True)
        loss = loss_fn(z1, z2)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)

#########################################################
# Train One Epoch
#########################################################

def train_epoch(dataset):

    train_loss.reset_state()

    for view1, view2 in tqdm(dataset, total=STEPS_PER_EPOCH, desc="Training", leave=False):
        train_step(view1, view2)

    return float(train_loss.result().numpy())

#########################################################
# Main Training
#########################################################

def train(dataset):

    best_loss = float("inf")

    for epoch in range(EPOCHS):

        loss_value = train_epoch(dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} "f"Loss: {loss_value:.4f}")

        if loss_value < best_loss:

            best_loss = loss_value
            manager.save()
            model.encoder.save(SAVED_MODEL_DIR / "encoder.keras", overwrite=True)
            model.projection_head.save(SAVED_MODEL_DIR / "projection_head.keras", overwrite=True)
            model.save(SAVED_MODEL_DIR / "simclr.keras", overwrite=True)
            print(f"Checkpoint Saved (Best Loss = {best_loss:.4f})")

    print("Training Finished.")