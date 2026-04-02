import os
os.makedirs("outputs/figures", exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt

log_path = "outputs/logs/training_log.csv"
df = pd.read_csv(log_path)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot loss curves
ax[0].plot(df["epoch"], df["train_loss"], label="train", color="tab:blue", ls='solid', lw=2)
ax[0].plot(df["epoch"], df["val_loss"], label="val", color="tab:orange", ls='dashed', lw=2)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Learning Curves")
ax[0].set_aspect((ax[0].get_xlim()[1] - ax[0].get_xlim()[0]) / (ax[0].get_ylim()[1] - ax[0].get_ylim()[0]))
ax[0].legend()

# Plot accuracy curves
ax[1].plot(df["epoch"], df["train_acc"], label="train", color="tab:blue", ls='solid', lw=2)
ax[1].plot(df["epoch"], df["val_acc"], label="val", color="tab:orange", ls='dashed', lw=2)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Accuracy Curves")
ax[1].set_aspect((ax[1].get_xlim()[1] - ax[1].get_xlim()[0]) / (ax[1].get_ylim()[1] - ax[1].get_ylim()[0]))
ax[1].legend()

plt.tight_layout()
plt.savefig("outputs/figures/learning_curves.png", dpi=300, bbox_inches="tight")
plt.show()