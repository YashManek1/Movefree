"""
Retraining Pipeline - FAST FINE-TUNING (Frozen Backbone)
"""

from ultralytics import YOLO
import torch
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrain_fast():
    if not torch.cuda.is_available():
        logger.error("❌ No GPU Found!")
        return

    data_yaml = "datasets/movefree_combined/movefree.yaml"
    previous_best = "runs/detect/movefree_ultimate_fast/weights/best.pt"

    if not Path(previous_best).exists():
        logger.error(f"❌ {previous_best} not found!")
        return

    model = YOLO(previous_best)
    logger.info(f"✅ Loaded: {previous_best}")

    train_args = {
        "data": data_yaml,
        "epochs": 20,  # 20 Epochs is enough for fine-tuning
        "patience": 5,  # Stop quickly if no gains
        # --- SPEED HACKS ---
        "freeze": 10,  # FREEZE BACKBONE: Massive speedup, keeps "Stairs" accuracy high
        "batch": 32,  # Higher batch size (fits because we froze layers)
        "imgsz": 512,  # Keep 512 for consistency
        "cache": False,  # No RAM caching (System limit)
        "workers": 8,  # Max CPU usage
        "device": 0,
        "project": "runs/detect",
        "name": "movefree_finetune_fast",
        "exist_ok": True,
        # --- LEARNING SETTINGS ---
        "optimizer": "AdamW",
        "lr0": 0.0005,  # Very Low LR to gently fix the 'Head'
        "cos_lr": True,  # Cosine scheduler for smooth convergence
    }

    logger.info("🚀 Starting Fast Fine-Tuning (Frozen Backbone)...")
    model.train(**train_args)
    logger.info("✅ Done.")


if __name__ == "__main__":
    retrain_fast()
