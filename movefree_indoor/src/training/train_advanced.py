"""
MoveFree Indoor Navigation - Complete Training Pipeline (WITH RESUME)
Optimized for Raspberry Pi 5 deployment with proper dataset
"""

from ultralytics import YOLO
import torch
import logging
import sys
from pathlib import Path
import yaml
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MoveFreeTrainer:
    """
    Complete training pipeline for MoveFree indoor navigation
    Supports both prototype (RPi5) and production (Jetson) models
    """

    def __init__(self, deployment_target="raspberry_pi"):
        """
        Args:
            deployment_target: "raspberry_pi" or "jetson"
        """
        self.deployment_target = deployment_target
        self.data_yaml = "datasets/movefree_combined/movefree.yaml"

        # Model selection based on target
        if deployment_target == "raspberry_pi":
            self.model_size = "n"  # nano - 3.2M params
            self.imgsz = 416  # Smaller for RPi5
            self.batch = 16
            logger.info("🎯 Target: Raspberry Pi 5 (Prototype)")
            logger.info("📦 Model: YOLOv8n (Optimized for edge)")
        else:  # jetson
            self.model_size = "s"  # small - 11.2M params
            self.imgsz = 640
            self.batch = 32
            logger.info("🎯 Target: Jetson Nano/Orin (Production)")
            logger.info("📦 Model: YOLOv8s (Balanced accuracy/speed)")

        self.model_name = f"yolov8{self.model_size}.pt"

        # Verify dataset
        if not Path(self.data_yaml).exists():
            logger.error(f"❌ Dataset not found: {self.data_yaml}")
            logger.error("Please run: python src/data/merge_datasets.py")
            sys.exit(1)

        # Check GPU
        self.device = 0 if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("⚠️ No GPU detected. Training will be VERY slow.")
            logger.warning("Recommended: Use Google Colab or Kaggle for training")

    def verify_dataset(self):
        """Verify dataset integrity before training"""
        logger.info("🔍 Verifying dataset...")

        with open(self.data_yaml, "r") as f:
            config = yaml.safe_load(f)

        dataset_path = Path(config["path"])

        # Check directories
        for split in ["train", "val", "test"]:
            img_dir = dataset_path / "images" / split
            lbl_dir = dataset_path / "labels" / split

            if not img_dir.exists():
                logger.error(f"❌ Missing: {img_dir}")
                return False

            img_count = len(list(img_dir.glob("*")))
            lbl_count = len(list(lbl_dir.glob("*.txt")))

            logger.info(f"   {split:5s}: {img_count:5d} images, {lbl_count:5d} labels")

            if img_count == 0:
                logger.error(f"❌ No images found in {split} set!")
                return False

        # Check class count
        num_classes = config.get("nc", len(config.get("names", {})))
        logger.info(f"   Classes: {num_classes}")

        if num_classes != 19:
            logger.error(f"❌ Expected 19 classes, found {num_classes}")
            return False

        logger.info("✅ Dataset verification passed")
        return True

    def train_from_scratch(self):
        """
        Train from scratch (FIRST TIME TRAINING)
        Use this for initial model training
        """
        logger.info("=" * 60)
        logger.info("🚀 TRAINING FROM SCRATCH")
        logger.info("=" * 60)

        if not self.verify_dataset():
            logger.error("❌ Dataset verification failed. Aborting.")
            return None

        # Load pretrained COCO weights
        model = YOLO(self.model_name)
        logger.info(f"✅ Loaded pretrained {self.model_name}")

        # Training configuration
        train_config = {
            "data": self.data_yaml,
            "epochs": 100,  # Full training
            "patience": 15,  # Early stopping
            "batch": self.batch,
            "imgsz": self.imgsz,
            "device": self.device,
            "workers": 8,
            "project": "runs/detect",
            "name": f"movefree_indoor_{self.model_size}",
            "exist_ok": True,
            # Optimization
            "optimizer": "AdamW",
            "lr0": 0.001,  # Initial learning rate
            "lrf": 0.01,  # Final learning rate
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            # Augmentation (CRITICAL for indoor scenes)
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,  # Rotation
            "translate": 0.1,  # Translation
            "scale": 0.5,  # Scaling
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,  # No vertical flip (gravity matters indoors)
            "fliplr": 0.5,  # Horizontal flip OK
            "mosaic": 1.0,  # Mosaic augmentation
            "mixup": 0.1,  # MixUp
            # Validation
            "val": True,
            "save": True,
            "save_period": -1,  # Save only best/last
            "cache": False,  # Don't cache (RAM limit)
            "rect": False,  # No rectangular training
            "cos_lr": True,  # Cosine LR scheduler
            "close_mosaic": 10,  # Disable mosaic last 10 epochs
            # Misc
            "verbose": True,
            "seed": 42,
            "deterministic": True,
        }

        logger.info("📊 Training Configuration:")
        logger.info(f"   Epochs: {train_config['epochs']}")
        logger.info(f"   Batch Size: {train_config['batch']}")
        logger.info(f"   Image Size: {train_config['imgsz']}")
        logger.info(f"   Device: {self.device}")

        # Start training
        logger.info("\n🏋️ Starting training...")
        logger.info("⏱️ Estimated time: 2-3 hours on RTX 3070 Ti Laptop GPU")

        try:
            results = model.train(**train_config)
            logger.info("\n✅ Training completed successfully!")
            return results
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def resume_training(self, checkpoint_path=None):
        """
        Resume training from checkpoint (INTERRUPTED TRAINING)

        Args:
            checkpoint_path: Path to last.pt checkpoint (auto-detects if None)
        """
        logger.info("=" * 60)
        logger.info("🔄 RESUMING TRAINING")
        logger.info("=" * 60)

        # Auto-detect checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = (
                f"runs/detect/movefree_indoor_{self.model_size}/weights/last.pt"
            )
            logger.info(f"📍 Auto-detected checkpoint: {checkpoint_path}")

        if not Path(checkpoint_path).exists():
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            logger.error("Available checkpoints:")

            # List available checkpoints
            runs_dir = Path("runs/detect")
            if runs_dir.exists():
                for run_path in runs_dir.iterdir():
                    if run_path.is_dir():
                        weights_dir = run_path / "weights"
                        if weights_dir.exists():
                            for weight_file in weights_dir.glob("*.pt"):
                                logger.error(f"   {weight_file}")
            return None

        # Load checkpoint
        model = YOLO(checkpoint_path)
        logger.info(f"✅ Loaded checkpoint: {checkpoint_path}")

        # Check which epoch we're resuming from
        try:
            import torch as pt

            ckpt = pt.load(checkpoint_path, map_location="cpu")
            start_epoch = ckpt.get("epoch", 0) + 1
            logger.info(f"📊 Resuming from epoch {start_epoch}")
        except:
            logger.warning("⚠️ Could not determine start epoch")

        # Resume training
        logger.info("🏋️ Resuming training...")

        try:
            results = model.train(
                resume=True, device=self.device  # KEY: This tells YOLO to resume
            )

            logger.info("\n✅ Resumed training completed successfully!")
            return results

        except Exception as e:
            logger.error(f"❌ Resume failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def fine_tune(self, base_model_path):
        """
        Fine-tune existing model (RETRAINING)
        Use this to improve an already trained model
        """
        logger.info("=" * 60)
        logger.info("🔧 FINE-TUNING EXISTING MODEL")
        logger.info("=" * 60)

        if not Path(base_model_path).exists():
            logger.error(f"❌ Base model not found: {base_model_path}")
            return None

        if not self.verify_dataset():
            logger.error("❌ Dataset verification failed. Aborting.")
            return None

        # Load existing model
        model = YOLO(base_model_path)
        logger.info(f"✅ Loaded model: {base_model_path}")

        # Fine-tuning configuration (lighter than from-scratch)
        finetune_config = {
            "data": self.data_yaml,
            "epochs": 30,  # Fewer epochs
            "patience": 8,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "device": self.device,
            "workers": 8,
            "project": "runs/detect",
            "name": f"movefree_finetune_{self.model_size}",
            "exist_ok": True,
            # Lower learning rate for fine-tuning
            "optimizer": "AdamW",
            "lr0": 0.0003,  # Much lower
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 1,
            # Lighter augmentation
            "hsv_h": 0.01,
            "hsv_s": 0.5,
            "hsv_v": 0.3,
            "degrees": 5.0,
            "translate": 0.05,
            "scale": 0.3,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.5,  # Reduced
            "mixup": 0.05,
            # Freeze backbone (FASTER training)
            "freeze": 10,  # Freeze first 10 layers
            "val": True,
            "save": True,
            "cache": False,
            "cos_lr": True,
            "verbose": True,
            "seed": 42,
        }

        logger.info("📊 Fine-tuning Configuration:")
        logger.info(f"   Epochs: {finetune_config['epochs']}")
        logger.info(f"   Learning Rate: {finetune_config['lr0']}")
        logger.info(f"   Frozen Layers: {finetune_config['freeze']}")

        logger.info("\n🏋️ Starting fine-tuning...")
        logger.info("⏱️ Estimated time: 30-60 minutes (GPU)")

        try:
            results = model.train(**finetune_config)
            logger.info("\n✅ Fine-tuning completed successfully!")
            return results
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def validate(self, weights_path):
        """Validate trained model"""
        logger.info("\n🔍 Running validation...")

        if not Path(weights_path).exists():
            logger.error(f"❌ Weights not found: {weights_path}")
            return None

        model = YOLO(weights_path)

        try:
            metrics = model.val(
                data=self.data_yaml,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device,
                verbose=True,
            )

            logger.info("\n" + "=" * 60)
            logger.info("📊 VALIDATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"mAP@0.5:      {metrics.box.map50:.4f}")
            logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
            logger.info(f"Precision:    {metrics.box.mp:.4f}")
            logger.info(f"Recall:       {metrics.box.mr:.4f}")
            logger.info("=" * 60)

            # Check critical classes
            if hasattr(metrics.box, "maps"):
                critical_classes = {9: "door", 13: "stairs", 12: "person"}

                logger.info("\n🚨 Critical Safety Classes:")
                for class_id, class_name in critical_classes.items():
                    if class_id < len(metrics.box.maps):
                        map50 = metrics.box.maps[class_id]
                        logger.info(f"   {class_name:10s}: {map50:.2%}")

            return metrics
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return None

    def export_for_deployment(self, weights_path):
        """Export model for Raspberry Pi deployment"""
        logger.info("\n📦 Exporting model for deployment...")

        if not Path(weights_path).exists():
            logger.error(f"❌ Weights not found: {weights_path}")
            return False

        model = YOLO(weights_path)

        try:
            # Export to ONNX (universal format)
            logger.info("   Exporting to ONNX...")
            model.export(format="onnx", imgsz=self.imgsz, dynamic=False, simplify=True)

            # Export to TFLite (Raspberry Pi optimized)
            logger.info("   Exporting to TFLite (INT8)...")
            model.export(
                format="tflite",
                imgsz=self.imgsz,
                int8=True,  # INT8 quantization for speed
            )

            logger.info("\n✅ Export completed!")
            logger.info(f"   ONNX: {Path(weights_path).parent / 'best.onnx'}")
            logger.info(f"   TFLite: {Path(weights_path).parent / 'best_int8.tflite'}")

            return True
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False

    def benchmark_speed(self, weights_path):
        """Benchmark inference speed"""
        logger.info("\n⚡ Benchmarking inference speed...")

        if not Path(weights_path).exists():
            logger.error(f"❌ Weights not found: {weights_path}")
            return

        model = YOLO(weights_path)

        import cv2
        import time

        # Create dummy image
        dummy_img = np.random.randint(
            0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8
        )

        # Warm-up
        for _ in range(10):
            model(dummy_img, verbose=False)

        # Benchmark
        times = []
        for _ in range(50):
            start = time.time()
            model(dummy_img, verbose=False)
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000  # Convert to ms
        fps = 1000 / avg_time

        logger.info(f"   Average inference time: {avg_time:.1f} ms")
        logger.info(f"   FPS: {fps:.1f}")

        if self.deployment_target == "raspberry_pi":
            if fps < 10:
                logger.warning("⚠️ FPS below 10. Consider using TFLite export.")
            elif fps >= 15:
                logger.info("✅ FPS adequate for real-time navigation")


def main():
    """Main training workflow"""
    import argparse

    parser = argparse.ArgumentParser(description="MoveFree Training Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "train",
            "resume",
            "finetune",
            "validate",
            "export",
            "benchmark",
        ],  # Added "resume"
        default="train",
        help="Training mode",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["raspberry_pi", "jetson"],
        default="raspberry_pi",
        help="Deployment target",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights (for resume/finetune/validate/export/benchmark)",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = MoveFreeTrainer(deployment_target=args.target)

    if args.mode == "train":
        # First-time training
        results = trainer.train_from_scratch()

        if results:
            # Auto-validate
            weights_path = (
                f"runs/detect/movefree_indoor_{trainer.model_size}/weights/best.pt"
            )
            trainer.validate(weights_path)

            # Auto-export
            trainer.export_for_deployment(weights_path)
            trainer.benchmark_speed(weights_path)

    elif args.mode == "resume":
        # Resume interrupted training
        if args.weights:
            results = trainer.resume_training(args.weights)
        else:
            # Auto-detect last checkpoint
            results = trainer.resume_training()

        if results:
            weights_path = (
                f"runs/detect/movefree_indoor_{trainer.model_size}/weights/best.pt"
            )
            trainer.validate(weights_path)
            trainer.export_for_deployment(weights_path)

    elif args.mode == "finetune":
        if not args.weights:
            logger.error("❌ --weights required for fine-tuning")
            return

        results = trainer.fine_tune(args.weights)

        if results:
            weights_path = (
                f"runs/detect/movefree_finetune_{trainer.model_size}/weights/best.pt"
            )
            trainer.validate(weights_path)
            trainer.export_for_deployment(weights_path)

    elif args.mode == "validate":
        if not args.weights:
            logger.error("❌ --weights required for validation")
            return
        trainer.validate(args.weights)

    elif args.mode == "export":
        if not args.weights:
            logger.error("❌ --weights required for export")
            return
        trainer.export_for_deployment(args.weights)

    elif args.mode == "benchmark":
        if not args.weights:
            logger.error("❌ --weights required for benchmark")
            return
        trainer.benchmark_speed(args.weights)


if __name__ == "__main__":
    main()
