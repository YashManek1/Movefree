"""
Main training script for MoveFree Indoor Navigation
Trains YOLO11 on HomeObjects-3K dataset
"""

import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoveFreeTrainer:
    def __init__(self, config_path="config/config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['name']
        self.training_config = self.config['training']
        self.dataset_config = self.config['dataset']
        
        # Setup device
        self.device = self.training_config['device']
        if self.device == 0:
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA not available, falling back to CPU")
                self.device = 'cpu'
        
        logger.info(f"🚀 MoveFree Trainer initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🎯 Model: {self.model_name}")
    
    def train(self):
        """Train YOLO11 model"""
        logger.info("=" * 60)
        logger.info("🏋️ Starting Training")
        logger.info("=" * 60)
        
        try:
            # Load pretrained model
            model = YOLO(self.model_name)
            logger.info(f"✅ Loaded pretrained model: {self.model_name}")
            
            # Train the model
            results = model.train(
                data=self.dataset_config['yaml'],
                epochs=self.training_config['epochs'],
                imgsz=self.config['model']['imgsz'],
                batch=self.training_config['batch'],
                device=self.device,
                workers=self.training_config['workers'],
                patience=self.training_config['patience'],
                save_period=self.training_config['save_period'],
                project='runs/detect',
                name='movefree_indoor',
                exist_ok=True,
                pretrained=True,
                optimizer=self.training_config['optimizer'],
                lr0=self.training_config['lr0'],
                lrf=self.training_config['lrf'],
                momentum=self.training_config['momentum'],
                weight_decay=self.training_config['weight_decay'],
                warmup_epochs=self.training_config['warmup_epochs'],
                augment=self.training_config['augmentation'],
                verbose=True,
                seed=42,
                deterministic=True,
            )
            
            logger.info("=" * 60)
            logger.info("✅ Training completed successfully!")
            logger.info("=" * 60)
            
            # Print training summary
            self.print_training_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def print_training_summary(self, results):
        """Print training summary"""
        logger.info("📊 Training Summary:")
        logger.info(f"   Best epoch: {results.best_epoch if hasattr(results, 'best_epoch') else 'N/A'}")
        logger.info(f"   Model saved to: runs/detect/movefree_indoor/weights/best.pt")
        logger.info(f"   Metrics available in: runs/detect/movefree_indoor/")
    
    def resume_training(self, checkpoint_path):
        """Resume training from checkpoint"""
        logger.info(f"🔄 Resuming training from {checkpoint_path}")
        
        model = YOLO(checkpoint_path)
        results = model.train(
            resume=True,
            device=self.device
        )
        
        return results
    
    def validate(self, weights_path="runs/detect/movefree_indoor/weights/best.pt"):
        """Validate trained model"""
        logger.info("🔍 Running validation...")
        
        model = YOLO(weights_path)
        metrics = model.val(
            data=self.dataset_config['yaml'],
            imgsz=self.config['model']['imgsz'],
            batch=self.training_config['batch'],
            device=self.device
        )
        
        logger.info("=" * 60)
        logger.info("📊 Validation Results:")
        logger.info(f"   mAP50: {metrics.box.map50:.4f}")
        logger.info(f"   mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"   Precision: {metrics.box.mp:.4f}")
        logger.info(f"   Recall: {metrics.box.mr:.4f}")
        logger.info("=" * 60)
        
        return metrics


def main():
    """Main training function"""
    # Initialize trainer
    trainer = MoveFreeTrainer()
    
    # Train model
    results = trainer.train()
    
    # Validate model
    trainer.validate()
    
    logger.info("🎉 Training pipeline completed!")


if __name__ == "__main__":
    main()
