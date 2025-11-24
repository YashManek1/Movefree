"""
Dataset downloader for MoveFree Indoor Navigation
Downloads and prepares HomeObjects-3K dataset
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    def __init__(self, config_path="config/config.yaml"):
        """Initialize dataset downloader"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_name = self.config['dataset']['name']
        self.dataset_path = Path(self.config['dataset']['path'])
    
    def download_homeobjects_3k(self):
        """Download HomeObjects-3K dataset"""
        logger.info(f"📦 Downloading {self.dataset_name} dataset...")
        
        try:
            # Create dataset directory
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize YOLO to trigger dataset download
            model = YOLO('yolo11n.pt')
            
            # The dataset will auto-download on first training attempt
            # Or we can manually download it
            from ultralytics.data.utils import download
            
            dataset_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip"
            download(dataset_url, dir=str(self.dataset_path.parent))
            
            logger.info(f"✅ Dataset downloaded successfully to {self.dataset_path}")
            
            # Verify dataset structure
            self.verify_dataset()
            
        except Exception as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            raise
    
    def verify_dataset(self):
        """Verify dataset structure"""
        logger.info("🔍 Verifying dataset structure...")
        
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                logger.warning(f"⚠️ Missing directory: {dir_path}")
            else:
                num_files = len(list(dir_path.iterdir()))
                logger.info(f"✓ {dir_name}: {num_files} files")
        
        logger.info("✅ Dataset verification complete")
    
    def get_dataset_stats(self):
        """Get dataset statistics"""
        stats = {
            'train_images': len(list((self.dataset_path / 'images/train').glob('*'))),
            'val_images': len(list((self.dataset_path / 'images/val').glob('*'))),
            'classes': len(self.config['dataset']['classes'])
        }
        
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"   Training images: {stats['train_images']}")
        logger.info(f"   Validation images: {stats['val_images']}")
        logger.info(f"   Number of classes: {stats['classes']}")
        
        return stats


def main():
    """Main function to download dataset"""
    downloader = DatasetDownloader()
    downloader.download_homeobjects_3k()
    downloader.get_dataset_stats()


if __name__ == "__main__":
    main()
