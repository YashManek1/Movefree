"""
Data augmentation for MoveFree training
Enhances dataset diversity for better generalization
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentation:
    def __init__(self):
        """Initialize augmentation pipeline"""
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.Perspective(p=0.3),
            A.HueSaturationValue(p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        self.val_transform = A.Compose([
            A.Resize(height=640, width=640),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def augment_image(self, image_path, label_path, output_dir, num_augmentations=3):
        """
        Augment a single image and its annotations
        
        Args:
            image_path: Path to input image
            label_path: Path to YOLO format label file
            output_dir: Directory to save augmented images
            num_augmentations: Number of augmented versions to create
        """
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read YOLO format labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        bboxes = []
        class_labels = []
        
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:]]
            class_labels.append(class_id)
            bboxes.append(bbox)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate augmented versions
        base_name = Path(image_path).stem
        
        for i in range(num_augmentations):
            try:
                # Apply augmentation
                augmented = self.train_transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Save augmented image
                aug_image = augmented['image']
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                
                output_image_path = Path(output_dir) / f"{base_name}_aug{i}.jpg"
                cv2.imwrite(str(output_image_path), aug_image_bgr)
                
                # Save augmented labels
                output_label_path = Path(output_dir) / f"{base_name}_aug{i}.txt"
                with open(output_label_path, 'w') as f:
                    for class_id, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                        f.write(f"{class_id} {' '.join(map(str, bbox))}\\n")
                
                logger.info(f"✅ Created augmentation {i+1}/{num_augmentations} for {base_name}")
                
            except Exception as e:
                logger.error(f"❌ Error augmenting {base_name}: {e}")
                continue


def main():
    """Main function for data augmentation"""
    augmentor = DataAugmentation()
    logger.info("✅ Data augmentation pipeline initialized")
    logger.info("Note: YOLO11 has built-in augmentation, so custom augmentation is optional")


if __name__ == "__main__":
    main()
