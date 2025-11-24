"""
Dataset Merger - ULTIMATE FIX (Handles Child Label Folders)
"""

import os
import shutil
import yaml
import logging
import random
import kaggle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetMerger:
    def __init__(self, base_path="datasets/movefree_combined"):
        self.base_path = Path(base_path)
        self.images_dir = self.base_path / "images"
        self.labels_dir = self.base_path / "labels"
        self.temp_dir = Path("datasets/temp_downloads")

        self.classes = [
            "bed",
            "sofa",
            "chair",
            "table",
            "lamp",
            "tv",
            "laptop",
            "wardrobe",
            "window",
            "door",
            "potted plant",
            "photo frame",
            "person",
            "stairs",
            "trash_can",
            "shoe",
            "cabinet",
            "shelf",
            "refrigerator",
        ]

    def setup_dirs(self):
        if self.base_path.exists():
            try:
                shutil.rmtree(self.base_path)
            except:
                pass

        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Cleaned directories")

    def download_kaggle(self, dataset_slug, folder_name):
        dest = self.temp_dir / folder_name
        if dest.exists():
            logger.info(f"✅ {folder_name} exists, skipping download")
            return dest

        logger.info(f"⬇️ Downloading {dataset_slug}...")
        try:
            kaggle.api.dataset_download_files(dataset_slug, path=dest, unzip=True)
            logger.info(f"✅ Downloaded {dataset_slug}")
            return dest
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_slug}: {e}")
            return None

    def split_and_copy(self, image_files, label_path_func, dest_split):
        count = 0
        for img_file in image_files:
            try:
                lbl_file = label_path_func(img_file)
                if lbl_file and lbl_file.exists():
                    shutil.copy(img_file, self.images_dir / dest_split / img_file.name)
                    shutil.copy(lbl_file, self.labels_dir / dest_split / lbl_file.name)
                    count += 1
            except:
                pass
        return count

    def process_dataset(self, source_path, name):
        source = Path(source_path)
        if not source.exists():
            return

        logger.info(f"🔄 Merging {name}...")
        all_images = []
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG"]:
            all_images.extend(list(source.rglob(ext)))

        if not all_images:
            logger.warning(f"⚠️ No images found in {name}")
            return

        random.shuffle(all_images)
        n = len(all_images)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_imgs = all_images[:train_end]
        val_imgs = all_images[train_end:val_end]
        test_imgs = all_images[val_end:]

        def get_label_path(img_path):
            # 1. Same folder (Basic)
            lbl = img_path.with_suffix(".txt")
            if lbl.exists():
                return lbl

            # 2. Child 'labels' folder (Stairs YOLO Fix)
            # e.g. .../final stairs/img.jpg -> .../final stairs/labels/img.txt
            try:
                lbl = img_path.parent / "labels" / img_path.with_suffix(".txt").name
                if lbl.exists():
                    return lbl
            except:
                pass

            # 3. Parallel 'labels' folder (Standard YOLO)
            try:
                parts = list(img_path.parts)
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i].lower() == "images":
                        parts[i] = "labels"
                        lbl = Path(*parts).with_suffix(".txt")
                        if lbl.exists():
                            return lbl
                        break
            except:
                pass

            # 4. VisionGuard Batch Fix (Parent's Parent)
            try:
                batch_root = img_path.parent.parent
                lbl = batch_root / "labels" / img_path.with_suffix(".txt").name
                if lbl.exists():
                    return lbl
            except:
                pass

            return None

        c1 = self.split_and_copy(train_imgs, get_label_path, "train")
        c2 = self.split_and_copy(val_imgs, get_label_path, "val")
        c3 = self.split_and_copy(test_imgs, get_label_path, "test")

        logger.info(f"   Merged {c1+c2+c3} valid pairs from {name}")

    def create_yaml(self):
        yaml_content = {
            "path": str(self.base_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(self.classes)},
        }
        with open(self.base_path / "movefree.yaml", "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)


def main():
    try:
        kaggle.api.authenticate()
    except:
        pass

    merger = DatasetMerger()
    merger.setup_dirs()

    # 1. HomeObjects
    merger.process_dataset("homeobjects-3K", "HomeObjects")

    # 2. Kaggle Indoor
    p1 = merger.download_kaggle("thepbordin/indoor-object-detection", "kaggle_indoor")
    if p1:
        merger.process_dataset(p1, "Kaggle-Indoor")

    # 3. Door/Window/Stairs
    p3 = merger.download_kaggle("nderalparslan/dwsonder", "dw_stairs")
    if p3:
        merger.process_dataset(p3, "DoorWindowStairs")

    # 4. Doors
    p4 = merger.download_kaggle("sayedmohamed1/doors-detection", "doors_yolo")
    if p4:
        merger.process_dataset(p4, "DoorsYOLO")

    # 5. Stairs (FIXED)
    p5 = merger.download_kaggle("samuelayman/stairs", "stairs_yolo")
    if p5:
        merger.process_dataset(p5, "StairsYOLO")

    # 6. VisionGuard
    p6 = merger.download_kaggle("samuelayman/object-detection", "visionguard")
    if p6:
        merger.process_dataset(p6, "VisionGuard")

    merger.create_yaml()
    logger.info("🎉 ALL DATASETS MERGED!")


if __name__ == "__main__":
    main()
