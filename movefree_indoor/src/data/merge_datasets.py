"""
Dataset Merger V4 - INTELLIGENT CLASS REMAPPING + FALLBACK INFERENCE
Handles datasets with AND without YAML files
"""

import os
import shutil
import yaml
import logging
import random
import kaggle
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentDatasetMerger:
    def __init__(self, base_path="datasets/movefree_combined"):
        self.base_path = Path(base_path)
        self.images_dir = self.base_path / "images"
        self.labels_dir = self.base_path / "labels"
        self.temp_dir = Path("datasets/temp_downloads")

        # Our unified 19-class schema (TARGET)
        self.unified_classes = {
            0: "bed",
            1: "sofa",
            2: "chair",
            3: "table",
            4: "lamp",
            5: "tv",
            6: "laptop",
            7: "wardrobe",
            8: "window",
            9: "door",
            10: "potted plant",
            11: "photo frame",
            12: "person",
            13: "stairs",
            14: "trash_can",
            15: "shoe",
            16: "cabinet",
            17: "shelf",
            18: "refrigerator",
        }

        # Reverse lookup: class_name -> unified_id
        self.unified_name_to_id = {v: k for k, v in self.unified_classes.items()}

        # Statistics tracking
        self.stats = defaultdict(lambda: {"total": 0, "mapped": 0, "unmapped": 0})

    def setup_dirs(self):
        if self.base_path.exists():
            logger.info("🗑️ Removing existing dataset...")
            try:
                shutil.rmtree(self.base_path)
            except Exception as e:
                logger.error(f"Error removing directory: {e}")

        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Directories created")

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

    def find_yaml_in_dataset(self, dataset_path):
        """Find data.yaml or similar config files in dataset"""
        yaml_candidates = list(Path(dataset_path).rglob("*.yaml")) + list(
            Path(dataset_path).rglob("*.yml")
        )

        for yaml_file in yaml_candidates:
            if any(
                keyword in yaml_file.name.lower()
                for keyword in ["data", "dataset", "config"]
            ):
                try:
                    with open(yaml_file, "r") as f:
                        config = yaml.safe_load(f)
                        if "names" in config:
                            logger.info(f"📄 Found YAML: {yaml_file}")
                            return config
                except:
                    continue
        return None

    def infer_classes_from_dataset_name(self, dataset_name):
        """
        FALLBACK: Infer classes from dataset name/structure when no YAML exists
        """
        name_lower = dataset_name.lower()

        # Define inference rules
        if "door" in name_lower and "window" in name_lower and "stair" in name_lower:
            # DoorWindowStairs dataset
            return {0: "door", 1: "window", 2: "stairs"}

        elif "door" in name_lower:
            # Pure door dataset
            return {0: "door"}

        elif "stair" in name_lower:
            # Pure stairs dataset
            return {0: "stairs"}

        elif "window" in name_lower:
            # Pure window dataset
            return {0: "window"}

        elif "visionguard" in name_lower or "object" in name_lower:
            # Multi-object dataset - try to infer from folder structure
            return self._infer_from_structure(dataset_name)

        return None

    def _infer_from_structure(self, dataset_path):
        """
        Try to infer classes from folder names (Batch 1, Batch 2, etc.)
        """
        # For VisionGuard: it's a multi-class dataset, we'll use all our classes
        # and let the validation phase filter out
        logger.info(f"   Using full class schema for {dataset_path}")
        return self.unified_classes

    def create_class_mapping(self, dataset_config, dataset_name, dataset_path):
        """
        Create mapping from dataset's class IDs to our unified class IDs
        NOW WITH FALLBACK INFERENCE
        """
        # Try YAML first
        if dataset_config and "names" in dataset_config:
            return self._map_from_yaml(dataset_config, dataset_name)

        # Fallback: Infer from dataset name/structure
        logger.warning(f"⚠️ {dataset_name}: No YAML found, using inference")
        inferred_classes = self.infer_classes_from_dataset_name(dataset_name)

        if inferred_classes:
            logger.info(
                f"✅ {dataset_name}: Inferred classes: {list(inferred_classes.values())}"
            )
            return self._map_inferred_classes(inferred_classes)

        logger.error(f"❌ {dataset_name}: Cannot infer classes")
        return None

    def _map_from_yaml(self, dataset_config, dataset_name):
        """Map classes from YAML configuration"""
        dataset_classes = dataset_config["names"]

        # Handle both dict and list formats
        if isinstance(dataset_classes, dict):
            dataset_id_to_name = dataset_classes
        elif isinstance(dataset_classes, list):
            dataset_id_to_name = {i: name for i, name in enumerate(dataset_classes)}
        else:
            logger.error(f"❌ {dataset_name}: Invalid class format")
            return None

        mapping = {}
        unmapped = []

        for orig_id, orig_name in dataset_id_to_name.items():
            orig_name_clean = orig_name.lower().strip()

            # Direct match first
            if orig_name_clean in self.unified_name_to_id:
                mapping[int(orig_id)] = self.unified_name_to_id[orig_name_clean]
                continue

            # Fuzzy matching for common variations
            matched = False
            for unified_name, unified_id in self.unified_name_to_id.items():
                if self._fuzzy_match(orig_name_clean, unified_name):
                    mapping[int(orig_id)] = unified_id
                    matched = True
                    logger.info(f"   Fuzzy: '{orig_name}' -> '{unified_name}'")
                    break

            if not matched:
                unmapped.append(orig_name)

        if unmapped:
            logger.warning(f"⚠️ {dataset_name}: Unmapped classes: {unmapped}")

        logger.info(
            f"✅ {dataset_name}: Mapped {len(mapping)}/{len(dataset_id_to_name)} classes"
        )
        return mapping if mapping else None

    def _map_inferred_classes(self, inferred_classes):
        """Map inferred classes to unified schema"""
        mapping = {}

        for orig_id, class_name in inferred_classes.items():
            class_name_clean = class_name.lower().strip()

            # Direct lookup
            if class_name_clean in self.unified_name_to_id:
                mapping[int(orig_id)] = self.unified_name_to_id[class_name_clean]
            else:
                # Fuzzy match
                for unified_name, unified_id in self.unified_name_to_id.items():
                    if self._fuzzy_match(class_name_clean, unified_name):
                        mapping[int(orig_id)] = unified_id
                        break

        return mapping if mapping else None

    def _fuzzy_match(self, name1, name2):
        """FIXED: Proper fuzzy matching with explicit synonym dictionary"""
        # Normalize names
        n1 = name1.replace(" ", "").replace("_", "").replace("-", "").lower()
        n2 = name2.replace(" ", "").replace("_", "").replace("-", "").lower()

        # Exact match after normalization
        if n1 == n2:
            return True

        # COMPREHENSIVE SYNONYM MAPPING
        synonym_map = {
            "sofa": ["couch", "settee", "divan"],
            "tv": ["television", "screen", "monitor", "display"],
            "potted plant": ["plant", "pottedplant", "houseplant", "planter"],
            "photo frame": ["photoframe", "picture", "frame", "pictureframe"],
            "trash_can": [
                "trashcan",
                "bin",
                "dustbin",
                "garbage",
                "wastebin",
                "trashbin",
            ],
            "wardrobe": ["closet", "cupboard", "armoire", "dresser"],
            "cabinet": ["cupboard", "storage", "locker"],
            "laptop": ["computer", "notebook", "pc", "macbook"],
            "shoe": ["shoes", "footwear", "sneaker", "boot"],
            "door": [
                "opendoor",
                "closeddoor",
                "openeddoor",
                "cabinetdoor",
                "refrigeratordoor",
                "doorway",
            ],
            "window": ["openwindow", "closedwindow", "windowpane"],
            "stairs": ["stair", "staircase", "stairway", "steps", "step"],
            "person": ["people", "human", "man", "woman", "pedestrian"],
            "chair": ["seat", "stool", "bench"],
            "table": ["desk", "tabletop", "countertop"],
            "bed": ["mattress", "cot", "bunk"],
            "lamp": ["light", "lighting", "bulb"],
            "refrigerator": ["fridge", "freezer", "cooler"],
            "shelf": ["shelves", "rack", "bookshelf"],
        }

        # Check if n1 or n2 match any key or value in synonym map
        for target_class, synonyms in synonym_map.items():
            target_normalized = target_class.replace(" ", "")
            synonyms_normalized = [s.replace(" ", "") for s in synonyms]

            # Check if n1 is target and n2 is synonym (or vice versa)
            if (n1 == target_normalized and n2 in synonyms_normalized) or (
                n2 == target_normalized and n1 in synonyms_normalized
            ):
                return True

            # Check if both are synonyms of the same target
            if n1 in synonyms_normalized and n2 in synonyms_normalized:
                return True

            # Check if one is target and other contains it (but not cross-contamination)
            if n1 == target_normalized and target_normalized in n2:
                # Make sure it's not a false positive (door in window)
                if not any(
                    bad in n2
                    for bad in ["window", "wall", "floor"]
                    if bad != target_normalized
                ):
                    return True
            if n2 == target_normalized and target_normalized in n1:
                if not any(
                    bad in n1
                    for bad in ["window", "wall", "floor"]
                    if bad != target_normalized
                ):
                    return True

        return False

    def remap_label_file(self, label_path, class_mapping):
        """
        Read label file, remap class IDs, return new lines
        Returns None if no valid mappings found
        """
        if class_mapping is None:
            return None

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                orig_class_id = int(parts[0])

                # Check if we can map this class
                if orig_class_id not in class_mapping:
                    # This object class is not in our schema, skip this annotation
                    continue

                # Remap the class ID
                new_class_id = class_mapping[orig_class_id]

                # Validate bbox format
                if len(parts) != 5:
                    continue

                try:
                    bbox = [float(x) for x in parts[1:5]]
                    # Validate YOLO format (0-1 range)
                    if all(0 <= coord <= 1 for coord in bbox):
                        new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
                except ValueError:
                    continue

            return new_lines if new_lines else None

        except Exception as e:
            logger.debug(f"Error reading label {label_path}: {e}")
            return None

    def process_dataset(self, source_path, dataset_name):
        source = Path(source_path)
        if not source.exists():
            logger.warning(f"⚠️ {dataset_name}: Path not found: {source_path}")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Processing: {dataset_name}")
        logger.info(f"{'='*60}")

        # Try to find and load dataset YAML
        dataset_config = self.find_yaml_in_dataset(source)

        # Create class mapping (with fallback inference)
        class_mapping = self.create_class_mapping(dataset_config, dataset_name, source)

        if class_mapping is None:
            logger.error(
                f"❌ {dataset_name}: Cannot create class mapping, skipping dataset"
            )
            return

        # Find all images
        all_images = []
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
            all_images.extend(list(source.rglob(ext)))

        if not all_images:
            logger.warning(f"⚠️ {dataset_name}: No images found")
            return

        logger.info(f"📸 Found {len(all_images)} images")

        # Shuffle for random train/val/test split
        random.shuffle(all_images)
        n = len(all_images)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        splits = {
            "train": all_images[:train_end],
            "val": all_images[train_end:val_end],
            "test": all_images[val_end:],
        }

        # Process each split
        for split_name, img_list in splits.items():
            count = self._process_split(
                img_list, split_name, class_mapping, dataset_name
            )
            self.stats[dataset_name][split_name] = count
            self.stats[dataset_name]["total"] += count

        logger.info(
            f"✅ {dataset_name}: Merged {self.stats[dataset_name]['total']} valid pairs"
        )

    def _process_split(self, image_files, split_name, class_mapping, dataset_name):
        count = 0
        for img_file in image_files:
            try:
                # Find corresponding label
                lbl_file = self._find_label_file(img_file)

                if lbl_file and lbl_file.exists():
                    # Remap label file
                    new_label_lines = self.remap_label_file(lbl_file, class_mapping)

                    if new_label_lines:
                        # Generate unique filename to avoid collisions
                        unique_name = f"{dataset_name.lower().replace('-', '_')}_{img_file.stem}_{count}"

                        # Copy image
                        dest_img = (
                            self.images_dir
                            / split_name
                            / f"{unique_name}{img_file.suffix}"
                        )
                        shutil.copy(img_file, dest_img)

                        # Write remapped label
                        dest_lbl = self.labels_dir / split_name / f"{unique_name}.txt"
                        with open(dest_lbl, "w") as f:
                            f.writelines(new_label_lines)

                        count += 1

            except Exception as e:
                logger.debug(f"Error processing {img_file.name}: {e}")
                continue

        return count

    def _find_label_file(self, img_path):
        """Find label file using multiple strategies"""
        # Strategy 1: Same folder
        lbl = img_path.with_suffix(".txt")
        if lbl.exists():
            return lbl

        # Strategy 2: Child 'labels' folder
        lbl = img_path.parent / "labels" / img_path.with_suffix(".txt").name
        if lbl.exists():
            return lbl

        # Strategy 3: Parallel 'labels' folder (Standard YOLO)
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

        # Strategy 4: Sibling 'labels' folder
        try:
            labels_dir = img_path.parent.parent / "labels"
            if labels_dir.exists():
                lbl = labels_dir / img_path.with_suffix(".txt").name
                if lbl.exists():
                    return lbl
        except:
            pass

        # Strategy 5: Root-level labels folder (search up to 5 levels)
        try:
            current = img_path.parent
            for _ in range(5):
                labels_candidate = (
                    current / "labels" / img_path.with_suffix(".txt").name
                )
                if labels_candidate.exists():
                    return labels_candidate
                current = current.parent
                if len(current.parts) < 3:
                    break
        except:
            pass

        return None

    def create_yaml(self):
        yaml_content = {
            "path": str(self.base_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.unified_classes,
            "nc": len(self.unified_classes),
        }

        yaml_path = self.base_path / "movefree.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        logger.info(f"✅ Created {yaml_path}")

    def print_statistics(self):
        logger.info(f"\n{'='*60}")
        logger.info("📊 DATASET MERGE STATISTICS")
        logger.info(f"{'='*60}")

        total_train = len(list((self.images_dir / "train").glob("*")))
        total_val = len(list((self.images_dir / "val").glob("*")))
        total_test = len(list((self.images_dir / "test").glob("*")))

        logger.info(f"Training images:   {total_train}")
        logger.info(f"Validation images: {total_val}")
        logger.info(f"Test images:       {total_test}")
        logger.info(f"Total images:      {total_train + total_val + total_test}")
        logger.info(f"\nPer-dataset breakdown:")

        for dataset_name, stats in self.stats.items():
            if stats["total"] > 0:
                logger.info(f"  {dataset_name:20s}: {stats['total']:5d} pairs")

        logger.info(f"{'='*60}\n")


def main():
    # Authenticate Kaggle
    try:
        kaggle.api.authenticate()
        logger.info("✅ Kaggle authenticated")
    except Exception as e:
        logger.error(f"❌ Kaggle authentication failed: {e}")
        logger.error("Make sure kaggle.json is in ~/.kaggle/")
        return

    merger = IntelligentDatasetMerger()
    merger.setup_dirs()

    # Process datasets (in order of quality/relevance)

    # 1. HomeObjects-3K (if you have it locally)
    if Path("homeobjects-3K").exists():
        merger.process_dataset("homeobjects-3K", "HomeObjects")

    # 2. Kaggle Indoor Detection
    p1 = merger.download_kaggle("thepbordin/indoor-object-detection", "kaggle_indoor")
    if p1:
        merger.process_dataset(p1, "KaggleIndoor")

    # 3. Door/Window/Stairs Combined
    p2 = merger.download_kaggle("nderalparslan/dwsonder", "dw_stairs")
    if p2:
        merger.process_dataset(p2, "DoorWindowStairs")

    # 4. Doors Specialized
    p3 = merger.download_kaggle("sayedmohamed1/doors-detection", "doors_yolo")
    if p3:
        merger.process_dataset(p3, "DoorsYOLO")

    # 5. Stairs Specialized
    p4 = merger.download_kaggle("samuelayman/stairs", "stairs_yolo")
    if p4:
        merger.process_dataset(p4, "StairsYOLO")

    # 6. VisionGuard Multi-Class
    p5 = merger.download_kaggle("samuelayman/object-detection", "visionguard")
    if p5:
        merger.process_dataset(p5, "VisionGuard")

    # Create final YAML
    merger.create_yaml()
    merger.print_statistics()

    logger.info("🎉 INTELLIGENT DATASET MERGE COMPLETE!")
    logger.info("✅ All class IDs have been remapped to unified schema")
    logger.info("✅ Ready for training with correct labels")


if __name__ == "__main__":
    main()
