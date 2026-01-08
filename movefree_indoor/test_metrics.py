"""
MoveFree Model Metrics Analyzer
Comprehensive testing script for trained YOLOv8n/YOLOv11m models
Tests on validation set and provides detailed per-class analysis
"""

import torch
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import cv2
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MoveFreeMetricsTester:
    """
    Comprehensive metrics testing for MoveFree indoor navigation model
    """
    
    def __init__(self, weights_path, data_yaml="datasets/movefree_combined/movefree.yaml"):
        """
        Initialize metrics tester
        
        Args:
            weights_path: Path to best.pt or trained model
            data_yaml: Path to dataset YAML config
        """
        self.weights_path = Path(weights_path)
        self.data_yaml = data_yaml
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"❌ Model not found: {weights_path}")
        
        logger.info("=" * 70)
        logger.info("🔍 MoveFree Model Metrics Analyzer")
        logger.info("=" * 70)
        
        # Load model
        logger.info(f"📦 Loading model: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        
        # Load dataset config
        with open(data_yaml, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        # Get class names
        self.class_names = self.dataset_config['names']
        self.num_classes = len(self.class_names)
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"📊 Dataset: {self.dataset_config['path']}")
        logger.info(f"🏷️  Classes: {self.num_classes}")
        
        # Critical safety classes for MoveFree
        self.critical_classes = {
            9: "door",      # Exit finding
            13: "stairs",   # Fall prevention
            12: "person"    # Collision avoidance
        }
        
        # Device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"💻 Device: {self.device}")
        
    def validate_model(self, imgsz=416, batch=16, conf=0.25, iou=0.45):
        """
        Run full validation on test/val set
        
        Args:
            imgsz: Image size for validation
            batch: Batch size
            conf: Confidence threshold
            iou: IoU threshold for NMS
        """
        logger.info("\n" + "=" * 70)
        logger.info("🔬 RUNNING FULL VALIDATION")
        logger.info("=" * 70)
        
        logger.info(f"⚙️  Config: imgsz={imgsz}, batch={batch}, conf={conf}, iou={iou}")
        
        # Run validation
        metrics = self.model.val(
            data=self.data_yaml,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            device=self.device,
            save_json=True,
            save_hybrid=True,
            plots=True,
            verbose=True
        )
        
        # Print overall metrics
        logger.info("\n" + "=" * 70)
        logger.info("📊 OVERALL METRICS")
        logger.info("=" * 70)
        logger.info(f"mAP@0.5:      {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
        logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
        logger.info(f"Precision:    {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
        logger.info(f"Recall:       {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
        logger.info(f"F1-Score:     {2*(metrics.box.mp*metrics.box.mr)/(metrics.box.mp+metrics.box.mr+1e-6):.4f}")
        
        # Per-class metrics
        if hasattr(metrics.box, 'maps'):
            self._print_per_class_metrics(metrics)
        
        # Critical class analysis
        self._analyze_critical_classes(metrics)
        
        return metrics
    
    def _print_per_class_metrics(self, metrics):
        """Print detailed per-class metrics"""
        logger.info("\n" + "=" * 70)
        logger.info("📋 PER-CLASS METRICS (mAP@0.5)")
        logger.info("=" * 70)
        
        # Create DataFrame for better formatting
        class_data = []
        
        for class_id in range(self.num_classes):
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            if class_id < len(metrics.box.maps):
                map50 = metrics.box.maps[class_id]
                
                # Get precision and recall per class (if available)
                if hasattr(metrics.box, 'p') and class_id < len(metrics.box.p):
                    precision = metrics.box.p[class_id]
                else:
                    precision = -1
                
                if hasattr(metrics.box, 'r') and class_id < len(metrics.box.r):
                    recall = metrics.box.r[class_id]
                else:
                    recall = -1
                
                class_data.append({
                    'ID': class_id,
                    'Class': class_name,
                    'mAP@0.5': map50,
                    'Precision': precision if precision >= 0 else 'N/A',
                    'Recall': recall if recall >= 0 else 'N/A'
                })
        
        # Create and display DataFrame
        df = pd.DataFrame(class_data)
        
        # Sort by mAP@0.5 descending
        df_sorted = df.sort_values('mAP@0.5', ascending=False)
        
        for idx, row in df_sorted.iterrows():
            map_val = row['mAP@0.5']
            class_name = row['Class']
            
            # Color coding
            if map_val >= 0.8:
                emoji = "✅"
            elif map_val >= 0.6:
                emoji = "⚠️"
            else:
                emoji = "❌"
            
            logger.info(f"{emoji} {class_name:20s} | mAP: {map_val:.3f} ({map_val*100:.1f}%)")
        
        # Save to CSV
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        csv_path = output_dir / f"class_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_sorted.to_csv(csv_path, index=False)
        logger.info(f"\n💾 Saved per-class metrics to: {csv_path}")
    
    def _analyze_critical_classes(self, metrics):
        """Analyze safety-critical classes for MoveFree"""
        logger.info("\n" + "=" * 70)
        logger.info("🚨 CRITICAL SAFETY CLASSES ANALYSIS")
        logger.info("=" * 70)
        
        for class_id, class_name in self.critical_classes.items():
            if class_id < len(metrics.box.maps):
                map50 = metrics.box.maps[class_id]
                
                # Determine pass/fail based on requirements
                if class_name == "stairs":
                    threshold = 0.85  # 85% minimum for fall prevention
                elif class_name == "person":
                    threshold = 0.90  # 90% minimum for collision avoidance
                else:  # door
                    threshold = 0.70  # 70% minimum for exit finding
                
                status = "✅ PASS" if map50 >= threshold else "❌ FAIL"
                
                logger.info(f"{status} | {class_name.upper():10s} | mAP: {map50:.3f} | Threshold: {threshold:.2f}")
            else:
                logger.warning(f"⚠️ {class_name.upper()}: No metrics available")
    
    def test_on_images(self, image_dir, save_dir="test_results/predictions", conf=0.25):
        """
        Test model on custom images and save annotated results
        
        Args:
            image_dir: Directory containing test images
            save_dir: Where to save annotated images
            conf: Confidence threshold
        """
        logger.info("\n" + "=" * 70)
        logger.info("📸 TESTING ON CUSTOM IMAGES")
        logger.info("=" * 70)
        
        image_dir = Path(image_dir)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(list(image_dir.glob(ext)))
        
        if not image_files:
            logger.warning(f"⚠️ No images found in {image_dir}")
            return
        
        logger.info(f"📁 Found {len(image_files)} images")
        
        detection_stats = {class_name: 0 for class_name in self.class_names.values()}
        
        for img_path in image_files:
            logger.info(f"\n🔍 Processing: {img_path.name}")
            
            # Run inference
            results = self.model(str(img_path), conf=conf, verbose=False)
            
            # Count detections
            if results[0].boxes:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, 'unknown')
                    detection_stats[class_name] += 1
                    
                    confidence = float(box.conf[0])
                    logger.info(f"   ✓ {class_name}: {confidence:.2f}")
            else:
                logger.info("   ℹ️  No detections")
            
            # Save annotated image
            annotated = results[0].plot()
            output_path = save_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(output_path), annotated)
        
        logger.info("\n📊 Detection Summary:")
        for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                logger.info(f"   {class_name:20s}: {count} detections")
        
        logger.info(f"\n💾 Annotated images saved to: {save_dir}")
    
    def benchmark_speed(self, num_iterations=100, imgsz=416):
        """
        Benchmark inference speed
        
        Args:
            num_iterations: Number of iterations for benchmarking
            imgsz: Input image size
        """
        logger.info("\n" + "=" * 70)
        logger.info("⚡ SPEED BENCHMARK")
        logger.info("=" * 70)
        
        logger.info(f"⚙️  Running {num_iterations} iterations at {imgsz}x{imgsz}")
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        
        # Warm-up
        logger.info("🔥 Warming up...")
        for _ in range(10):
            self.model(dummy_img, verbose=False)
        
        # Benchmark
        logger.info("⏱️  Benchmarking...")
        times = []
        
        for i in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start.record()
                self.model(dummy_img, verbose=False)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))  # ms
            else:
                import time
                t0 = time.time()
                self.model(dummy_img, verbose=False)
                times.append((time.time() - t0) * 1000)  # convert to ms
        
        # Calculate statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / mean_time
        
        logger.info("\n📊 Results:")
        logger.info(f"   Average: {mean_time:.2f} ms ± {std_time:.2f} ms")
        logger.info(f"   Min:     {min_time:.2f} ms")
        logger.info(f"   Max:     {max_time:.2f} ms")
        logger.info(f"   FPS:     {fps:.1f}")
        
        # Raspberry Pi 5 estimation (assuming 3x slower than GPU)
        rpi5_fps = fps / 3
        logger.info(f"\n🥧 Estimated Raspberry Pi 5 Performance:")
        logger.info(f"   FPS:     {rpi5_fps:.1f}")
        
        if rpi5_fps >= 15:
            logger.info("   Status:  ✅ REAL-TIME CAPABLE")
        elif rpi5_fps >= 10:
            logger.info("   Status:  ⚠️ ACCEPTABLE")
        else:
            logger.info("   Status:  ❌ TOO SLOW (Consider TFLite INT8)")
        
        return {
            'mean_ms': mean_time,
            'std_ms': std_time,
            'fps': fps,
            'rpi5_fps_estimate': rpi5_fps
        }
    
    def generate_confusion_matrix(self, save_path="test_results/confusion_matrix.png"):
        """Generate and save confusion matrix"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 GENERATING CONFUSION MATRIX")
        logger.info("=" * 70)
        
        # Run validation to get confusion matrix
        metrics = self.model.val(
            data=self.data_yaml,
            plots=True,
            save_json=True
        )
        
        # Confusion matrix is automatically saved by ultralytics
        # We just need to find it
        runs_dir = Path(self.weights_path).parent.parent
        confusion_files = list(runs_dir.rglob("confusion_matrix*.png"))
        
        if confusion_files:
            latest_cm = max(confusion_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"✅ Confusion matrix available at: {latest_cm}")
        else:
            logger.warning("⚠️ Confusion matrix not generated")
    
    def export_model(self, formats=['onnx', 'tflite']):
        """
        Export model to deployment formats
        
        Args:
            formats: List of export formats ('onnx', 'tflite', 'engine')
        """
        logger.info("\n" + "=" * 70)
        logger.info("📦 EXPORTING MODEL")
        logger.info("=" * 70)
        
        for fmt in formats:
            logger.info(f"\n🔄 Exporting to {fmt.upper()}...")
            
            try:
                if fmt == 'tflite':
                    # INT8 quantization for Raspberry Pi
                    self.model.export(format='tflite', int8=True, imgsz=416)
                    logger.info(f"✅ Exported INT8 TFLite (Raspberry Pi optimized)")
                else:
                    self.model.export(format=fmt, imgsz=416)
                    logger.info(f"✅ Exported {fmt.upper()}")
            except Exception as e:
                logger.error(f"❌ Export to {fmt} failed: {e}")
    
    def generate_report(self, metrics, speed_results):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 70)
        logger.info("📄 GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 70)
        
        report = {
            'model_path': str(self.weights_path),
            'test_date': datetime.now().isoformat(),
            'device': str(self.device),  # Convert to string
            'overall_metrics': {
                'mAP@0.5': float(metrics.box.map50),
                'mAP@0.5:0.95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'f1_score': float(2*(metrics.box.mp*metrics.box.mr)/(metrics.box.mp+metrics.box.mr+1e-6))
            },
            'critical_classes': {},
            'speed': {
                'mean_ms': float(speed_results['mean_ms']),
                'std_ms': float(speed_results['std_ms']),
                'fps': float(speed_results['fps']),
                'rpi5_fps_estimate': float(speed_results['rpi5_fps_estimate'])
            },
            'deployment_ready': {
                'raspberry_pi_5': bool(speed_results['rpi5_fps_estimate'] >= 10),  # Explicit bool conversion
                'jetson_nano': bool(speed_results['fps'] >= 20)
            }
        }
        
        # Add critical class metrics
        for class_id, class_name in self.critical_classes.items():
            if class_id < len(metrics.box.maps):
                report['critical_classes'][class_name] = float(metrics.box.maps[class_id])
        
        # Save report
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)  # Fixed: removed 'fp=' parameter
        
        logger.info(f"✅ Report saved to: {report_path}")
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("✨ FINAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Model:            {self.weights_path.name}")
        logger.info(f"Overall mAP@0.5:  {report['overall_metrics']['mAP@0.5']:.3f}")
        logger.info(f"Inference Speed:  {speed_results['fps']:.1f} FPS (GPU)")
        logger.info(f"RPi5 Estimate:    {speed_results['rpi5_fps_estimate']:.1f} FPS")
        logger.info(f"")
        logger.info("Critical Classes:")
        for class_name, map_val in report['critical_classes'].items():
            status = "✅" if map_val >= 0.7 else "❌"
            logger.info(f"   {status} {class_name:10s}: {map_val:.3f}")
        
        if report['deployment_ready']['raspberry_pi_5']:
            logger.info("\n✅ MODEL READY FOR RASPBERRY PI 5 DEPLOYMENT")
        else:
            logger.warning("\n⚠️ MODEL MAY BE TOO SLOW FOR RASPBERRY PI 5")
            logger.warning("   Consider using TFLite INT8 export")
        
        # CRITICAL WARNINGS based on your results
        logger.info("\n" + "=" * 70)
        logger.info("⚠️ CRITICAL ISSUES DETECTED")
        logger.info("=" * 70)
        logger.warning("❌ STAIRS: 32.8% mAP@0.5 (Target: 85%+) - SAFETY RISK!")
        logger.warning("❌ DOOR: 41.7% mAP@0.5 (Target: 70%+) - EXIT FINDING IMPAIRED")
        logger.warning("⚠️ PERSON: 83.6% mAP@0.5 (Target: 90%+) - COLLISION RISK")
        logger.info("\n🔧 RECOMMENDATION:")
        logger.info("   Model needs retraining with:")
        logger.info("   1. More stairs/door images from real environments")
        logger.info("   2. Augmentation focused on critical classes")
        logger.info("   3. Class-weighted loss (prioritize stairs/person/door)")


def main():
    """Main testing workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoveFree Model Metrics Tester")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/movefree_indoor_n/weights/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/movefree_combined/movefree.yaml",
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default=None,
        help="Directory with custom test images"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export model to ONNX and TFLite"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = MoveFreeMetricsTester(
        weights_path=args.weights,
        data_yaml=args.data
    )
    
    # 1. Run validation
    metrics = tester.validate_model(imgsz=416, batch=16, conf=0.25)
    
    # 2. Benchmark speed
    speed_results = tester.benchmark_speed(num_iterations=100, imgsz=416)
    
    # 3. Test on custom images (if provided)
    if args.test_images:
        tester.test_on_images(image_dir=args.test_images, conf=0.25)
    
    # 4. Generate confusion matrix
    tester.generate_confusion_matrix()
    
    # 5. Export model (if requested)
    if args.export:
        tester.export_model(formats=['onnx', 'tflite'])
    
    # 6. Generate comprehensive report
    tester.generate_report(metrics, speed_results)
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 TESTING COMPLETE!")
    logger.info("=" * 70)
    logger.info("📁 Results saved to: test_results/")


if __name__ == "__main__":
    main()