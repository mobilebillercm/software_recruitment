"""
Test script for the Computer Vision recruitment task.

Validates the MarkerDetector implementation against generated test images.

Usage:
    1. First generate test images:  python3 generate_test_images.py
    2. Then run tests:              python3 test_detector.py
"""

import json
import os
import sys

import cv2
import numpy as np

from marker_detector import MarkerDetector, compute_iou, filter_detections

TEST_DIR = "test_images"
BBOX_TOLERANCE = 30  # pixels of tolerance for bounding box matching
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def load_test_case(name):
    img_path = os.path.join(TEST_DIR, f"{name}.png")
    gt_path = os.path.join(TEST_DIR, f"{name}.json")
    if not os.path.exists(img_path) or not os.path.exists(gt_path):
        return None, None
    img = cv2.imread(img_path)
    with open(gt_path) as f:
        gt = json.load(f)
    return img, gt


def bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def match_detection_to_gt(det, gt_list, tolerance=BBOX_TOLERANCE):
    """Find best matching ground truth entry for a detection."""
    det_cx, det_cy = det["center"]
    best_dist = float("inf")
    best_idx = -1
    for i, gt in enumerate(gt_list):
        gt_cx, gt_cy = bbox_center(gt["bbox"])
        dist = ((det_cx - gt_cx) ** 2 + (det_cy - gt_cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_dist <= tolerance:
        return best_idx
    return -1


def test_iou():
    """Test the compute_iou function."""
    print("\n--- Testing compute_iou ---")
    passed = 0
    total = 0

    # Test 1: identical boxes
    total += 1
    iou = compute_iou((10, 10, 50, 50), (10, 10, 50, 50))
    if iou is not None and abs(iou - 1.0) < 0.01:
        print(f"  {PASS} Identical boxes: IoU = {iou:.3f}")
        passed += 1
    else:
        print(f"  {FAIL} Identical boxes: expected 1.0, got {iou}")

    # Test 2: no overlap
    total += 1
    iou = compute_iou((0, 0, 10, 10), (100, 100, 10, 10))
    if iou is not None and abs(iou - 0.0) < 0.01:
        print(f"  {PASS} No overlap: IoU = {iou:.3f}")
        passed += 1
    else:
        print(f"  {FAIL} No overlap: expected 0.0, got {iou}")

    # Test 3: partial overlap
    total += 1
    iou = compute_iou((0, 0, 20, 20), (10, 10, 20, 20))
    expected = 100.0 / 700.0  # intersection = 10*10, union = 400 + 400 - 100
    if iou is not None and abs(iou - expected) < 0.02:
        print(f"  {PASS} Partial overlap: IoU = {iou:.3f} (expected {expected:.3f})")
        passed += 1
    else:
        print(f"  {FAIL} Partial overlap: expected {expected:.3f}, got {iou}")

    # Test 4: one box inside another
    total += 1
    iou = compute_iou((0, 0, 100, 100), (25, 25, 50, 50))
    expected = 2500.0 / 10000.0  # intersection = 50*50, union = 10000 + 2500 - 2500
    if iou is not None and abs(iou - expected) < 0.02:
        print(f"  {PASS} Box inside box: IoU = {iou:.3f} (expected {expected:.3f})")
        passed += 1
    else:
        print(f"  {FAIL} Box inside box: expected {expected:.3f}, got {iou}")

    return passed, total


def test_filter_detections():
    """Test the filter_detections function."""
    print("\n--- Testing filter_detections ---")
    passed = 0
    total = 0

    # Test 1: no overlapping detections — all should be kept
    total += 1
    dets = [
        {"color": "red", "bbox": (0, 0, 50, 50), "center": (25, 25), "area": 2000},
        {"color": "blue", "bbox": (200, 200, 50, 50), "center": (225, 225), "area": 1800},
    ]
    result = filter_detections(dets, iou_threshold=0.5)
    if result is not None and len(result) == 2:
        print(f"  {PASS} No overlap: kept {len(result)} detections")
        passed += 1
    else:
        count = len(result) if result else 0
        print(f"  {FAIL} No overlap: expected 2 detections, got {count}")

    # Test 2: fully overlapping — should keep the larger one
    total += 1
    dets = [
        {"color": "red", "bbox": (10, 10, 80, 80), "center": (50, 50), "area": 5000},
        {"color": "red", "bbox": (15, 15, 60, 60), "center": (45, 45), "area": 3000},
    ]
    result = filter_detections(dets, iou_threshold=0.3)
    if result is not None and len(result) == 1 and result[0]["area"] == 5000:
        print(f"  {PASS} Overlapping: kept larger detection (area={result[0]['area']})")
        passed += 1
    else:
        count = len(result) if result else 0
        print(f"  {FAIL} Overlapping: expected 1 detection with area 5000, got {count}")

    return passed, total


def test_detection(name, min_expected):
    """Test marker detection on a specific test image."""
    print(f"\n--- Testing {name} ---")
    img, gt = load_test_case(name)
    if img is None:
        print(f"  {FAIL} Test image not found. Run 'python3 generate_test_images.py' first.")
        return 0, 1

    detector = MarkerDetector()
    detections = detector.detect(img)

    passed = 0
    total = 0

    if detections is None:
        print(f"  {FAIL} detect() returned None — not yet implemented?")
        return 0, 3

    # Test: at least the minimum expected number of detections found
    total += 1
    if len(detections) >= min_expected:
        print(f"  {PASS} Found {len(detections)} detections (expected >= {min_expected})")
        passed += 1
    else:
        print(f"  {FAIL} Found {len(detections)} detections (expected >= {min_expected})")

    # Test: each detection has required keys
    total += 1
    required_keys = {"color", "bbox", "center", "area"}
    all_valid = True
    for i, det in enumerate(detections):
        missing = required_keys - set(det.keys())
        if missing:
            print(f"  {FAIL} Detection {i} missing keys: {missing}")
            all_valid = False
            break
    if all_valid:
        print(f"  {PASS} All detections have required keys")
        passed += 1

    # Test: color accuracy
    total += 1
    gt_colors = sorted([g["color"] for g in gt])
    det_colors = sorted([d["color"] for d in detections if "color" in d])

    # Count how many ground truth colors are found in detections
    gt_color_counts = {}
    for c in gt_colors:
        gt_color_counts[c] = gt_color_counts.get(c, 0) + 1
    det_color_counts = {}
    for c in det_colors:
        det_color_counts[c] = det_color_counts.get(c, 0) + 1

    color_matches = 0
    for color, count in gt_color_counts.items():
        color_matches += min(count, det_color_counts.get(color, 0))

    if color_matches >= len(gt_colors) * 0.7:  # 70% of colors correct
        print(f"  {PASS} Color accuracy: {color_matches}/{len(gt_colors)} correct")
        passed += 1
    else:
        print(f"  {FAIL} Color accuracy: {color_matches}/{len(gt_colors)} correct")
        print(f"         Expected colors: {gt_colors}")
        print(f"         Detected colors: {det_colors}")

    return passed, total


def main():
    if not os.path.exists(TEST_DIR):
        print(f"Error: '{TEST_DIR}/' directory not found.")
        print("Run 'python3 generate_test_images.py' first to generate test images.")
        sys.exit(1)

    total_passed = 0
    total_tests = 0

    # IoU tests
    p, t = test_iou()
    total_passed += p
    total_tests += t

    # NMS tests
    p, t = test_filter_detections()
    total_passed += p
    total_tests += t

    # Detection tests
    test_cases = [
        ("test_01_single_red_circle", 1),
        ("test_02_single_blue_rectangle", 1),
        ("test_03_multiple_shapes", 4),  # at least 4 of the 5 shapes
        ("test_05_noisy_scene", 2),  # at least 2 of the 3 shapes (noisy)
    ]

    for name, min_expected in test_cases:
        p, t = test_detection(name, min_expected)
        total_passed += p
        total_tests += t

    # Summary
    print(f"\n{'='*50}")
    if total_passed == total_tests:
        print(f"  All tests passed! ({total_passed}/{total_tests})")
    else:
        print(f"  {total_passed}/{total_tests} tests passed")
        print(f"  {total_tests - total_passed} tests failed")
    print(f"{'='*50}")

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
