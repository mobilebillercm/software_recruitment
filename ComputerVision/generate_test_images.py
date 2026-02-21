"""
Test image generator for the Computer Vision recruitment task.

Generates synthetic images with colored geometric markers on noisy backgrounds.
Each image is saved alongside a JSON ground truth file.

Usage:
    python3 generate_test_images.py
"""

import json
import os
import random

import cv2
import numpy as np

OUTPUT_DIR = "test_images"

# Colors in BGR format
COLORS_BGR = {
    "red": (0, 0, 220),
    "green": (0, 200, 0),
    "blue": (220, 0, 0),
    "yellow": (0, 220, 220),
}


def noisy_background(h, w):
    """Create a gray background with some Gaussian noise."""
    bg = np.full((h, w, 3), 180, dtype=np.uint8)
    noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def draw_circle(img, cx, cy, r, color_bgr):
    cv2.circle(img, (cx, cy), r, color_bgr, -1)
    return {"shape": "circle", "center": (cx, cy), "bbox": (cx - r, cy - r, 2 * r, 2 * r)}


def draw_triangle(img, cx, cy, size, color_bgr):
    half = size // 2
    pts = np.array(
        [[cx, cy - half], [cx - half, cy + half], [cx + half, cy + half]], dtype=np.int32
    )
    cv2.fillPoly(img, [pts], color_bgr)
    x, y, w, h = cv2.boundingRect(pts)
    return {"shape": "triangle", "center": (cx, cy), "bbox": (x, y, w, h)}


def draw_rectangle(img, cx, cy, w_half, h_half, color_bgr):
    x1, y1 = cx - w_half, cy - h_half
    x2, y2 = cx + w_half, cy + h_half
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, -1)
    return {
        "shape": "rectangle",
        "center": (cx, cy),
        "bbox": (x1, y1, x2 - x1, y2 - y1),
    }


def save_image_and_gt(img, detections, name):
    img_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    gt_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    cv2.imwrite(img_path, img)
    with open(gt_path, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"  Created {img_path} with {len(detections)} markers")


def generate_test_01_single_red_circle():
    """Single red circle on noisy background."""
    img = noisy_background(480, 640)
    info = draw_circle(img, 320, 240, 60, COLORS_BGR["red"])
    info["color"] = "red"
    save_image_and_gt(img, [info], "test_01_single_red_circle")


def generate_test_02_single_blue_rectangle():
    """Single blue rectangle on noisy background."""
    img = noisy_background(480, 640)
    info = draw_rectangle(img, 300, 250, 70, 50, COLORS_BGR["blue"])
    info["color"] = "blue"
    save_image_and_gt(img, [info], "test_02_single_blue_rectangle")


def generate_test_03_multiple_shapes():
    """Multiple non-overlapping colored shapes."""
    img = noisy_background(480, 640)
    detections = []

    info = draw_circle(img, 120, 120, 50, COLORS_BGR["red"])
    info["color"] = "red"
    detections.append(info)

    info = draw_triangle(img, 350, 130, 90, COLORS_BGR["green"])
    info["color"] = "green"
    detections.append(info)

    info = draw_rectangle(img, 530, 130, 45, 40, COLORS_BGR["blue"])
    info["color"] = "blue"
    detections.append(info)

    info = draw_circle(img, 200, 350, 55, COLORS_BGR["yellow"])
    info["color"] = "yellow"
    detections.append(info)

    info = draw_triangle(img, 450, 360, 80, COLORS_BGR["red"])
    info["color"] = "red"
    detections.append(info)

    save_image_and_gt(img, detections, "test_03_multiple_shapes")


def generate_test_04_overlapping_shapes():
    """Two overlapping shapes of the same color (tests NMS)."""
    img = noisy_background(480, 640)
    detections = []

    info = draw_circle(img, 300, 240, 70, COLORS_BGR["green"])
    info["color"] = "green"
    detections.append(info)

    info = draw_circle(img, 340, 250, 50, COLORS_BGR["green"])
    info["color"] = "green"
    detections.append(info)

    info = draw_rectangle(img, 150, 200, 40, 35, COLORS_BGR["yellow"])
    info["color"] = "yellow"
    detections.append(info)

    save_image_and_gt(img, detections, "test_04_overlapping_shapes")


def generate_test_05_noisy_scene():
    """Shapes on a more complex noisy background with varying sizes."""
    img = noisy_background(480, 640)

    # Add some random gray blobs for extra noise
    for _ in range(15):
        cx = random.randint(0, 639)
        cy = random.randint(0, 479)
        r = random.randint(3, 12)
        gray = random.randint(100, 220)
        cv2.circle(img, (cx, cy), r, (gray, gray, gray), -1)

    detections = []

    info = draw_circle(img, 500, 100, 40, COLORS_BGR["blue"])
    info["color"] = "blue"
    detections.append(info)

    info = draw_rectangle(img, 100, 380, 55, 45, COLORS_BGR["red"])
    info["color"] = "red"
    detections.append(info)

    info = draw_triangle(img, 400, 380, 100, COLORS_BGR["yellow"])
    info["color"] = "yellow"
    detections.append(info)

    save_image_and_gt(img, detections, "test_05_noisy_scene")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating test images...")
    generate_test_01_single_red_circle()
    generate_test_02_single_blue_rectangle()
    generate_test_03_multiple_shapes()
    generate_test_04_overlapping_shapes()
    generate_test_05_noisy_scene()
    print(f"\nDone! Test images saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
