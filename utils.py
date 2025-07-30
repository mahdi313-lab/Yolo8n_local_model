import os
import re
import random
# --------
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def get_latest_checkpoint(weights_dir):
    pattern = re.compile(r"yolo8_aircraft_model_(\d+)\.pth")
    latest_epoch = -1
    latest_file = None
    for filename in os.listdir(weights_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = os.path.join(weights_dir, filename)
    if latest_file is None:
        return None, None
    return latest_file, latest_epoch

def plot_yolo_image_with_boxes(image_path, label_path):
    img = Image.open(image_path)
    width, height = img.size

    boxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            # class_id is ignored
            _, x_center, y_center, w, h = parts
            x_center, y_center, w, h = map(float, [x_center, y_center, w, h])

            x = (x_center - w / 2) * width
            y = (y_center - h / 2) * height
            w *= width
            h *= height
            boxes.append((x, y, w, h))

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for (x, y, w, h) in boxes:
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # === Parameters ===
    out_img_dir = r'f:\rahimzamani_objectdetection_models\HRPlanes_v2\tiled_data\train_1024_tiles'
    out_label_dir = r'f:\rahimzamani_objectdetection_models\HRPlanes_v2\tiled_data\train_1024_labels'
    num_samples = 20  # Number of random images  to visualize

    # List all tiled images
    image_files = [f for f in os.listdir(out_img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    if not image_files:
        print("No images found in output image directory.")
    else:
        sample_images = random.sample(image_files, min(num_samples, len(image_files)))

        for img_file in sample_images:
            img_path = os.path.join(out_img_dir, img_file)
            label_path = os.path.join(out_label_dir, os.path.splitext(img_file)[0] + '.txt')

            if os.path.exists(label_path):
                plot_yolo_image_with_boxes(img_path, label_path)
            else:
                print(f"Label not found for {img_file}")
