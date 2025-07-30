import os
from PIL import Image, ImageOps
import math
import random

# -------------------------------
# preprocessing Functions
# -------------------------------
def split_image_and_labels(image_path, label_path, out_img_dir, out_label_dir, tile_size=1024, empty_keep_prob=1/3):
    img = Image.open(image_path)
    width, height = img.size

    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = parts
                class_id = int(class_id)
                x_center, y_center, w, h = map(float, [x_center, y_center, w, h])
                x_center_abs = x_center * width
                y_center_abs = y_center * height
                w_abs = w * width
                h_abs = h * height
                x_min = x_center_abs - w_abs / 2
                y_min = y_center_abs - h_abs / 2
                x_max = x_center_abs + w_abs / 2
                y_max = y_center_abs + h_abs / 2
                labels.append([class_id, x_min, y_min, x_max, y_max])

    x_tiles = math.ceil(width / tile_size)
    y_tiles = math.ceil(height / tile_size)

    basename = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(x_tiles):
        for j in range(y_tiles):
            left = i * tile_size
            upper = j * tile_size
            right = min(left + tile_size, width)
            lower = min(upper + tile_size, height)

            tile = img.crop((left, upper, right, lower))

            tile_w, tile_h = tile.size
            if tile_w < tile_size or tile_h < tile_size:
                pad_right = tile_size - tile_w
                pad_bottom = tile_size - tile_h
                padding = (0, 0, pad_right, pad_bottom)
                tile = ImageOps.expand(tile, border=padding, fill=0)

            tile_name = f"{basename}_{i}_{j}.jpg"
            tile_path = os.path.join(out_img_dir, tile_name)

            tile_labels = []
            for class_id, x_min, y_min, x_max, y_max in labels:
                original_area = (x_max - x_min) * (y_max - y_min)

                # Compute intersection with tile
                inter_x_min = max(x_min, left)
                inter_y_min = max(y_min, upper)
                inter_x_max = min(x_max, right)
                inter_y_max = min(y_max, lower)

                if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                    clipped_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

                    if clipped_area >= 0.5 * original_area:
                        box_x_min = inter_x_min - left
                        box_y_min = inter_y_min - upper
                        box_x_max = inter_x_max - left
                        box_y_max = inter_y_max - upper

                        box_w = box_x_max - box_x_min
                        box_h = box_y_max - box_y_min
                        box_x_center = box_x_min + box_w / 2
                        box_y_center = box_y_min + box_h / 2

                        norm_x_center = box_x_center / tile_size
                        norm_y_center = box_y_center / tile_size
                        norm_w = box_w / tile_size
                        norm_h = box_h / tile_size

                        if norm_w > 0 and norm_h > 0:
                            tile_labels.append(f"{class_id} {norm_x_center:.6f} {norm_y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            # === Keep all tiles with objects, keep ~1/3 of empty tiles ===
            save_tile = True
            if not tile_labels:
                save_tile = random.random() < empty_keep_prob

            if save_tile:
                tile.save(tile_path)
                label_name = f"{basename}_{i}_{j}.txt"
                label_path_out = os.path.join(out_label_dir, label_name)
                # Always write label file, even if empty
                with open(label_path_out, 'w') as f:
                    if tile_labels:
                        f.write('\n'.join(tile_labels))
                    # else write nothing (empty file)

def preprocess_dataset(data_dir, out_img_dir, out_label_dir, tile_size=1024):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        label_path = os.path.join(data_dir, os.path.splitext(img_file)[0] + '.txt')
        print(f"Processing {img_file} ...")
        split_image_and_labels(img_path, label_path, out_img_dir, out_label_dir, tile_size)

    print("All images processed!")

if __name__ == "__main__":
    # === Parameters ===
    data_dir = r''  # Folder with images and labels
    out_img_dir = r''
    out_label_dir = r''

    preprocess_dataset(data_dir, out_img_dir, out_label_dir, tile_size=1024)
