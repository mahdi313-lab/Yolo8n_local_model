import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from train import train_model


if __name__ == "__main__":

    # Training Stage
    train_model(
        train_images_dir='tiled_train_and_val/train_images_1024',
        train_labels_dir='tiled_train_and_val/train_labels_1024',
        val_images_dir='tiled_train_and_val/val_images_1024',
        val_labels_dir='tiled_train_and_val/val_labels_1024',
        num_epochs=100,
        batch_size=2,
        lr=0.001,
    )
