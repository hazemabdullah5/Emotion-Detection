import os
import random
import shutil


def move_subset(source_dir, target_dir, split_ratio=0.2):
    """
    Moves a fraction (split_ratio) of images from source_dir to target_dir.
    Preserves subfolders (class labels).
    """
    os.makedirs(target_dir, exist_ok=True)

    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        # Determine how many to move
        to_move = int(len(images) * split_ratio)
        subset = images[:to_move]

        # Create the corresponding class folder in the target directory
        target_class_path = os.path.join(target_dir, class_folder)
        os.makedirs(target_class_path, exist_ok=True)

        # Move the selected files
        for img_name in subset:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(target_class_path, img_name)
            shutil.move(src, dst)

        print(f"Moved {to_move} images from '{class_folder}' to '{target_dir}'")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Path to your 'emotion_detection' folder
    data_dir = os.path.join(base_dir, "data")
    train_dir = os.path.join(data_dir, "train")
    old_val_dir = os.path.join(data_dir, "validation")

    # 1) Rename 'validation' to 'test'
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(old_val_dir):
        raise FileNotFoundError(f"Could not find the folder: {old_val_dir}")

    if os.path.exists(test_dir):
        raise FileExistsError(f"'test' folder already exists. Remove or rename it before running this script.")

    print(f"Renaming '{old_val_dir}' to '{test_dir}'...")
    os.rename(old_val_dir, test_dir)

    # 2) Create a brand-new validation set from the train set by moving ~20%
    val_dir = os.path.join(data_dir, "validation")
    print(f"Splitting 20% of images from '{train_dir}' into new '{val_dir}'...")
    move_subset(train_dir, val_dir, split_ratio=0.2)

    print("\nDone! Final folder structure should now be:")
    print(f" - {train_dir}")
    print(f" - {val_dir}")
    print(f" - {test_dir}")


if __name__ == "__main__":
    main()
