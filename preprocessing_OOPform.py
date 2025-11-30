"""
Class:
    BDDPreprocessor - provides methods for:
        - filtering annotations JSON by classes
        - removing images without valid labels
        - converting annotations to YOLO format
        - copying valid images and labels to organized dirs
        - splitting a validation directory into val/test subsets

"""

from pathlib import Path
import json
import shutil
import random
from typing import List, Tuple, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class BDDPreprocessor:
    """
    Unified preprocessor class for converting dataset annotations and
    organizing images/labels for training with YOLO-style format.
    """

    def __init__(self, input_json: str, output_dir: str):
        """
        Parameters
        ----------
        input_json : str
            Path to the source annotations JSON (e.g., bdd100k labels).
        output_dir : str
            Base output directory where processed files are written.
        """
        self.input_json = Path(input_json)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Initialized BDDPreprocessor with JSON: {self.input_json} and output: {self.output_dir}")

    # ---------------------------
    # 1) Filter annotations JSON by selected classes
    # ---------------------------
    def filter_annotations(self, keep_classes: List[str], out_json: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Keep only annotations whose category is in keep_classes.

        Parameters
        ----------
        keep_classes : List[str]
            Categories to keep (e.g., ["car", "person", "traffic light"]).
        out_json : Optional[str]
            Optional path to save the filtered JSON. If None, writes to
            output_dir / 'annotations_filtered.json'.

        Returns
        -------
        filtered_list : List[dict]
            Filtered annotation list (same high-level structure as input JSON).
        """
        out_json = Path(out_json) if out_json else self.output_dir / "annotations_filtered.json"

        with self.input_json.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Expecting data to be a list of image annotation dicts.
        filtered = []
        removed_images = 0

        for item in data:
            # each item should contain an 'labels' list (bdd100k style)
            labels = item.get("labels", [])
            keep_labels = [lab for lab in labels if lab.get("category") in keep_classes]

            if keep_labels:
                new_item = dict(item)  # shallow copy
                new_item["labels"] = keep_labels
                filtered.append(new_item)
            else:
                removed_images += 1

        with out_json.open("w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

        logging.info(f"Filtered annotations saved to: {out_json}")
        logging.info(f"Images removed (no matching classes): {removed_images}")
        return filtered

    # ---------------------------
    # 2) Remove images without valid labels (on disk)
    # ---------------------------
    def remove_images_without_labels(self, images_dir: str, labels_json: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Remove image files from images_dir that are not present in the labels_json list.

        Parameters
        ----------
        images_dir : str
            Directory containing image files to check.
        labels_json : Optional[List[dict]]
            If provided, a list of annotation dicts (as returned by filter_annotations).
            If not provided, function will try to read self.output_dir / 'annotations_filtered.json'.

        Returns
        -------
        removed_count : int
            Number of image files removed.
        """
        images_dir = Path(images_dir)
        if labels_json is None:
            possible = self.output_dir / "annotations_filtered.json"
            if not possible.exists():
                raise FileNotFoundError("Filtered JSON not found. Call filter_annotations or supply labels_json.")
            with possible.open("r", encoding="utf-8") as f:
                labels_json = json.load(f)

        # Build a set of expected filenames
        expected_files = set()
        for item in labels_json:
            image_name = item.get("name") or item.get("image") or item.get("image_path")
            if image_name:
                # keep only filename portion
                expected_files.add(Path(image_name).name)

        removed_count = 0
        for img_path in images_dir.glob("*"):
            if img_path.is_file():
                if img_path.name not in expected_files:
                    logging.debug(f"Removing image with no valid labels: {img_path}")
                    img_path.unlink()
                    removed_count += 1
        logging.info(f"Images removed (no valid labels): {removed_count}")
        return removed_count

    # ---------------------------
    # 3) Convert JSON -> YOLO format
    # ---------------------------
    def convert_to_yolo(self, labels_json: Optional[List[Dict[str, Any]]] = None, images_dir: Optional[str] = None, labels_out_dir: Optional[str] = None) -> int:
        """
        Convert annotations (json elements) into YOLO-format .txt label files.

        Each YOLO line: <class_id> <x_center_rel> <y_center_rel> <width_rel> <height_rel>

        Notes:
        - Supports typical BDD-style 'box2d' dictionaries with x1, y1, x2, y2.
        - Skips unlabeled or unsupported shapes (logs warnings).

        Parameters
        ----------
        labels_json : Optional[List[dict]]
            Preloaded annotations list. If None, function will read from output_dir / 'annotations_filtered.json'.
        images_dir : Optional[str]
            Path to images folder (used to read image sizes if available). If not given, uses bbox absolute coords if image size exists in JSON.
        labels_out_dir : Optional[str]
            Directory to write .txt files. Defaults to output_dir/labels.

        Returns
        -------
        count_files : int
            Number of label files written.
        """
        if labels_json is None:
            possible = self.output_dir / "annotations_filtered.json"
            if not possible.exists():
                raise FileNotFoundError("Filtered JSON not found. Call filter_annotations or supply labels_json.")
            with possible.open("r", encoding="utf-8") as f:
                labels_json = json.load(f)

        labels_out_dir = Path(labels_out_dir) if labels_out_dir else self.output_dir / "labels"
        labels_out_dir.mkdir(parents=True, exist_ok=True)

        images_dir = Path(images_dir) if images_dir else None

        # Create a mapping from category to integer ID
        # Collect categories in the file (ordered appearance)
        categories = {}
        next_cat_id = 0

        written_files = 0

        for item in labels_json:
            # Determine image filename and (optionally) width/height
            image_name = item.get("name") or item.get("image") or item.get("image_path")
            if not image_name:
                logging.warning("Skipping item with no image name field.")
                continue
            image_filename = Path(image_name).name
            # If JSON contains width/height, use them; else, attempt to read image file.
            img_width = item.get("width")
            img_height = item.get("height")
            if (not img_width or not img_height) and images_dir:
                # try to read image size using Pillow to get width/height
                try:
                    from PIL import Image
                    with Image.open(images_dir / image_filename) as im:
                        img_width, img_height = im.size
                except Exception:
                    logging.debug(f"Could not read image size for {image_filename}; some bboxes may be skipped.")
            if not img_width or not img_height:
                # Some JSONs store image width/height as strings or nested - keep as None if absent
                logging.debug(f"No image size for {image_filename}; conversion will attempt to use absolute coords if present.")

            labels_txt_lines = []

            for lab in item.get("labels", []):
                category = lab.get("category")
                if category is None:
                    continue

                if category not in categories:
                    categories[category] = next_cat_id
                    next_cat_id += 1

                class_id = categories[category]

                # Handle 'box2d' format (common in bdd100k)
                box = lab.get("box2d")
                if box and all(k in box for k in ("x1", "y1", "x2", "y2")):
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                else:
                    # fallback: look for 'x1','x2' top-level, or seat poly2d (skip)
                    poly = lab.get("poly2d")
                    if poly:
                        logging.warning(f"poly2d found for {image_filename}, category {category}. poly2d -> bounding box extraction not implemented; skipping.")
                        continue
                    # try direct fields if present
                    x1 = lab.get("x1")
                    y1 = lab.get("y1")
                    x2 = lab.get("x2")
                    y2 = lab.get("y2")
                    if None in (x1, y1, x2, y2):
                        logging.debug(f"No usable bbox for {image_filename}, category {category}. Skipping this label.")
                        continue

                # compute center, width, height
                try:
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    bbox_w = abs(x2 - x1)
                    bbox_h = abs(y2 - y1)
                except Exception as e:
                    logging.debug(f"Invalid bbox coordinates in {image_filename}: {e}")
                    continue

                # If we have image dimensions, normalize
                if img_width and img_height:
                    x_center_rel = x_center / float(img_width)
                    y_center_rel = y_center / float(img_height)
                    w_rel = bbox_w / float(img_width)
                    h_rel = bbox_h / float(img_height)
                else:
                    # If image size missing, try to interpret coords as already normalized (0..1).
                    # Otherwise skip to avoid wrong scaling.
                    if 0 < x_center <= 1 and 0 < y_center <= 1 and 0 < bbox_w <= 1 and 0 < bbox_h <= 1:
                        x_center_rel = x_center
                        y_center_rel = y_center
                        w_rel = bbox_w
                        h_rel = bbox_h
                    else:
                        logging.warning(f"No image size and bbox not normalized for {image_filename}. Skipping bbox.")
                        continue

                # Ensure values are inside (0,1)
                vals = [x_center_rel, y_center_rel, w_rel, h_rel]
                if any(v <= 0 or v > 1 for v in vals):
                    logging.debug(f"Normalized bbox out-of-range for {image_filename}, skipping: {vals}")
                    continue

                labels_txt_lines.append(f"{class_id} {x_center_rel:.6f} {y_center_rel:.6f} {w_rel:.6f} {h_rel:.6f}")

            if labels_txt_lines:
                label_file_path = labels_out_dir / (Path(image_filename).with_suffix(".txt").name)
                with label_file_path.open("w", encoding="utf-8") as lf:
                    lf.write("\n".join(labels_txt_lines))
                written_files += 1

        logging.info(f"Wrote {written_files} YOLO label files to: {labels_out_dir}")
        logging.info(f"Category to ID mapping: {categories}")
        return written_files

    # ---------------------------
    # 4) Copy valid images & labels into clean dataset structure
    # ---------------------------
    def copy_valid_files(self, labels_json: Optional[List[Dict[str, Any]]] = None, src_images_dir: Optional[str] = None, dst_images_dir: Optional[str] = None, labels_txt_dir: Optional[str] = None) -> Tuple[int, int]:
        """
        Copy images referenced in labels_json from src_images_dir to dst_images_dir,
        and copy label txt files from labels_txt_dir (if provided) to a labels subfolder.

        Parameters
        ----------
        labels_json : Optional[List[dict]]
            Preloaded annotations list. If None, reads output_dir / 'annotations_filtered.json'.
        src_images_dir : Optional[str]
            Directory with the original images. If None, function will raise if image not found.
        dst_images_dir : Optional[str]
            Destination for copied images. Defaults to output_dir / 'images'.
        labels_txt_dir : Optional[str]
            Directory where .txt label files are stored (from convert_to_yolo).
            Defaults to output_dir / 'labels'.

        Returns
        -------
        copied_images : int
            Number of images copied.
        copied_labels : int
            Number of label files copied.
        """
        if labels_json is None:
            possible = self.output_dir / "annotations_filtered.json"
            if not possible.exists():
                raise FileNotFoundError("Filtered JSON not found. Call filter_annotations or supply labels_json.")
            with possible.open("r", encoding="utf-8") as f:
                labels_json = json.load(f)

        src_images_dir = Path(src_images_dir) if src_images_dir else None
        dst_images_dir = Path(dst_images_dir) if dst_images_dir else self.output_dir / "images"
        dst_labels_dir = Path(labels_txt_dir) if labels_txt_dir else self.output_dir / "labels"

        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        copied_labels = 0

        for item in labels_json:
            image_name = item.get("name") or item.get("image") or item.get("image_path")
            if not image_name:
                continue
            image_filename = Path(image_name).name

            if src_images_dir:
                src_img_path = src_images_dir / image_filename
                if src_img_path.exists():
                    shutil.copy2(src_img_path, dst_images_dir / image_filename)
                    copied_images += 1
                else:
                    logging.debug(f"Source image not found, skipping copy: {src_img_path}")
            else:
                logging.debug("No source images dir provided to copy from; skipping image copy step.")
                break

            # Copy corresponding label txt file if exists
            src_label_txt = dst_labels_dir / Path(image_filename).with_suffix(".txt").name
            # Note: we assume convert_to_yolo wrote labels into dst_labels_dir already; if not, user should pass labels_txt_dir param
            if src_label_txt.exists():
                # If labels were already in the target labels dir, nothing to do. Otherwise copy from specified labels_txt_dir.
                copied_labels += 1
            else:
                # try to copy label from the convert_to_yolo location (output_dir/labels) to dst_labels_dir
                default_label_src = self.output_dir / "labels" / Path(image_filename).with_suffix(".txt").name
                if default_label_src.exists():
                    shutil.copy2(default_label_src, dst_labels_dir / default_label_src.name)
                    copied_labels += 1
                else:
                    logging.debug(f"No label file found for {image_filename} to copy.")

        logging.info(f"Copied {copied_images} images to {dst_images_dir}")
        logging.info(f"Copied {copied_labels} label txt files to {dst_labels_dir}")
        return copied_images, copied_labels

    # ---------------------------
    # 5) Split validation directory into new val & test
    # ---------------------------
    def split_val_test(self, val_images_dir: str, val_labels_dir: str, new_val_dir: str, test_dir: str, test_split_ratio: float = 0.5, seed: Optional[int] = 42) -> Tuple[int, int]:
        """
        Split the existing validation folder into a new validation folder and a test folder.

        The function copies both images and their corresponding label txt files.

        Parameters
        ----------
        val_images_dir : str
            Directory containing current validation images.
        val_labels_dir : str
            Directory containing corresponding label txt files.
        new_val_dir : str
            Destination directory for the reduced validation set.
        test_dir : str
            Destination directory for the new test set.
        test_split_ratio : float
            Fraction of files from the original val to send to the test set (0..1).
        seed : Optional[int]
            Random seed for reproducible splits.

        Returns
        -------
        (len_new_val, len_new_test) : Tuple[int, int]
            Number of images in new validation and test sets.
        """
        val_images_dir = Path(val_images_dir)
        val_labels_dir = Path(val_labels_dir)
        new_val_images = Path(new_val_dir) / "images"
        new_val_labels = Path(new_val_dir) / "labels"
        test_images = Path(test_dir) / "images"
        test_labels = Path(test_dir) / "labels"

        for p in [new_val_images, new_val_labels, test_images, test_labels]:
            p.mkdir(parents=True, exist_ok=True)

        all_images = sorted([p for p in val_images_dir.glob("*") if p.is_file()])
        random.seed(seed)
        random.shuffle(all_images)

        split_at = int(len(all_images) * (1 - test_split_ratio))
        new_val_split = all_images[:split_at]
        test_split = all_images[split_at:]

        copied_val = 0
        copied_test = 0

        for src_img in new_val_split:
            shutil.copy2(src_img, new_val_images / src_img.name)
            label_src = Path(val_labels_dir) / src_img.with_suffix(".txt").name
            if label_src.exists():
                shutil.copy2(label_src, new_val_labels / label_src.name)
            copied_val += 1

        for src_img in test_split:
            shutil.copy2(src_img, test_images / src_img.name)
            label_src = Path(val_labels_dir) / src_img.with_suffix(".txt").name
            if label_src.exists():
                shutil.copy2(label_src, test_labels / label_src.name)
            copied_test += 1

        logging.info("Done splitting validation set!")
        logging.info(f"Total images: {len(all_images)}")
        logging.info(f"New validation: {copied_val} images")
        logging.info(f"New test set: {copied_test} images")
        logging.info(f"New validation path: {new_val_images.parent}")
        logging.info(f"New test path: {test_images.parent}")

        return copied_val, copied_test


# ---------------------------
# Example usage (if run as script)
# ---------------------------
if __name__ == "__main__":
    # Example - change these paths to match your filesystem
    INPUT_JSON = r"C:\Users\Raneem\Desktop\project car\train\annotations\bdd100k_labels_images_train.json"
    OUTPUT_DIR = r"C:\Users\Raneem\Desktop\project car\train\annotations\processed"
    SRC_IMAGES_DIR = r"C:\Users\Raneem\Desktop\project car\train\images"  # where original images are stored

    # Initialize
    pre = BDDPreprocessor(INPUT_JSON, OUTPUT_DIR)

    # 1) Filter annotations by classes
    KEEP = ["car", "traffic light", "person", "traffic sign"]
    filtered = pre.filter_annotations(KEEP)

    # 2) Remove images on disk that don't have labels in filtered JSON
    # pre.remove_images_without_labels(SRC_IMAGES_DIR, labels_json=filtered)

    # 3) Convert filtered JSON to YOLO .txt labels
    pre.convert_to_yolo(labels_json=filtered, images_dir=SRC_IMAGES_DIR)

    # 4) Copy valid image files and their labels to the output folder
    pre.copy_valid_files(labels_json=filtered, src_images_dir=SRC_IMAGES_DIR)

    # 5) Optionally split a validation folder into val/test
    # pre.split_val_test(val_images_dir="path/to/val/images", val_labels_dir="path/to/val/labels",
    #                    new_val_dir="path/to/new_val", test_dir="path/to/new_test", test_split_ratio=0.5)
