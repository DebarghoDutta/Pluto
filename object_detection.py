import argparse
import importlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "object_dataset"
IMAGE_ROOT = DATASET_ROOT / "images"
COCO_META_ROOT = PROJECT_ROOT / "coco"
LABEL_ROOT = COCO_META_ROOT / "labels"
DATASET_YAML = DATASET_ROOT / "coco_detection.yaml"
LABELS_ZIP = PROJECT_ROOT / "coco2017labels-segments.zip"
TRAIN_ZIP = IMAGE_ROOT / "train2017.zip"
VAL_ZIP = IMAGE_ROOT / "val2017.zip"

LABELS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "coco2017labels-segments.zip"
)
TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def download_with_retry(url: str, destination: Path, max_retries: int = 3, delay: int = 5):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"Skipping download, file already exists: {destination}")
        return

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {destination.name} ({attempt}/{max_retries})")
            urllib.request.urlretrieve(url, destination)
            print(f"Saved to {destination}")
            return
        except Exception as error:
            print(f"Download failed: {error}")
            if attempt == max_retries:
                raise
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)


def extract_zip_if_needed(zip_path: Path, output_dir: Path):
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Skipping extraction, folder already populated: {output_dir}")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    print(f"Extracting {zip_path.name} to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir.parent)


def write_dataset_yaml():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    yaml_lines = [
        f"path: {DATASET_ROOT.as_posix()}",
        "train: images/train2017",
        "val: images/val2017",
        "",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(COCO_CLASSES))
    DATASET_YAML.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"Dataset config created at {DATASET_YAML}")


def prepare_dataset(download_missing: bool = False, extract_archives: bool = True):
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    if download_missing:
        if not LABELS_ZIP.exists() and not LABEL_ROOT.exists():
            download_with_retry(LABELS_URL, LABELS_ZIP)
        if not VAL_ZIP.exists() and not (IMAGE_ROOT / "val2017").exists():
            download_with_retry(VAL_URL, VAL_ZIP)
        if not TRAIN_ZIP.exists() and not (IMAGE_ROOT / "train2017").exists():
            download_with_retry(TRAIN_URL, TRAIN_ZIP)

    if extract_archives:
        if LABELS_ZIP.exists() and not LABEL_ROOT.exists():
            extract_zip_if_needed(LABELS_ZIP, COCO_META_ROOT)
        if VAL_ZIP.exists():
            extract_zip_if_needed(VAL_ZIP, IMAGE_ROOT / "val2017")
        if TRAIN_ZIP.exists():
            extract_zip_if_needed(TRAIN_ZIP, IMAGE_ROOT / "train2017")

    write_dataset_yaml()
    validate_dataset_layout()


def validate_dataset_layout():
    required_paths = {
        "validation images": IMAGE_ROOT / "val2017",
        "validation labels": LABEL_ROOT / "val2017",
    }

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset is incomplete. Missing: " + ", ".join(missing)
        )

    if not (IMAGE_ROOT / "train2017").exists():
        print(
            "Training images are not extracted yet. You can still run prediction or "
            "extract train2017.zip later for training."
        )


def ensure_ultralytics():
    try:
        ultralytics_module = importlib.import_module("ultralytics")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The 'ultralytics' package is required. Install it with:\n"
            "pip install ultralytics"
        ) from error
    return ultralytics_module


def train_model(model_name: str, epochs: int, imgsz: int, batch: int, device: str):
    prepare_dataset(download_missing=False, extract_archives=False)

    train_dir = IMAGE_ROOT / "train2017"
    train_labels = LABEL_ROOT / "train2017"
    if not train_dir.exists() or not train_labels.exists():
        raise FileNotFoundError(
            "Training data not ready. Make sure train2017 images are extracted and "
            "labels exist before starting training."
        )

    ultralytics_module = ensure_ultralytics()
    model = ultralytics_module.YOLO(model_name)
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(PROJECT_ROOT / "runs"),
        name="object_detection_train",
    )


def validate_model(weights: str):
    prepare_dataset(download_missing=False, extract_archives=False)
    ultralytics_module = ensure_ultralytics()
    model = ultralytics_module.YOLO(weights)
    metrics = model.val(data=str(DATASET_YAML))
    print(metrics)


def predict_objects(weights: str, source: str, conf: float):
    prepare_dataset(download_missing=False, extract_archives=False)
    ultralytics_module = ensure_ultralytics()
    model = ultralytics_module.YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        show=True,
        save=True,
        project=str(PROJECT_ROOT / "runs"),
        name="object_detection_predict",
    )
    print(f"Prediction complete. Result batches: {len(results)}")


def webcam_detection(weights: str, camera_index: int, conf: float):
    predict_objects(weights=weights, source=str(camera_index), conf=conf)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare a COCO-style dataset and run object detection with YOLO."
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset files.")
    prepare_parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing dataset zip files before preparing.",
    )
    prepare_parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Do not extract zip files.",
    )

    train_parser = subparsers.add_parser("train", help="Train an object detector.")
    train_parser.add_argument("--model", default="yolov8n.pt")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--device", default="cpu")

    val_parser = subparsers.add_parser("val", help="Validate a trained model.")
    val_parser.add_argument("--weights", default="yolov8n.pt")

    predict_parser = subparsers.add_parser("predict", help="Run prediction on a source.")
    predict_parser.add_argument("--weights", default="yolov8n.pt")
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--conf", type=float, default=0.25)

    webcam_parser = subparsers.add_parser("webcam", help="Run live webcam detection.")
    webcam_parser.add_argument("--weights", default="yolov8n.pt")
    webcam_parser.add_argument("--camera", type=int, default=0)
    webcam_parser.add_argument("--conf", type=float, default=0.25)

    return parser


def run_menu():
    while True:
        print("\nChoose an option:")
        print("1. Prepare dataset")
        print("2. Train object detector")
        print("3. Validate model")
        print("4. Detect objects in an image/video")
        print("5. Detect objects from webcam")
        print("6. Exit")

        choice = input("Enter your choice: ").strip()

        try:
            match choice:
                case "1":
                    download_choice = input(
                        "Download missing files too? (y/n): "
                    ).strip().lower()
                    prepare_dataset(download_missing=download_choice == "y")
                case "2":
                    train_model(
                        model_name=input("Model [yolov8n.pt]: ").strip() or "yolov8n.pt",
                        epochs=int(input("Epochs [10]: ").strip() or "10"),
                        imgsz=int(input("Image size [640]: ").strip() or "640"),
                        batch=int(input("Batch size [8]: ").strip() or "8"),
                        device=input("Device [cpu]: ").strip() or "cpu",
                    )
                case "3":
                    validate_model(input("Weights [yolov8n.pt]: ").strip() or "yolov8n.pt")
                case "4":
                    predict_objects(
                        weights=input("Weights [yolov8n.pt]: ").strip() or "yolov8n.pt",
                        source=input("Image/video path: ").strip(),
                        conf=float(input("Confidence [0.25]: ").strip() or "0.25"),
                    )
                case "5":
                    webcam_detection(
                        weights=input("Weights [yolov8n.pt]: ").strip() or "yolov8n.pt",
                        camera_index=int(input("Camera index [0]: ").strip() or "0"),
                        conf=float(input("Confidence [0.25]: ").strip() or "0.25"),
                    )
                case "6":
                    print("Exiting program.")
                    break
                case _:
                    print("Invalid choice. Please select 1 to 6.")
        except Exception as error:
            print(f"Error: {error}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        run_menu()
        return

    if args.command == "prepare":
        prepare_dataset(
            download_missing=args.download,
            extract_archives=not args.skip_extract,
        )
    elif args.command == "train":
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    elif args.command == "val":
        validate_model(weights=args.weights)
    elif args.command == "predict":
        predict_objects(weights=args.weights, source=args.source, conf=args.conf)
    elif args.command == "webcam":
        webcam_detection(
            weights=args.weights,
            camera_index=args.camera,
            conf=args.conf,
        )


if __name__ == "__main__":
    main()
