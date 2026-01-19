from ultralytics import YOLO

if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Load a model
    model = YOLO("yolov8n-pose.pt")

    # Train the model
    model.train(
        data="coco-pose.yaml",
        epochs=3,
        imgsz=640,
        batch=16,
        # workers=4,
        name="yolov8n-pose-swinTransformer",
        exist_ok=True,
    )
