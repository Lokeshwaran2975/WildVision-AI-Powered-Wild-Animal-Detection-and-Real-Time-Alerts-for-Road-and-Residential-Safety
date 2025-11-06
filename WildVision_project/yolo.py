from ultralytics import YOLO

def main():
    # Load pretrained YOLOv11 small model
    model = YOLO("yolo11s.pt")

    # Train the model on GPU
    model.train(
        data=r"D:\ANIMAL-DETECTION-main\ANIMAL-DETECTION-main\Wildvision Dataset\data.yaml",
        epochs=150,
        imgsz=416,
        batch=4,
        device=0,        # Use GPU
        workers=0        # Set to 0 to avoid multiprocessing issues (or leave default)
    )

    # Validate the model
    model.val(device=0)

    # Run prediction on test images
    results = model.predict(
        source=r"D:\ANIMAL-DETECTION-main\ANIMAL-DETECTION-main\Wildvision Dataset\images\test",
        save=True,
        device=0
    )

if _name_ == "_main_":
    main()