# cd yolov5
# python train.py --img 640 --batch 5 --epochs 3 
# --data ../datasets/wheat.yaml --weights yolov5s.pt --workers 0


#oppure


# Load a COCO-pretrained YOLOv5n model and train it on the COCO8 example dataset for 100 epochs
# yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640





yolo train imgsz=640 batch=5 epochs=3  data=./datasets/wheat.yaml model=yolov5s.pt

#oppure

    # from ultralytics import YOLO

    # # Load a COCO-pretrained YOLOv5n model
    # model = YOLO("yolov5n.pt")

    # # Display model information (optional)
    # model.info()

    # # Train the model on the COCO8 example dataset for 100 epochs
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)