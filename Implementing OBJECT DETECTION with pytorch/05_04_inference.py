# cd yolov5
# python detect.py --weights runs/train/exp/weights/best.pt 
# --source ../datasets/wheat/images/val --conf 0.25



#oppure


# Load a COCO-pretrained YOLOv5n model and run inference on the 'bus.jpg' image
yolo predict model=runs/detect/train4/weights/best.pt source=./datasets/wheat/images/val conf=0.25


#oppure

# Run inference with the YOLOv5n model on the 'bus.jpg' image
#results = model("path/to/bus.jpg")