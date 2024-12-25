import argparse
import requests
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_url", type=str, default=None)
    args = parser.parse_args()

    model = YOLO("model.pt")

    # Retrieve the image
    if args.image_url is not None:
        response = requests.get(args.image_url)
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Must provide --image_url")

    results = model.predict(source=img)
    
    for r in results:
        bounding_boxes = r.boxes.xyxy
        confidences = r.boxes.conf
        class_labels = r.boxes.cls
        for box, confidence, cls in zip(bounding_boxes, confidences, class_labels):
            x_min, y_min, x_max, y_max = box.tolist()
            print(f"Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            print(f"Confidence: {confidence.item()}, Class: {cls.item()}")
        
    # Visualize the results
#     for i, r in enumerate(results):
#         r.show()
#         r.save(filename=f"results{i}.jpg")

if __name__ == "__main__":
    main()
