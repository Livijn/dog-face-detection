from pathlib import Path
import argparse
import requests
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_url", type=str, default=None, help="the url to parse")
    parser.add_argument("--show", help="show the result", action="store_true")
    parser.add_argument("--save", help="save the result", action="store_true")
    args = parser.parse_args()

    # Retrieve the image
    if args.image_url is not None:
        response = requests.get(args.image_url)
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        height, width, *_ = img.shape
        print(f"width={width}, height={height}")
    else:
        raise ValueError("Must provide --image_url")

    # Load the model and predict on the image
    model = YOLO(str(Path(__file__).resolve().parent) + '/model.pt')
    results = model.predict(source=img)
    
    # Print the results
    for r in results:
        bounding_boxes = r.boxes.xyxy
        confidences = r.boxes.conf
        for box, confidence in zip(bounding_boxes, confidences):
            x_min, y_min, x_max, y_max = box.tolist()
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            x_center_percentage = (x_center / width) * 100
            y_center_percentage = (y_center / height) * 100
      
            print(f"Bounding Box: x_center={x_center_percentage:.2f}%, y_center={y_center_percentage:.2f}%, x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, conf={confidence.item()}")
        
    # Visualize the results
    for i, r in enumerate(results):
        if args.show is not None: 
            r.show()
            
        if args.save is not None: 
            r.save(filename=f"tmp/results{i}.jpg")

if __name__ == "__main__":
    main()
