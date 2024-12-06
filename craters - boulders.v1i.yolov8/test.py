from ultralytics import YOLO
import cv2
import os

def detect_objects(model_path, image_path, conf_threshold=0.25):
    """
    Detect objects in images using a trained YOLO model.
    
    Args:
        model_path (str): Path to the trained .pt model file
        image_path (str): Path to image or directory of images
        conf_threshold (float): Confidence threshold for detections
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        save_txt=True  # Save detection results in YOLO format
    )
    
    # Process and display each image
    for result in results:
        image = cv2.imread(str(result.path))
        
        # Draw bounding boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name} {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Display image
        cv2.imshow('Detection', image)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    model_path = r"D:\Projects\Boulders_Craters\craters - boulders.v1i.yolov8\best.pt"  # Replace with your model path
    image_path = r"D:\Projects\Boulders_Craters\craters - boulders.v1i.yolov8\test\images"  # Replace with your image path
    detect_objects(model_path, image_path, conf_threshold=0.25)