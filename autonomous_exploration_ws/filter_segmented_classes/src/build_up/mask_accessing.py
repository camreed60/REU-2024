from ultralytics import YOLO
import cv2
import numpy as np

def get_segmentation_masks(image_path, model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Load image
    img = cv2.imread(image_path)

    # Perform inference
    results = model.predict(img)

    # Extract segmentation masks
    if results[0].masks is not None:
        masks = results[0].masks.data
        binary_masks = [(mask.cpu().numpy() * 255).astype(np.uint8).reshape(-1) for mask in masks]
    else:
        binary_masks = []

    return binary_masks

# Example usage
image_path = '/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/color_image.jpg'
model_path = '/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/walkway_v1.pt'
masks = get_segmentation_masks(image_path, model_path)

# Print the shape of the masks to verify
for i, mask in enumerate(masks):
    print(f'Mask {i+1} shape: {mask.shape}')
    print(mask)
