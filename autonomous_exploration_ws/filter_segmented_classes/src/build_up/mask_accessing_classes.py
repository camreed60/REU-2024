from ultralytics import YOLO
import cv2
import numpy as np

def get_segmentation_masks_and_classes(image_path, model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Load image
    img = cv2.imread(image_path)

    # Perform inference
    results = model.predict(img)

    # Extract class names from the model
    class_names = model.names

    # Extract segmentation masks and class labels
    binary_masks = []
    class_labels = []
    if results[0].masks is not None:
        masks = results[0].masks.data
        for mask, cls in zip(masks, results[0].boxes.cls):
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8).reshape(-1)
            binary_masks.append(mask_np)
            class_labels.append(int(cls.item()))

    return binary_masks, class_labels, class_names

# Example usage
image_path = '/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/color_image.jpg'
model_path = '/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/walkway_v1.pt'
masks, classes, class_names = get_segmentation_masks_and_classes(image_path, model_path)

# Print the shape of the masks and the classes to verify
for i, (mask, cls) in enumerate(zip(masks, classes)):
    print(f'Mask {i+1} shape: {mask.shape}')
    print(f'Mask {i+1} class: {cls} ({class_names[cls]})')
    print(mask)
    print(class_names)
