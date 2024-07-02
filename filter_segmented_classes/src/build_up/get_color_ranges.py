import cv2
import numpy as np

# Function to capture RGB values of selected points
def get_color_ranges(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = image[y, x].tolist()
            points.append(point)
            print(f"Point selected at ({x}, {y}) with RGB value: {point}")

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    print("Click on the image to select points. Press 'q' to quit.")
    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if not points:
        print("No points selected.")
        return

    points = np.array(points)
    lower_bounds = np.max([points - 55, np.zeros(points.shape)], axis=0)  # Avoid negative values
    upper_bounds = np.min([points + 55, np.full(points.shape, 255)], axis=0)  # Avoid values above 255

    print("Suggested color ranges:")
    for lower, upper in zip(lower_bounds, upper_bounds):
        print(f"Lower: {lower}, Upper: {upper}")

if __name__ == "__main__":
    # Path to your segmented image
    image_path = '/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/example.png'
    get_color_ranges(image_path)
