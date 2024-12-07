import numpy as np 
import cv2 

# Capturing video through webcam 
webcam = cv2.VideoCapture(0) 

# Start a while loop 
while(1): 
    # Reading the video from the webcam in image frames 
    _, imageFrame = webcam.read() 

    # Convert the imageFrame in BGR (RGB color space) to HSV (hue-saturation-value) color space 
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

    # Define color ranges and create masks
    # Red color
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Green color
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Blue color
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Yellow color
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Black color
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    # White color
    white_lower = np.array([0, 0, 200], np.uint8)
    white_upper = np.array([180, 20, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    # Gray color
    gray_lower = np.array([0, 0, 40], np.uint8)
    gray_upper = np.array([180, 20, 200], np.uint8)
    gray_mask = cv2.inRange(hsvFrame, gray_lower, gray_upper)

    # Beige color
    beige_lower = np.array([20, 40, 200], np.uint8)
    beige_upper = np.array([35, 150, 255], np.uint8)
    beige_mask = cv2.inRange(hsvFrame, beige_lower, beige_upper)

    # Brown color
    brown_lower = np.array([10, 100, 20], np.uint8)
    brown_upper = np.array([20, 255, 200], np.uint8)
    brown_mask = cv2.inRange(hsvFrame, brown_lower, brown_upper)

    # Purple color
    purple_lower = np.array([130, 50, 50], np.uint8)
    purple_upper = np.array([160, 255, 255], np.uint8)
    purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)

    # Orange color
    orange_lower = np.array([5, 150, 150], np.uint8)
    orange_upper = np.array([15, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Dilate masks to fill gaps
    kernel = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    black_mask = cv2.dilate(black_mask, kernel)
    white_mask = cv2.dilate(white_mask, kernel)
    gray_mask = cv2.dilate(gray_mask, kernel)
    beige_mask = cv2.dilate(beige_mask, kernel)
    brown_mask = cv2.dilate(brown_mask, kernel)
    purple_mask = cv2.dilate(purple_mask, kernel)
    orange_mask = cv2.dilate(orange_mask, kernel)

    # Detect and label each color
    colors = {
        "Red": (red_mask, (0, 0, 255)),
        "Green": (green_mask, (0, 255, 0)),
        "Blue": (blue_mask, (255, 0, 0)),
        "Yellow": (yellow_mask, (0, 255, 255)),
        "Black": (black_mask, (0, 0, 0)),
        "White": (white_mask, (255, 255, 255)),
        "Gray": (gray_mask, (128, 128, 128)),
        "Beige": (beige_mask, (245, 245, 220)),
        "Brown": (brown_mask, (42, 42, 165)),
        "Purple": (purple_mask, (128, 0, 128)),
        "Orange": (orange_mask, (0, 165, 255))
    }

    for color_name, (mask, box_color) in colors.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(imageFrame, f"{color_name} Color", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Display the resulting frame
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
