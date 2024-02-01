import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def capture_and_analyze():
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture a single frame
        ret, frame = cap.read()

        # Display the frame in a window
        cv2.imshow("Camera Preview", frame)

        # Check for key press 'q' to exit, 'c' to capture an image
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save the captured frame as an image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"captured_image_{timestamp}.jpg"
            cv2.imwrite(image_filename, frame)
            print(f"Image captured successfully: {image_filename}")

            # Perform image recognition
            predict_image(image_filename)

            # Detect and point out any metal substance in the image
            detect_metal_substance(image_filename)

            # Prompt for the person's birthdate
            birthdate = input("Enter the person's birthdate (YYYY-MM-DD): ")

            # Calculate age in days
            age_days = calculate_age(birthdate)
            print(f"The person has been alive on Earth for approximately {age_days} days.")

            # Display the captured image
            cv2.imshow("Captured Image", cv2.imread(image_filename))
            cv2.waitKey(0)
            cv2.destroyWindow("Captured Image")

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

def predict_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Expand dimensions to match the model's expected input shape
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)

    # Decode and print the top-3 predicted classes
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

def detect_metal_substance(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and check for metal-like features (you may need to fine-tune this based on your requirements)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust the area threshold as needed
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

    # Save the result
    cv2.imwrite("metal_substance_detection_result.jpg", img)
    print("Metal substance detection result saved as metal_substance_detection_result.jpg")

def calculate_age(birthdate):
    # Convert birthdate string to datetime object
    birth_date = datetime.strptime(birthdate, "%Y-%m-%d")

    # Calculate age in days
    age_days = (datetime.now() - birth_date).days

    return age_days

if __name__ == "__main__":
    # Capture image, perform analysis, and detect metal substance
    capture_and_analyze()
