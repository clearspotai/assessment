import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path="ssd_mobilenet_v1_1_default_1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]


# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()

    # Resize and normalize the frame for MobileNetV1
    # Assumes the input size for the model is 320X320
    frame_resized = cv2.resize(frame, (300, 300))
    # frame_normalized = (np.uint8(frame_resized) - 127.5) / 127.5
    frame_input = np.expand_dims(frame_resized, axis=0)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], frame_input)
    
    # Run inference
    interpreter.invoke()

    # Extract the detection boxes and labels
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw the bounding boxes and labels onto the frame
    for i in range(len(boxes)):
        if np.max(scores[i]) > 0.6:  # Only show predictions with a confidence score above 0.5
            # print(boxes[i])
            box = boxes[i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            class_id = int(classes[i])
            label_name = labels[class_id]
            label = f"{label_name}: {scores[i]:.2f}"

            # Draw the bounding box
            cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)

            # Draw the class label
            cv2.putText(frame, label, (int(box[1]), int(box[0]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
