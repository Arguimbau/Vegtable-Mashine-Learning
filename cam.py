import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Define the class names and their order
class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Brocoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

vid = cv2.VideoCapture(0)

predictions_list = []

while (True):
    # Capture the video frame
    ret, frame = vid.read()

    # Preprocess the image
    img = cv2.resize(frame, (100, 100))  # Resize the image to the target size
    img = np.expand_dims(img, axis=0)  # Expand the dimensions

    # Use the model to make a prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])

    # Add the prediction to the list
    predictions_list.append(predicted_class)

    # If there are more than 20 predictions, remove the oldest one
    if len(predictions_list) > 20:
        predictions_list.pop(0)

    # Get the most common prediction in the list
    most_common_prediction = max(set(predictions_list), key=predictions_list.count)

    # Get the class name from the class_names list
    class_name = class_names[most_common_prediction]

    # Print the predicted class on the screen
    cv2.putText(frame, class_name, (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,(255,255,255), 4, 2)

    # Display the resulting frame
    cv2.imshow('Camera feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()