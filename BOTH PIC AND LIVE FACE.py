import cv2
from fer import FER # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os

def process_frame(frame):
    # Detect emotions in the frame
    emotions = emotion_detector.detect_emotions(frame)

    # Draw a rectangle around the face and put the emotion label
    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        emotion_label = max(emotion["emotions"], key=emotion["emotions"].get)

        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Put emotion label above the rectangle
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (36,255,12), 2)

    return frame

def live_video_emotion_recognition():
    cap = cv2.VideoCapture(0)  # Use '0' for the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        
        cv2.imshow('Emotion Recognition - Live Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def image_emotion_recognition(image_path):
    frame = cv2.imread(image_path)
    frame = process_frame(frame)
    
    # Display the resulting frame
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

# Initialize the FER detector
emotion_detector = FER(mtcnn=True)

def main():
    print("Choose an option:")
    print("1. Live Video Emotion Recognition")
    print("2. Emotion Recognition from Image")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        live_video_emotion_recognition()
    elif choice == '2':
        image_path = input("Enter the path to the image file: ")
        if os.path.exists(image_path):
            image_emotion_recognition(image_path)
        else:
            print("Invalid file path. Please try again.")
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()
