import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load our trained hand model
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

# Start the webcam and hand tracker
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with vision.HandLandmarker.create_from_options(options) as tracker:
    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not access camera!")
                break

            # Mirror the video like a selfie
            frame = cv2.flip(frame, 1)

            # OpenCV uses BGR; MediaPipe needs RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = time.monotonic_ns() // 1_000_000
            result = tracker.detect_for_video(image, timestamp)

            # Draw the 21 landmarks for each detected hand
            height, width, _ = frame.shape

            for hand in result.hand_landmarks:
                points = []

                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points.append((x, y))

                for start, end in [
                    (0,1), (1,2), (2,3), (3,4),
                    (0,5), (5,6), (6,7), (7,8),
                    (5,9), (9,10), (10,11), (11,12),
                    (9,13), (13,14), (14,15), (15,16),
                    (13,17), (17,18), (18,19), (19,20),
                    (0,17)
                ]:
                    cv2.line(frame, points[start],
                             points[end], (0,255,0), 2)

                for x, y in points:
                    cv2.circle(frame, (x,y), 5, (0,0,255), -1)

            cv2.namedWindow("HANDS-ON | Hand Tracking", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "HANDS-ON | Hand Tracking",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
            cv2.imshow("HANDS-ON | Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()