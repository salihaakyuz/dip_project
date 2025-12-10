import cv2
import numpy as np
import os


def detect_lanes(frame):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # 4. Region of Interest (ROI)
    height = frame.shape[0]
    mask = np.zeros_like(edges)

    # Define triangular region
    polygon = np.array(
        [
            [
                (0, height),
                (frame.shape[1], height),
                (frame.shape[1] // 2, int(height * 0.6)),
            ]
        ]
    )
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    # 5. Hough Transform
    lines = cv2.HoughLinesP(
        masked, rho=1, theta=np.pi / 180, threshold=50, minLineLength=40, maxLineGap=120
    )

    # 6. Draw lines
    line_image = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 6)

    # Overlay original + line image
    output = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return output


# video path : server/media/road.mp4
print("Absolute path:", os.path.abspath("server/media/road.mp4"))
path = os.path.abspath("../media/road.mp4")
cap = cv2.VideoCapture(path)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    lanes = detect_lanes(frame)
    cv2.imshow("Lane Detection", lanes)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
