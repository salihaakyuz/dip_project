import cv2
import numpy as np
import os


def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def average_slope_intercept(lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            continue  # avoid infinite slope

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_lane = np.mean(left_fit, axis=0) if len(left_fit) else None
    right_lane = np.mean(right_fit, axis=0) if len(right_fit) else None

    return left_lane, right_lane


def make_line_points(y1, y2, line):
    if line is None:
        return None

    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, y1, x2, y2)


def detect_lanes(frame):
    height, width = frame.shape[:2]

    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny
    edges = cv2.Canny(blur, 50, 150)

    # 4. Region of interest
    roi_vertices = np.array(
        [
            [
                (0, height),
                (width, height),
                (int(width * 0.55), int(height * 0.6)),
                (int(width * 0.45), int(height * 0.6)),
            ]
        ]
    )

    cropped_edges = region_of_interest(edges, roi_vertices)

    # 5. Hough Transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=150,
    )

    # 6. Average left and right lines
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = height
    y2 = int(height * 0.6)

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    # 7. Draw lines
    line_image = np.zeros_like(frame)

    if left_line is not None:
        cv2.line(
            line_image,
            (left_line[0], left_line[1]),
            (left_line[2], left_line[3]),
            (0, 255, 0),
            8,
        )
    if right_line is not None:
        cv2.line(
            line_image,
            (right_line[0], right_line[1]),
            (right_line[2], right_line[3]),
            (0, 255, 0),
            8,
        )

    # Merge with original image
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


# ---------------- RUN WITH IMAGE ----------------
img_path = "road3.jpeg"  # change this path
path = os.path.abspath("../media/" + img_path)
if not os.path.exists(path):
    print("❌ Image not found:", path)
    exit()

image = cv2.imread(path)
lanes = detect_lanes(image)

cv2.imshow("Lane Detection (Image)", lanes)
cv2.waitKey(0)
cv2.destroyAllWindows()
