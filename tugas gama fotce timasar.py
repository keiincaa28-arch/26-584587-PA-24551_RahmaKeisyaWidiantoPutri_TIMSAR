import cv2
import numpy as np
import math

# Load the input image
image = cv2.imread("input.jpg")

if image is None:
    raise FileNotFoundError(
        "Image not found. Place input.jpg in the project folder."
    )

original = image.copy()
result = image.copy()

# Convert image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Color masks based on the example image.
# These thresholds may need adjustment for other images.
color_ranges = {
    "Helipad": [
        (np.array([0, 100, 80]), np.array([6, 255, 255])),
        (np.array([170, 100, 80]), np.array([180, 255, 255]))
    ],
    "Korban": [
        (np.array([6, 100, 100]), np.array([20, 255, 255]))
    ],
    "Sumber Bahaya": [
        (np.array([21, 100, 100]), np.array([40, 255, 255]))
    ]
}

# Minimum contour area to reject small noise
MIN_AREA = 20

# Safe-distance threshold in pixels (example value)
SAFE_DISTANCE = 150


def detect_object(hsv_image, ranges):
    """Detect the largest suitable contour in the given color range."""
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        part = cv2.inRange(hsv_image, lower, upper)
        mask = cv2.bitwise_or(mask, part)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

    if not valid:
        return None

    contour = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return {
        "box": (x, y, w, h),
        "center": (cx, cy),
        "contour": contour
    }


# Detect all three objects
detected = {}

colors = {
    "Helipad": (0, 0, 255),       # Red
    "Korban": (0, 165, 255),      # Orange
    "Sumber Bahaya": (0, 255, 255) # Yellow
}

for name, ranges in color_ranges.items():
    obj = detect_object(hsv, ranges)

    if obj is None:
        print(f"WARNING: {name} was not detected.")
        continue

    detected[name] = obj

    x, y, w, h = obj["box"]
    cx, cy = obj["center"]
    color = colors[name]

    # Draw bounding box
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

    # Draw center marker (+)
    cv2.drawMarker(
        result, (cx, cy), color,
        markerType=cv2.MARKER_CROSS,
        markerSize=14, thickness=2
    )

    # Draw label
    cv2.putText(
        result, name, (x, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
    )

    print(f"{name}: center=({cx}, {cy}), box={(x, y, w, h)}")


# Calculate pixel distance between Korban and Sumber Bahaya
if "Korban" in detected and "Sumber Bahaya" in detected:
    p1 = detected["Korban"]["center"]
    p2 = detected["Sumber Bahaya"]["center"]

    distance = math.dist(p1, p2)

    # Draw connecting line
    cv2.line(result, p1, p2, (255, 255, 255), 2)

    # Draw distance label at the midpoint
    midpoint = (
        (p1[0] + p2[0]) // 2,
        (p1[1] + p2[1]) // 2
    )

    cv2.putText(
        result, f"{distance:.1f} px", midpoint,
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2
    )

    # Determine status
    if distance < SAFE_DISTANCE:
        status = "CRITICAL"
        status_color = (0, 0, 255)
    else:
        status = "SAFE"
        status_color = (0, 255, 0)

    cv2.putText(
        result, f"Status Korban: {status}",
        (15, result.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        status_color, 2
    )

    print(f"Distance: {distance:.2f} pixels")
    print(f"Status: {status}")
else:
    cv2.putText(
        result, "Distance unavailable: object missing",
        (15, result.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )


# Display Original Image and Result side by side
original_display = original.copy()
result_display = result.copy()

cv2.putText(
    original_display, "Original Image", (15, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
)

cv2.putText(
    result_display, "Result", (15, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
)

combined = np.hstack((original_display, result_display))

# Save output for the assignment
cv2.imwrite("result.jpg", result)
cv2.imwrite("comparison.jpg", combined)

cv2.imshow("Original Image | Result", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()