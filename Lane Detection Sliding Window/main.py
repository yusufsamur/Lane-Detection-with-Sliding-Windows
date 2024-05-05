import cv2
import numpy as np

cap = cv2.VideoCapture("project_video.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = 640
height = 480
fps = int(cap.get(cv2.CAP_PROP_FPS))

def nothing(x):
    pass

# Create Trackbar for Test
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Selecting Coordinates for Perspective
perspectiveRatios = [(0.4, 0.65), (0.18, 0.9), (0.59, 0.65), (0.87, 0.9)]
perspective_points = [
    (int(ratio[0] * width), int(ratio[1] * height)) for ratio in perspectiveRatios
]
pts1 = np.float32(
    [
        perspective_points[0],
        perspective_points[1],
        perspective_points[2],
        perspective_points[3],
    ]
)
pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width, height))
    ## drawing circles for detect the perspective points
    # cv2.circle(frame, perspective_points[0], 5, (0, 0, 255), -1)
    # cv2.circle(frame, perspective_points[1], 5, (0, 0, 255), -1)
    # cv2.circle(frame, perspective_points[2], 5, (0, 0, 255), -1)
    # cv2.circle(frame, perspective_points[3], 5, (0, 0, 255), -1)

    # Geometrical Transformation, Matrix to birds eye
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))

    # Image Threshold
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)
    mask = cv2.GaussianBlur(mask,(5,5),0)

    # Histogram for Lanes
    histogram = np.sum(
        mask[mask.shape[0] // 2 :, :], axis=0
    )  # The value of all columns in the middle row are summed and assigned to the histogram. We do this to create a threshold value
    midpoint = int(histogram.shape[0] / 2)
    left_peak = np.argmax(histogram[:midpoint])  # max value of left
    right_peak = (
        np.argmax(histogram[midpoint:]) + midpoint
    )  # max value of right

    # Sliding Windows for Curves
    y = 472  # starting point of windows
    lx = []
    rx = []
    msk = mask.copy()

    while y > 0:
        ## Left Sliding Windows
        rectangleImg = mask[
            y - 40 : y,
            left_peak - 50 : left_peak + 50,  # There will be 12 window (480/40)
        ]  # 40 and 100 for rectangle height and width
        contours, _ = cv2.findContours(
            rectangleImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(
                    M["m10"] / M["m00"]
                )  # x coordinate of detected lane in sliding window that we dealing with.
                cy = int(
                    M["m01"] / M["m00"]
                )  # y coordinate of detected lane in sliding window that we dealing with.
                lx.append(left_peak - 50 + cx)
                left_peak = (
                    left_peak - 50 + cx
                )  # Changing the peak value of left lane according to previous window

        ## Right Sliding Windows
        rectangleImg = mask[
            y - 40 : y,
            right_peak - 50 : right_peak + 50,  # There will be 12 window (480/40)
        ]  # 40 and 100 for rectangle height and width
        contours, _ = cv2.findContours(
            rectangleImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(
                    M["m10"] / M["m00"]
                )  # x coordinate of detected lane in sliding window that we dealing with.
                cy = int(
                    M["m01"] / M["m00"]
                )  # y coordinate of detected lane in sliding window that we dealing with.
                rx.append(right_peak - 50 + cx)
                right_peak = (
                    right_peak - 50 + cx
                )  # Changing the peak value of left lane according to previous window
        cv2.rectangle(
            msk, (left_peak - 50, y), (left_peak + 50, y - 40), (255, 255, 255), 1
        )
        cv2.rectangle(
            msk, (right_peak - 50, y), (right_peak + 50, y - 40), (255, 255, 255), 1
        )
        y -= 40
    print(f"{rx} \n {lx}")
    cv2.imshow("video",frame)
    cv2.imshow("Windows", msk)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
