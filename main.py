import cv2
import imutils
import numpy as np
from imutils import paths


def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


# initialize the known distance from the camera to the object, which
# in this case is 24 inches
# KNOWN_DISTANCE = 30.0
KNOWN_DISTANCE = int(input("Distance of intial object from camera(inches): "))


# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
# KNOWN_WIDTH = 13.0
KNOWN_WIDTH = int(input("Width of object(inches): "))

# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
cap = cv2.VideoCapture(0)
start_image = None
while True:
    ret, image = cap.read()
    cv2.imshow("start!", image)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        start_image = image
        break
marker = find_marker(start_image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the imagess
# for imagePath in sorted(paths.list_images("images")):
#     # load the image, find the marker in the image, then compute the
#     # distance to the marker from the camera
#     image = cv2.imread(imagePath)
#     marker = find_marker(image)
#     inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
#
#     # draw a bounding box around the image and display it
#     box = cv2.cv.BoxPoints(
#         marker) if imutils.is_cv2() else cv2.boxPoints(marker)
#     box = np.int0(box)
#     cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
#     cv2.putText(image, "%.2fft" % (inches / 12),
#                 (image.shape[1] - 200, image.shape[0]
#                  - 20), cv2.FONT_HERSHEY_SIMPLEX,
#                 2.0, (0, 255, 0), 3)
#     cv2.imshow("image", image)
#     cv2.waitKey(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    marker = find_marker(frame)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

    # draw a bounding box around the image and display it
    box = cv2.cv.BoxPoints(
        marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    cv2.putText(frame, "%.2fft" % (inches / 12),
                (frame.shape[1] - 200, frame.shape[0]
                 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 0), 3)
    cv2.imshow("image", frame)
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
