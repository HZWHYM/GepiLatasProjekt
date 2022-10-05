import cv2
import numpy as np

PATH_TO_IMAGE = 'test.png'
loaded_image = cv2.imread(PATH_TO_IMAGE)
cv2.imshow("Loaded source image", loaded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

resized_image = cv2.resize(loaded_image, (600, 600), interpolation=cv2.INTER_LINEAR)
loaded_image = cv2.resize(loaded_image, (600, 600), interpolation=cv2.INTER_LINEAR)
# cv2.imshow("Resized image", resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

bfiltered_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("Blurred image", bfiltered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

edged_image = cv2.Canny(bfiltered_image, 30, 200)
cv2.imshow("Edge detection image", edged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
tmp_image = loaded_image.copy()
cv2.drawContours(tmp_image, contours, -1, (0, 255, 0), 1)
cv2.imshow("Contours on original image", tmp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
print(location)

if (location is None):
    output = "No license plate found."
else:
    mask = np.zeros(gray_image.shape, np.uint8)
    tmp_image = cv2.drawContours(mask, [location], 0, 255, -1)
    cv2.imshow("tmp image", tmp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    tmp_image2 = cv2.drawContours(loaded_image.copy(), [location], -1, (0, 255, 0), 2)
    cv2.imshow("Detected area on original resized picture.", tmp_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    tmp_image = cv2.bitwise_and(loaded_image, loaded_image, mask=mask)
    cv2.imshow("tmp image bitwise", tmp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
