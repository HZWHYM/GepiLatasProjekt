import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['en'])


def get_warp_perspective(img, coords):

    license_plate = np.zeros((4, 2), dtype="float32")
    summary = coords.sum(axis=1)
    license_plate[0] = coords[np.argmin(summary)]
    license_plate[2] = coords[np.argmax(summary)]
    difference = np.diff(coords, axis=1)
    license_plate[1] = coords[np.argmin(difference)]
    license_plate[3] = coords[np.argmax(difference)]

    (pt_A, pt_B, pt_C, pt_D) = license_plate


    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    output_pts = np.array([[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]], dtype="float32")

    M = cv2.getPerspectiveTransform(license_plate, output_pts)

    return cv2.flip(cv2.rotate(cv2.warpPerspective(img, M, (maxWidth, maxHeight)), cv2.ROTATE_90_COUNTERCLOCKWISE), 0)


def detect_license_plate(filename):
    # sourcery skip: inline-immediately-returned-variable
    loaded_image = cv2.imread(filename)
    resized_image = cv2.resize(loaded_image, (600, 600), interpolation=cv2.INTER_LINEAR)

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray image", gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    bfiltered_image = cv2.bilateralFilter(gray_image, 11, 17, 17)  # 11, 17, 17
    # cv2.imshow("Blurred image", bfiltered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edged_image = cv2.Canny(bfiltered_image, 100, 300)
    # cv2.imshow("Edge detection image", edged_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp_image = resized_image.copy()
    cv2.drawContours(tmp_image, contours, -1, (0, 255, 0), 1)
    # cv2.imshow("Contours on original image", tmp_image)
    # cv2.waitKey(0)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if (location is None):
        return "No plate detected"

    mask = np.zeros(gray_image.shape, np.uint8)
    tmp_image = cv2.drawContours(mask, [location], 0, 255, -1)
    # cv2.imshow("tmp image", tmp_image)
    # cv2.waitKey(0)

    tmp_image2 = cv2.drawContours(resized_image.copy(), [location], -1, (0, 255, 0), 2)
    # cv2.imshow("Detected area on original resized picture.", tmp_image2)
    # cv2.waitKey(0)

    tmp_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)
    # cv2.imshow("tmp image bitwise", tmp_image)
    # cv2.waitKey(0)

    (x, y) = np.where(mask == 255)
    (x_min_value, y_min_value) = (np.min(x), np.min(y))
    (x_max_value, y_max_value) = (np.max(x), np.max(y))
    cropped = gray_image[x_min_value: x_max_value + 1, y_min_value: y_max_value + 1]
    # cv2.imshow("cropped plate", cropped)
    # cv2.waitKey(0)
    cropped = get_warp_perspective(gray_image, location.reshape(4, 2))
    # cv2.imshow("warped plate", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    try:
        cropped = cv2.bilateralFilter(cropped, 11, 90, 90)
        result = reader.readtext(cropped)[0][1]
        result_formatted = ''.join(ch.upper() for ch in result if ch.isalnum())
        return result_formatted
    except Exception:
        return None
