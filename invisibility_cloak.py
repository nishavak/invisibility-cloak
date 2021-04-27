import numpy as np
import cv2


def main():
    stream = cv2.VideoCapture(0)

    frame_grabbed, background = stream.read()
    cv2.imshow("Background", background)

    while True:
        frame_grabbed, frame = stream.read()
        if frame_grabbed:

            cv2.imshow("Invisible Cloak Video Input", frame)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            low_r = np.array([100, 40, 40])
            up_r = np.array([100, 255, 255])
            mask_1 = cv2.inRange(hsv, low_r, up_r)

            low_r = np.array([155, 40, 40])
            up_r = np.array([180, 255, 255])
            mask_2 = cv2.inRange(hsv, low_r, up_r)

            mask_1 = mask_1 + mask_2

            mask_1 = cv2.morphologyEx(
                mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2
            )
            mask_1 = cv2.dilate(mask_1, np.ones((3, 3), np.uint8), iterations=1)
            mask_2 = cv2.bitwise_not(mask_1)

            result_1 = cv2.bitwise_and(background, background, mask=mask_1)
            result_2 = cv2.bitwise_and(frame, frame, mask=mask_2)
            final_output = cv2.addWeighted(result_1, 1, result_2, 1, 0)

            cv2.imshow("Invisible Cloak", final_output)

            if cv2.waitKey(1) == ord("q"):
                break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

