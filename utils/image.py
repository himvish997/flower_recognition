import cv2

def display(image,  time=1000, window_name="Image"):
    image = cv2.resize(image, (720, 720))
    cv2.imshow(winname=window_name, mat=image)
    key = cv2.waitKey(time)  # pauses for 3 seconds before fetching next image
    if key == 27:  # if ESC is pressed, exit loop
        cv2.destroyAllWindows()