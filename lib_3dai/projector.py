import cv2

# ------------------- Projector -------------------
def project_image(img_np_uint8,second_screen_x):
    win = "PROJECTOR"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(win, second_screen_x, 0)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, img_np_uint8)
    cv2.waitKey(100)
