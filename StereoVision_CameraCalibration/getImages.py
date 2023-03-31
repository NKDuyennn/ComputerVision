import cv2

imgleft = cv2.VideoCapture(0)
imgright = cv2.VideoCapture(1)

num = 0

while imgleft.isOpened()  :

    left, imgl = imgleft.read()
    right, imgr = imgright.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('Images_Model/Stereo_ImagesLeft/imageL' + str(num) + '.jpg', imgl)
        cv2.imwrite('Images_Model/Stereo_ImagesRight/imageR' + str(num) + '.jpg', imgr)
        print("images saved!")
        num += 1

    cv2.imshow('Img Left',imgl)
    cv2.imshow('Img Right',imgr)

# Release and destroy all windows before termination
imgleft.release()
imgright.release()

cv2.destroyAllWindows()