import cv2

cap = cv2.VideoCapture(0)
num = 0

while cap.isOpened():    # Test xem camera có được mở hay không
    ret, img = cap.read()
    k = cv2.waitKey(5) # Chờ 1 phím được nhấn trong 5ms , không nhấn phím nào trả về -1
    if k == 27:        # 27 là mã của nút "ESC"
        break
    elif k == ord('s'): # Nếu nhấn s thì sẽ lưu ảnh vào file Image_Input
        cv2.imwrite('ImageCali_Model\img' + str(num) + '.jpg', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img', img)
# Giải phóng tài nguyên đầu vào cho những lần sử dụng sau
cap.release()
# Đóng tất cả cửa số hiển thị
cv2.destroyAllWindows()