import numpy as np
import cv2 as cv
import glob # dùng khi muốn tạo ra danh sách nhiều ảnh



#===================FIND POINTS====================

# thuật toán tối ưu hóa ( criteria : dừng )
# EPS dừng lại khi sự chênh lệnh lỗi giữa các lần cập nhật dưới một ngưỡng nhất định
# (dưới đây là 0.001)
# MAX_ITER dừng lại khi số lần lặp lại tối đa (dưới đây là 30)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# chuẩn bị mảng numpy, như (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# Dòng đầu kích thước (6*7,3) 6,7 là số điểm đánh dấu trên các cột và hàng của bản đồ
# 3 là chiều không gian (x,y,z) , np.float32 kiểu dữ liệu
# Dòng hai dùng lát cắt để tạo mảng cho 2 cột đầu tiên của mảng objp
# objp[:, :2] là chỉ những cột 0,1 tức là cột (x,y) mới được gán giá trị : đầu tiên là full hàng :2 t2 là từ cột 1-2
# np.mgrid[0:7,0:6].T.reshape(-1,2) là tạo ra lưới điểm trên bàn cờ
# mgrid là mảng 2 chiều (chiều x và chiều y gồm 7 hàng 6 cột), reshape là biến mgrid thành full hàng 2 cột (x,y)
objp = np.zeros((17 * 24, 3), np.float32)
objp[:, :2] = np.mgrid[0:24, 0:17].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('ImageCali_Model\*.png')

for image in images :
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('vid', gray)
    #cv.waitKey(0)

    # Find the chess board corners
    # corners là mảng numpy có kích thước (n,1,2) , n là số góc, 1 là chiều của mảng, 2 là số phần tử (x,y)
    ret, corners = cv.findChessboardCorners(gray, (24, 17), None)
    #print(corners)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)    #Thêm objp vào mảng objpoints
        # tinh chỉnh chính xác các tọa độ 2D (11,11) là kích thước cửa sổ để tính toán (-1,-1) độ phân giải của ảnh
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)    #Thêm corners2 vào mảng imgpoints
        #print(objpoints)
        #print(imgpoints)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (24, 17), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)


cv.destroyAllWindows()



#==========================CALIBRATION=========================

# Trả về : độ chính xác (từ 0-1), ma trận camera, hệ số biến dạng, list vector quay, list vector dịch chuyển
# Đầu vào : (w,h) kích thước khung hình đvi pixel, None : xác định tự động bởi hàm
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (1440,1080), None, None)



#==========================UNDISTORTTION=======================

# Trả về NewCameraMatrix để tối ưu hóa ma trận camera và trả về ROI của ảnh => tăng độ chính xác cho nhận diện bàn cờ

images1 = glob.glob('Image_Input\*.png')
num = 0
for image1 in images1 :
    img1 = cv.imread(image1)
    h, w = img1.shape[:2]       # Đọc kích thước ảnh
    # (w,h) đầu là kích thước ảnh , 1 là tỉ lệ zoom , (w,h) sau là kích thước ảnh sau zoom
    # roi là vùng quan tâm của ảnh ( thường được cắt để phân tích, xử lí riêng )
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


#--------Undistorttion-------
# Method 1 : Use undistort()
    # Giảm biến dạng hình ảnh
    dst = cv.undistort(img1, cameraMatrix, dist, None, newCameraMatrix)
# crop the image
    # biến x,y là tọa độ của roi, w, h là chiều rộng và chiều cao của roi
    x, y, w, h = roi
    # cắt ảnh dst thành ảnh có tọa độ và độ rộng, cao giống roi ( cắt từ y đến y+h-1, cắt từ x đến x+w-1 )
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('ImageCali_Output\caliResults' + str(num) + '.jpg' , dst)
    num += 1

# Method 2 : Remapping
# Chưa nghiên cứu



#===========================REPROJECTION ERROR=======================

mean_error = 0
# len(objpoints) độ dài của mảng, range là số nguyên từ 0 đến len(objpoints)-1
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )