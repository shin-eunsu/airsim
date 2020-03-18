import cv2
import numpy as np
#import matplotlib.pyplot as plt

#경계선 자르기
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    #print(image.shape) # (높이,폭, 채널) (704, 1279, 3)
    y1 = image.shape[0] #높이
    y2 = int(y1*(3/5)) #높이의 5분의3길이
    x1 = int((y1 - intercept) / slope) # 전체높이의 기울기  , int -> float으로 바꿔서 dtype (float64)로바꾸기
    x2 = int((y2 - intercept) / slope) # 5분의 3지점 기울기
    return np.array([x1, y1, x2, y2])

#평균기울기얻기
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        # print(line)#[[x1,y1,x2,y2]]2차원 배열 값이 나옴
        x1, y1, x2, y2 = line.reshape(4) #4개의 행인 2차원배열을 1차원으로 바꿔줌 unpack 오류뜨면 이거수정해본다
        parameters = np.polyfit((x1, x2), (y1, y2), 1) #노이즈에서 직선을 찾는것 np.polyfit(다항식,다항식, 찾고자 하는 함수 차수
        #print(parameters)# 기울기와 절편나옴 [ 1.03448276 -302.27586207]
        slope = parameters[0] #기울기
        intercept = parameters[1] #y절편
        if slope < 0:
            left_fit.append((slope, intercept)) #append 추가하는거
        else:
            right_fit.append((slope, intercept))


    left_fit_average = np.average(left_fit, axis = 0) #axis=0는 x축을 기준으로 합을 구하는 방식입니다
    right_fit_average = np.average(right_fit, axis =0)
    #print(left_fit_average)
    #print(right_fit_average)
    #left_line = make_coordinates(image, left_fit_average)
    #right_line = make_coordinates(image, right_fit_average)
    left_line = None if not left_fit else make_coordinates(image, left_fit_average)
    right_line = None if not right_fit else make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line] )

    #x =np.array([left_line,right_line])
    #print(x.dtype)


# 윤곽선 만들기
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 그레이톤으로 바꾸기
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 선에 노이즈를 줄이기위해서 GaussianBlur를준다 커널크기는 5*5 0=편차
    canny = cv2.Canny(blur, 50, 150)  # 캐니 - (최소값,최대값) 값으로선 선만 남기기
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines[0] is not None and lines[1] is not None: #lines가 비어있지않으면
        for x1, y1, x2, y2 in lines:
            #print(line)#[[x1,y1,x2,y2]]2차원 배열 값이 나옴
            #x1, y1, x2, y2 = line.reshape(4) # 4개의 행인 2차원배열을 1차원으로 바꿔줌 unpack 오류뜨면 이거수정해본다
            cv2.line(line_image, (x1, y1),(x2, y2),(255, 0, 0), 10) #선잇기,선 색깔 정해주기,선두깨
    return line_image


# 도로라인 다각형으로 추출하기 &  다각형으로 도로만 선그리기
def region_of_interest(image):
    height = image.shape[0]  # [0]세로 [1] 가로 이미지창의 길이
    #다각형 좌표 알기  주의 *[[ ]]*
    polygons = np.array([[(200, height),(1100, height),(550, 250)]])  #np.array([(),()=밑변 ()수직길이 = x축좌표점을 알기위함
    # 0과1 배열로 이미지를 바꿔준다.
    mask = np.zeros_like(image) # 이미 있는 array와 동일한 모양과 데이터형태를 유지한 상태에서 '0', '1', '빈 배열'을반환함
    cv2.fillPoly(mask, polygons, 255) #마스크를 흰색삼각형모양으로 채워라
    masked_image = cv2.bitwise_and(image , mask) # bit연산 이미지 지울때씀  (이미지파일, 마스크파일)
    return masked_image

image= cv2.imread('./test_image.jpg') #사진 불러오기
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
# HoughLines = 누락되거나 꺠진영역 복원기능, 선라인 이으는데 씀(선검출)
#(image ,r(0~1실수)값의 범위,theta(0~180정수),threshold ,선의 최소길이, 선과 선사이 간격 이 값보다 작으면 값을 안받음
#threshold – 만나는 점의 기준, 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap= 5)
averaged_lines = average_slope_intercept(lane_image, lines) # lines의 선을 한줄로 만들어주는것
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1 ,1) #addWeighted = 이미지 섞기 (img,비율,img,비율,밝기)
cv2.imshow('result', combo_image) #이미지보기
cv2.waitKey(0)  #이미지 키는시간  0=무한

# cap = cv2.VideoCapture("./test2.mp4")
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = canny(frame)
#     cropped_image = region_of_interest(canny_image)
#     # HoughLines = 누락되거나 꺠진영역 복원기능, 선라인 이으는데 씀(선검출)
#     # (image ,r(0~1실수)값의 범위,theta(0~180정수),threshold ,선의 최소길이, 선과 선사이 간격 이 값보다 작으면 값을 안받음
#     # threshold – 만나는 점의 기준, 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
#     lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = average_slope_intercept(frame, lines)  # lines의 선을 한줄로 만들어주는것
#     line_image = display_lines(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  # addWeighted = 이미지 섞기 (img,비율,img,비율,밝기)
#
#     cv2.imshow('result', combo_image)  # 이미지보기
#     if cv2.waitKey(1) & 0xFF == ord('q'): # 이미지 키는시간  0=무한
#         break

cap.release()
cv2.destroyAllWindows()
#그래프보기
#plt.imshow(canny)
#plt.show()