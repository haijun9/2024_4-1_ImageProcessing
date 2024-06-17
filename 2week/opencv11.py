import cv2

video_file = './jin.avi' # 동영상 경로
cap = cv2.VideoCapture(video_file) # 동영상 캡처 객체 생성

if cap.isOpened(): # 객체 초기화 정상
    fps = 30.0 # FPS, 초당 프레임 수
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 인코딩 포맷 문자
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 영상 저장을 위한 객체 생성
    out = cv2.VideoWriter('./record_color.avi', fourcc, fps, (w, h), isColor=True)
    while True:
        ret, frame = cap.read() # 다음 프레임 읽기
        if ret: # 프레임 읽기 정상
            cv2.namedWindow(video_file, flags=cv2.WINDOW_NORMAL)
            cv2.imshow(video_file, frame) # win_name, img
            out.write(frame) # frame 저장
            if cv2.waitKey(int(1000/fps)) != -1: # 지연시간 = 1000/fps
                break
        else: # 프레임 읽기 비정상
            break
    out.release() # 저장 객체 소멸
else: # 객체 초기화 비정상
    print("can't open video")

cap.release() # 캡처 객체 소멸
cv2.destroyAllWindows()