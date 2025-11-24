import cv2
import numpy as np
from ultralytics import YOLO

# ===  색상 분석 함수 (HSV 30단위 간격) ===
def get_simple_color(image_crop):
    # 이미지가 너무 작거나 없으면 예외 처리
    if image_crop is None or image_crop.size == 0:
        return "Unknown"

    # 1. HSV 변환
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    
    # 2. 채널 분리
    h, s, v = cv2.split(hsv)

    # 3. 영역의 '평균값' 구하기
    mean_h = h.mean()
    mean_s = s.mean()
    mean_v = v.mean()

    # 4. 무채색(흰/검/회) 먼저 걸러내기
    # 채도가 낮거나(40 미만), 명도가 너무 낮으면(50 미만) 무채색으로 판단
    if mean_s < 40: 
        return "White" if mean_v > 140 else "Grey" # 명도가 높으면 흰색, 아니면 회색
    if mean_v < 50: 
        return "Black"

    # 5. 색상 구분 (OpenCV Hue: 0~179, 30단위 간격)
    if mean_h < 15 or mean_h > 165: return "Red"
    elif mean_h < 45: return "Yellow"
    elif mean_h < 75: return "Green"
    elif mean_h < 105: return "Cyan"
    elif mean_h < 135: return "Blue"
    elif mean_h < 165: return "Purple"
    
    return "Other"

# 1. YOLO 모델 불러오기
model = YOLO('yolov8n.pt')

# 2. 웹캠 켜기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("실행 중... 종료하려면 'q' 키를 누르세요.")

while True:
    # 3. 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 4. YOLO로 물체 탐지 (사람만 탐지하도록 classes=[0] 설정)
    results = model(frame, classes=[0], verbose=False)

    # 5. 결과 시각화 (기본 박스 그리기)
    annotated_frame = results[0].plot()

    # === 상/하의 구분 및 색상 표시 로직 ===
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 좌표 추출 (정수형 변환)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 사람 키와 너비 계산
            height = y2 - y1
            width = x2 - x1

            # 기하학적 분할: 상의(Top)와 하의(Bottom) 영역 정의
            # 상의: 머리 아래부터 허리까지 (대략 상단 15% ~ 45%)
            top_y1 = y1 + int(height * 0.15)
            top_y2 = y1 + int(height * 0.45)
            
            # 하의: 허리부터 무릎까지 (대략 상단 50% ~ 80%)
            bot_y1 = y1 + int(height * 0.50)
            bot_y2 = y1 + int(height * 0.80)

            # 이미지 잘라내기 (Crop)
            top_crop = frame[top_y1:top_y2, x1:x2]
            bot_crop = frame[bot_y1:bot_y2, x1:x2]

            # 색상 분석 실행
            top_color = get_simple_color(top_crop)
            bot_color = get_simple_color(bot_crop)

            # 화면에 정보 표시
            # 1) 분석 영역 박스 표시 (선택 사항)
            cv2.rectangle(annotated_frame, (x1, top_y1), (x2, top_y2), (255, 255, 0), 2) # 상의 박스 (하늘색)
            cv2.rectangle(annotated_frame, (x1, bot_y1), (x2, bot_y2), (255, 0, 255), 2) # 하의 박스 (분홍색)

            # 2) 텍스트 띄우기
            info_text = f"Top: {top_color} | Bot: {bot_color}"
            cv2.putText(annotated_frame, info_text, (x1, y1 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 6. 화면에 출력
    cv2.imshow('YOLOv8 Fashion Detector', annotated_frame)

    # 7. 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. 자원 해제
cap.release()
cv2.destroyAllWindows()