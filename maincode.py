import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# 1. YOLO 모델 불러오기
model = YOLO('yolov8n.pt')

# 2. 색상 카테고리 분류 함수
def get_color_category(h, s, v):
    if s < 35: 
        if v > 180: return "White"
        elif v < 85: return "Black"
        else: return "Grey"

    if 0 <= h < 15 or 165 <= h <= 179: return "Red"
    elif 15 <= h < 45: return "Yellow"
    elif 45 <= h < 75: return "Green"
    elif 75 <= h < 105: return "Cyan"
    elif 105 <= h < 135: return "Blue"
    elif 135 <= h < 165: return "Purple"
    
    return "Other"

# 3. 5x5 그리드 투표 방식 
def detect_color_by_grid(image_crop, label=""):
    if image_crop is None or image_crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape

    grid_rows = 5
    grid_cols = 5
    step_h = h // grid_rows
    step_w = w // grid_cols

    votes = [] 

    for i in range(grid_rows):
        for j in range(grid_cols):
            y1 = i * step_h
            y2 = (i + 1) * step_h
            x1 = j * step_w
            x2 = (j + 1) * step_w
            
            cell = hsv[y1:y2, x1:x2]
            
            if cell.size > 0:
                mean_color = np.mean(cell, axis=(0, 1))
                category = get_color_category(mean_color[0], mean_color[1], mean_color[2])
                votes.append(category)

    if not votes: return "Unknown"
    
    vote_result = Counter(votes)
    most_common_color = vote_result.most_common(1)[0][0]
    
    return most_common_color

# [추가됨] 화면에 5x5 격자를 그려주는 도우미 함수
def draw_grid(image, x1, y1, x2, y2, rows=5, cols=5, color=(200, 200, 200)):
    step_h = (y2 - y1) // rows
    step_w = (x2 - x1) // cols
    
    # 가로선 그리기
    for i in range(1, rows):
        cv2.line(image, (x1, y1 + i * step_h), (x2, y1 + i * step_h), color, 1)
    # 세로선 그리기
    for j in range(1, cols):
        cv2.line(image, (x1 + j * step_w, y1), (x1 + j * step_w, y2), color, 1)

# 4. 웹캠 실행
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, classes=[0], verbose=False)
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 사람 박스
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            height = y2 - y1
            top_y1 = y1 + int(height * 0.15)
            top_y2 = y1 + int(height * 0.45)
            bot_y1 = y1 + int(height * 0.50)
            bot_y2 = y1 + int(height * 0.80)

            top_crop = frame[top_y1:top_y2, x1:x2]
            bot_crop = frame[bot_y1:bot_y2, x1:x2]

            # 함수 호출 시 라벨(Top/Bot)을 같이 넘겨서 로그 식별
            top_color = detect_color_by_grid(top_crop, "Top")
            bot_color = detect_color_by_grid(bot_crop, "Bot")

            # [확인용] 화면에 실제로 5x5 격자를 그려서 보여줌
            draw_grid(annotated_frame, x1, top_y1, x2, top_y2, color=(255, 255, 0)) # 상의: 하늘색 격자
            draw_grid(annotated_frame, x1, bot_y1, x2, bot_y2, color=(255, 0, 255)) # 하의: 분홍색 격자

            # 정보 표시
            info_text = f"Top: {top_color} | Bot: {bot_color}"
            (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            text_y = y1 + 25
            cv2.rectangle(annotated_frame, (x1, text_y - text_h - 5), (x1 + text_w + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(annotated_frame, info_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 영역 박스 표시
            cv2.rectangle(annotated_frame, (x1, top_y1), (x2, top_y2), (255, 255, 0), 2)
            cv2.rectangle(annotated_frame, (x1, bot_y1), (x2, bot_y2), (255, 0, 255), 2)

    cv2.imshow('Fashion AI - Debug Mode', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()