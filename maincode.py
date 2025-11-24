import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# 1. YOLO 모델 불러오기
model = YOLO('yolov8n.pt')

# === [핵심 1] 23가지 색상 분류 로직 ===
def get_color_category(h, s, v):
    # 1. 무채색 (Achromatic)
    if s < 20:
        if v > 180: return "White"
        elif v < 90: return "Black"
        else: return "Grey"

    # 2. 유채색 (Chromatic)
    mode = "Bright" if v > 130 else "Dark"

    # Hue 범위 (0~179)를 10등분하여 매핑
    if 0 <= h < 10 or 175 <= h <= 179: return "Vivid Red" if mode == "Bright" else "Dark Red"
    elif 10 <= h < 25: return "Orange" if mode == "Bright" else "Brown"
    elif 25 <= h < 40: return "Yellow" if mode == "Bright" else "Beige"
    elif 40 <= h < 65: return "Lime" if mode == "Bright" else "Olive"
    elif 65 <= h < 85: return "Green" if mode == "Bright" else "Forest"
    elif 85 <= h < 105: return "Cyan" if mode == "Bright" else "Teal"
    elif 105 <= h < 125: return "Sky Blue" if mode == "Bright" else "Royal Blue"
    elif 125 <= h < 145: return "Indigo" if mode == "Bright" else "Navy"
    elif 145 <= h < 160: return "Lavender" if mode == "Bright" else "Deep Purple"
    elif 160 <= h < 175: return "Magenta" if mode == "Bright" else "Wine"
    
    return "Other"

# 3. 5x5 그리드 투표 방식
def detect_color_by_grid(image_crop, label=""):
    if image_crop is None or image_crop.size == 0: return "Unknown"

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
    for i in range(1, rows):
        cv2.line(image, (x1, y1 + i * step_h), (x2, y1 + i * step_h), color, 1)
    for j in range(1, cols):
        cv2.line(image, (x1 + j * step_w, y1), (x1 + j * step_w, y2), color, 1)

# === [수정됨] 심플한 패션 평가 엔진 (띄어쓰기 수정 완료) ===
def evaluate_outfit(top, bot):
    if top == "Unknown" or bot == "Unknown": 
        return "Detecting..."

    # 1. 톤온톤
    tone_pairs = [
        ("Sky Blue", "Royal Blue"), ("Royal Blue", "Sky Blue"),
        ("Beige", "Brown"), ("Brown", "Beige"),
        ("Lime", "Forest"), ("Forest", "Lime"),
        ("Lavender", "Deep Purple"), ("Deep Purple", "Lavender"),
        ("Cyan", "Teal"), ("Teal", "Cyan"),
        ("Indigo", "Navy"), ("Navy", "Indigo"),
        ("Magenta", "Wine"), ("Wine", "Magenta")
    ]
    if (top, bot) in tone_pairs:
        return "Elegant Tone-on-Tone"
    
    if top == bot:
        return "Good Identical color"

    # 2. 얼스룩
    earth_colors = ["Brown", "Beige", "Olive", "Forest", "Green"]
    if top in earth_colors and bot in earth_colors:
        return "Natural Earth Look"

    # 3. 무채색
    neutrals = ["Black", "White", "Grey", "Navy", "Beige", "Denim"]
    if top in neutrals or bot in neutrals:
        darks = ["Black", "Brown", "Navy", "Deep Purple", "Wine", "Forest"]
        if top in darks and bot in darks:
            return "Too Dark?"
        return "Safe Balance"

    # 4. 보색 포인트
    if (top == "Yellow" and bot in ["Navy", "Royal Blue"]) or \
       (bot == "Yellow" and top in ["Navy", "Royal Blue"]):
        return "Active Pop Style"

    return "Bad Choice..."

# 4. 웹캠 실행 및 메인 루프
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("실행 중... (5x5 Grid Voting Mode)")

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
            bot_y2 = y1 + int(height * 0.90)

            top_crop = frame[top_y1:top_y2, x1:x2]
            bot_crop = frame[bot_y1:bot_y2, x1:x2]

            top_color = detect_color_by_grid(top_crop, "Top")
            bot_color = detect_color_by_grid(bot_crop, "Bot")

            draw_grid(annotated_frame, x1, top_y1, x2, top_y2, color=(255, 255, 0)) 
            draw_grid(annotated_frame, x1, bot_y1, x2, bot_y2, color=(255, 0, 255))

            info_text = f"Top: {top_color} | Bot: {bot_color}"
            eval_text = evaluate_outfit(top_color, bot_color)
            
            (tw, th), _ = cv2.getTextSize(eval_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            text_y = y1 + 45  

            # 배경 박스 그리기
            cv2.rectangle(annotated_frame, (x1, text_y - th*2 - 15), (x1 + tw + 100, text_y + 10), (0, 0, 0), -1)
            
            # 텍스트 쓰기
            cv2.putText(annotated_frame, info_text, (x1+5, text_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(annotated_frame, eval_text, (x1+5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.rectangle(annotated_frame, (x1, top_y1), (x2, top_y2), (255, 255, 0), 1)
            cv2.rectangle(annotated_frame, (x1, bot_y1), (x2, bot_y2), (255, 0, 255), 1)

    cv2.imshow('Fashion AI - Debug Mode', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()