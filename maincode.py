import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# 1. Import YOLO Model
model = YOLO('yolov8n.pt')

# 23 color classification logic
def get_color_category(h, s, v):
    # 1. Achromatic
    if s < 20:
        if v > 180: return "White"
        elif v < 90: return "Black"
        else: return "Grey"

    # 2. Chromatic
    mode = "Bright" if v > 130 else "Dark"

    # Mapping the Hue range (0 to 179) by dividing it into 10 equal parts
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

# 3. 5x5 Grid Voting Method
# Error Countermeasures
def detect_color_by_grid(image_crop, label=""):
    if image_crop is None or image_crop.size == 0: return "Unknown"
# h,s,v conversion 
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape

    grid_rows = 5
    grid_cols = 5
    step_h = h // grid_rows
    step_w = w // grid_cols

    votes = [] 
# Location-based color judgment
    for i in range(grid_rows):
        for j in range(grid_cols):
            y1 = i * step_h
            y2 = (i + 1) * step_h
            x1 = j * step_w
            x2 = (j + 1) * step_w
            
            cell = hsv[y1:y2, x1:x2]
            # color judgment
            if cell.size > 0:
                mean_color = np.mean(cell, axis=(0, 1))
                category = get_color_category(mean_color[0], mean_color[1], mean_color[2])
                votes.append(category)
# Error Countermeasures
    if not votes: return "Unknown"
# Return the Mode    
    vote_result = Counter(votes)
    most_common_color = vote_result.most_common(1)[0][0]
    
    return most_common_color

# 4. Helper function to draw a 5x5 grid on the screen
def draw_grid(image, x1, y1, x2, y2, rows=5, cols=5, color=(200, 200, 200)):
    step_h = (y2 - y1) // rows
    step_w = (x2 - x1) // cols
    for i in range(1, rows):
        cv2.line(image, (x1, y1 + i * step_h), (x2, y1 + i * step_h), color, 1)
    for j in range(1, cols):
        cv2.line(image, (x1 + j * step_w, y1), (x1 + j * step_w, y2), color, 1)

# 5. A simple fashion evaluation engine
# Error Countermeasures
def evaluate_outfit(top, bot):
    if top == "Unknown" or bot == "Unknown": 
        return "Detecting..."

    # 1. Tone-on-tone
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

    # 2. Earth tone
    earth_colors = ["Brown", "Beige", "Olive", "Forest", "Green"]
    if top in earth_colors and bot in earth_colors:
        return "Natural Earth Look"

    # 3. achromatic color
    neutrals = ["Black", "White", "Grey", "Navy", "Beige", "Denim"]
    if top in neutrals or bot in neutrals:
        darks = ["Black", "Brown", "Navy", "Deep Purple", "Wine", "Forest"]
        if top in darks and bot in darks:
            return "Too Dark"
        return "Safe Balance"

    # 4. Tone in tone(contrast)
    if (top == "Yellow" and bot in ["Navy", "Royal Blue"]) or \
       (bot == "Yellow" and top in ["Navy", "Royal Blue"]):
        return "Active Pop Style"

    return "Bad Choice..."

# 4. Running Webcam and Main Loop
cap = cv2.VideoCapture(0)
# Adjust if the room is too dark or bright (integer between -13 and -1)
cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
# Error Countermeasures
if not cap.isOpened():
    print("error...")
    exit()

print("Running...")

while True:
    ret, frame = cap.read()
    if not ret: break
    # Using YOLO & creating visualization image
    results = model(frame, classes=[0], verbose=False)
    annotated_frame = frame.copy()
# vertex coordinates in the bounding box
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # a person's box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Dividing the upper and lower body boxes
            height = y2 - y1
            top_y1 = y1 + int(height * 0.15)
            top_y2 = y1 + int(height * 0.45)
            bot_y1 = y1 + int(height * 0.50)
            bot_y2 = y1 + int(height * 0.90)

            top_crop = frame[top_y1:top_y2, x1:x2]
            bot_crop = frame[bot_y1:bot_y2, x1:x2]
            # color judgment
            top_color = detect_color_by_grid(top_crop, "Top")
            bot_color = detect_color_by_grid(bot_crop, "Bot")
            # Drawing a visualization grid
            draw_grid(annotated_frame, x1, top_y1, x2, top_y2, color=(255, 255, 0)) 
            draw_grid(annotated_frame, x1, bot_y1, x2, bot_y2, color=(255, 0, 255))
            # Color information and evaluation
            info_text = f"Top: {top_color} | Bot: {bot_color}"
            eval_text = evaluate_outfit(top_color, bot_color)
            # Text Size Measurement
            (tw, th), _ = cv2.getTextSize(eval_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            text_y = y1 + 45  

            # Draw Background Boxes
            cv2.rectangle(annotated_frame, (x1, text_y - th*2 - 15), (x1 + tw + 100, text_y + 10), (0, 0, 0), -1)
            
            # Write text
            cv2.putText(annotated_frame, info_text, (x1+5, text_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(annotated_frame, eval_text, (x1+5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.rectangle(annotated_frame, (x1, top_y1), (x2, top_y2), (255, 255, 0), 1)
            cv2.rectangle(annotated_frame, (x1, bot_y1), (x2, bot_y2), (255, 0, 255), 1)
    # Visualization
    cv2.imshow('Fashion AI', annotated_frame)
# 5. Waiting for the break command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()