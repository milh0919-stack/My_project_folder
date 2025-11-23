import cv2
from ultralytics import YOLO

# 1. YOLO 모델 불러오기
# 'yolov8n.pt'는 가장 가볍고 빠른 모델입니다. (처음 실행 시 자동 다운로드됨)
model = YOLO('yolov8n.pt')

# 2. 웹캠 켜기 (0번은 기본 노트북 캠)
cap = cv2.VideoCapture(0)

# 캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("실행 중... 종료하려면 'q' 키를 누르세요.")

while True:
    # 3. 프레임(이미지) 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 4. YOLO로 물체 탐지 (Inference)
    # verbose=False는 터미널에 지저분한 로그가 안 뜨게 합니다.
    results = model(frame, verbose=False)

    # 5. 결과 시각화
    # plot() 함수는 탐지된 물체에 네모 박스와 이름을 자동으로 그려줍니다.
    annotated_frame = results[0].plot()

    # 6. 화면에 출력
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # 7. 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. 자원 해제
cap.release()
cv2.destroyAllWindows()