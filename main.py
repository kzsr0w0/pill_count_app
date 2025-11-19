'''
YOLOで錠剤カウント
作成日：20251113
最終編集日：20251120

'''
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict, deque

st.title("リアルタイム錠剤カウント（ROIなし・マスク対応）")

# -----------------------
# モデル読み込み
# -----------------------
# MODEL_PATH = "yolov8n-seg.pt"
MODEL_PATH = "study_pill_model.pt"
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.5
model = YOLO(MODEL_PATH)

# -----------------------
# カメラ入力
# -----------------------
st.write("カメラ映像を表示し、錠剤をカウントします。")
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

# -----------------------
# ID色生成用
# -----------------------
track_history = defaultdict(lambda: deque(maxlen=30))
def generate_color(track_id):
    np.random.seed(int(track_id))
    return tuple(map(int, np.random.randint(0,255,3)))

# -----------------------
# カウントループ
# -----------------------
frame_count = 0
max_pills_seen = 0

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("カメラから映像を取得できません")
        break
    frame_count += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO推論（tracking）
    results = model.track(img_rgb, persist=True, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    current_count = 0

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else list(range(len(boxes)))
        masks = results[0].masks

        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x1, y1, x2, y2 = map(int, box)
            center = (int((x1+x2)/2), int((y1+y2)/2))
            current_count += 1
            color = generate_color(track_id)

            # セグメンテーション描画
            if masks is not None:
                mask = masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
                colored_mask = np.zeros_like(frame)
                colored_mask[mask==1] = color
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
                # 輪郭
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)
                # ID表示
                if len(contours) > 0:
                    M = cv2.moments(contours[0])
                    cX = int(M["m10"]/M["m00"]) if M["m00"] != 0 else center[0]
                    cY = int(M["m01"]/M["m00"]) if M["m00"] != 0 else center[1]
                    label = f'{track_id}'
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7,2)
                    cv2.rectangle(frame, (cX-label_w//2-5, cY-label_h//2-5),
                                  (cX+label_w//2+5, cY+label_h//2+5), (0,0,0), -1)
                    cv2.putText(frame, label, (cX-label_w//2, cY+label_h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                cv2.rectangle(frame, (x1,y1),(x2,y2), color,2)
                cv2.putText(frame, f'{track_id}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2)

    # 現在カウント表示
    count_text = f"Pills: {current_count}"
    (text_w, text_h), baseline = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(frame, (10,10), (20+text_w, 30+text_h), (0,0,0), -1)
    cv2.putText(frame, count_text, (10, 10+text_h+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

    # Streamlit表示
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # 最大数更新
    max_pills_seen = max(max_pills_seen, current_count)


'''
【履歴】
20251113  作成
20251120  モデルを学習済みモデルに変更
'''
