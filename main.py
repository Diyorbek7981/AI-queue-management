import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from model import Person
from database import session, engine

# ==== Config ====
MODEL_PATH = 'yolov8n.pt'
VIDEO_SOURCE = 'queue_videos.mp4'
WAITING_MASK_PATH = 'pic/waiting.jpg'
SERVING_MASK_PATH = 'pic/service.jpg'
STUF_MASK_PATH = 'pic/staf.jpg'
OUTPUT_DIR = 'output_videos'  # Video run bolgandan keyin uni saqlsh pathi
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = session(bind=engine)


# ==== Queue Tracker ====
class QueueTracker:
    def __init__(self, model, waiting_mask_path, serving_mask_path, stuf_mask_path, frame_size, exclude_ids=None):
        self.model = model

        # Maskalarni grayscale holatda yuklab, videoga moslashtiramiz
        self.waiting_mask = cv2.imread(waiting_mask_path, cv2.IMREAD_GRAYSCALE)
        self.serving_mask = cv2.imread(serving_mask_path, cv2.IMREAD_GRAYSCALE)
        self.stuf_mask = cv2.imread(stuf_mask_path, cv2.IMREAD_GRAYSCALE)

        # Maskalarni resize qilish
        if self.waiting_mask is not None:
            self.waiting_mask = cv2.resize(self.waiting_mask, (frame_size[0], frame_size[1]))
        if self.serving_mask is not None:
            self.serving_mask = cv2.resize(self.serving_mask, (frame_size[0], frame_size[1]))
        if self.stuf_mask is not None:
            self.stuf_mask = cv2.resize(self.stuf_mask, (frame_size[0], frame_size[1]))

        # Kutish va xizmat vaqtlarini hisoblash uchun
        self.stuff_enter_time = {}  # Hodimlar kadrga kirgan vaqt
        self.enter_time = {}  # Obyekt kadrga  kirgan vaqt  (kutish zonasi uchun)
        self.start_service = {}  # Xizmat ko‘rsatish jarayoni boshlangan vaqtni saqlaydi.
        self.exclude_ids = set(exclude_ids) if exclude_ids else set()  # E’tiborga olinmaydigan ID’larni saqlaydi.
        self.is_servise = {}  # servise korsatilgan yoki yoqligini tekshirish (True, False korinishida)

    # Aniqlangan odam markaziy nuqtasi maska ichidami yoki yoqligi tekshiriladi
    def is_inside_mask(self, mask, x, y):
        # Markaz mask ichida ekanligini tekshiradi
        if mask is None:
            return False
        h, w = mask.shape
        if 0 <= y < h and 0 <= x < w:  # odam markazi maska shapidan tashqariga chiqib ketmasligi uchun
            return mask[y, x] > 0
        return False

    def update(self, frame):
        # YOLO modelidan natijani olish (odam kardinatalari)
        results = self.model.track(frame, persist=True, classes=0, conf=0.5)[0]
        current_time = datetime.now()
        serving_count, waiting_count, stuff_count = 0, 0, 0

        if results.boxes.id is None:
            return frame

        active_ids = set()  # id larni tekshirish uchun (kadrda yoki kadrdan chiqib ketmagankigini)

        for box, track_id in zip(results.boxes.xyxy, results.boxes.id):
            track_id = int(track_id)

            active_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box[:4])  # odam kordinatasi 2  ta nuqta joylashuvi
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # odam kordinatasi markazi

            # ❗ Birinchi kirganda enter_time ni yozish
            person = session.query(Person).filter_by(track_id=track_id).first()
            if not person:
                person = Person(
                    track_id=track_id,
                    enter_time=current_time  # ✅ Kirish vaqti
                )
                session.add(person)
                session.commit()

            # Hudud ichida ekanligini aniqlash
            in_serving = self.is_inside_mask(self.serving_mask, cx, cy)
            in_waiting = self.is_inside_mask(self.waiting_mask, cx, cy)
            in_stuff = self.is_inside_mask(self.stuf_mask, cx, cy)

            # Kutish vaqtini hisoblash
            if in_waiting and not self.is_servise.get(track_id, False):
                if track_id not in self.enter_time:
                    self.enter_time[track_id] = current_time
                wait_time = (current_time - self.enter_time[track_id]).total_seconds()
            else:
                wait_time = (current_time - self.enter_time.get(track_id, current_time)).total_seconds()
                person = session.query(Person).filter_by(track_id=track_id).first()
                if not self.is_servise.get(track_id, False):
                    person.wait_time = wait_time
                    session.commit()

            # Xizmat vaqtini hisoblash
            if in_serving:
                if track_id not in self.start_service:
                    self.start_service[track_id] = current_time
                service_secs = (current_time - self.start_service[track_id]).total_seconds()

                if service_secs >= 5:
                    self.is_servise[track_id] = True
            else:
                service_secs = (current_time - self.start_service.get(track_id, current_time)).total_seconds()
                person = session.query(Person).filter_by(track_id=track_id).first()
                if person.service_time == 0:
                    person.service_time = service_secs
                    session.commit()

            # Hodimlar vaqtini hisoblash
            if in_stuff:
                if track_id not in self.stuff_enter_time:
                    # Agar hodim birinchi marta kadrga kirsa — vaqtni saqlaymiz
                    self.stuff_enter_time[track_id] = current_time
                # Hodim kadrga kirganidan beri qancha vaqt o'tganini hisoblaymiz
                stuff_time = (current_time - self.stuff_enter_time[track_id]).total_seconds()
            else:
                # Hodim kadrdan chiqib ketgan bo'lsa, vaqtni tozalaymiz
                self.stuff_enter_time.pop(track_id, None)

            # Status va rang
            if in_serving:
                status = f"Serving: {service_secs:.0f}s"
                color = (0, 255, 0)
                serving_count += 1
            elif in_waiting and not self.is_servise.get(track_id, False):
                status = f"Waiting: {wait_time:.0f}s"
                color = (0, 0, 255)
                waiting_count += 1
            elif in_stuff:
                status = f"Stuff: {stuff_time:.0f}s"
                color = (255, 0, 0)
                stuff_count += 1
            else:
                status = "Outside"
                color = (200, 200, 200)

            # Bounding box va status yozuvini chiqarish
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{track_id} {status}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        #  ✔️ Kadrdan chiqqan odamlarni aniqlash
        finished_ids = set(self.enter_time.keys()) | set(self.start_service.keys()) | set(
            self.stuff_enter_time.keys())
        print(finished_ids)
        finished_ids -= active_ids  # endi faqat chiqib ketgan odamlar qoladi

        if finished_ids is not None:
            for i in finished_ids:

                person = session.query(Person).filter_by(track_id=i).first()
                if person and person.exit_time is None:
                    person.exit_time = current_time
                    session.commit()

                self.enter_time.pop(i, None)
                self.start_service.pop(i, None)
                self.stuff_enter_time.pop(i, None)

        #         # ❗ Id yoqolgach osha frameni saqlab olish
        #         # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #         # save_path = os.path.join(OUTPUT_DIR, f"finished_{'_'.join(map(str, finished_ids))}_{timestamp}.jpg")
        #         # cv2.imwrite(save_path, frame)
        #         # print(f"Kadr saqlandi: {save_path}")

        # Statistikani chiqarish
        cv2.putText(frame, f"Serving: {serving_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Waiting: {waiting_count}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Stuff: {stuff_count}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # ✔️ Mask overlay qilish — faqat o‘lcham mos bo‘lsa
        # Mask overlay qilish — bu tasvir ustiga niqob (mask)ni qisman yoki shaffof tarzda qo‘yish (maskani boyab korsatasi)
        if self.waiting_mask is not None:
            waiting_colored = cv2.merge([self.waiting_mask, self.waiting_mask * 0, self.waiting_mask * 0])
            waiting_colored = cv2.resize(waiting_colored, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 1, waiting_colored, 0.2, 0)

        if self.serving_mask is not None:
            serving_colored = cv2.merge([self.serving_mask * 0, self.serving_mask, self.serving_mask * 0])
            serving_colored = cv2.resize(serving_colored, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 1, serving_colored, 0.2, 0)

        if self.stuf_mask is not None:
            stuff_colored = cv2.merge([self.stuf_mask * 0, self.stuf_mask * 0, self.stuf_mask])
            stuff_colored = cv2.resize(stuff_colored, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 1, stuff_colored, 0.2, 0)

        return frame


cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, frame = cap.read()

if not ret:
    print("Video topilmadi.")
    exit()

height, width = frame.shape[:2]
model = YOLO(MODEL_PATH)

# Classga malumotlarni yuborish
tracker = QueueTracker(model, WAITING_MASK_PATH, SERVING_MASK_PATH, STUF_MASK_PATH, (width, height))

# Output (xosil bolgan) videoni saqlash uchun uni yozib olish
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, 'queue_output_mask.mp4'), fourcc, 20, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = tracker.update(frame)
    out.write(result)
    cv2.imshow("Queue Monitor", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
