# Hybrid Detection and Tracking System
# YOLO Detection + Custom CNN Classification + DeepSORT Tracking
# Author: Zaamin Qadeer W1906890
#
# Pipeline:
# YOLOv8s - finds vehicles in each frame and draws bounding boxes
# Custom CNN (ResNet50) - classifies the vehicle type from the cropped region
# DeepSORT - assigns persistent IDs and tracks vehicles across frames
#
# Key changes made for dashcam footage:
#NMS before DeepSORT to remove duplicate boxes on the same vehicle
#ROI filter ignores sky (top 20%) and bonnet (bottom 8%)
#Confidence floor of 0.40 to reduce false positives from night glare
#max_iou_distance=0.99 disables IoU association (unreliable on moving camera)
#MobileNet embedder used for appearance based re-identification instead
#n_init=3 requires 3 frames to confirm a track (stops junk night tracks)
# max_age=30 drops stale tracks quickly (vehicles leave frame fast on dashcam)

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
import sys
import json
from collections import defaultdict

sys.path.append(".")
from customcnn import CustomVehicleCNN


def _nms_detections(detections, iou_threshold=0.45):
    # removes overlapping boxes on the same vehicle before passing to DeepSORT
    # stops multiple IDs being assigned to one vehicle on night footage
    if len(detections) <= 1:
        return detections

    # convert to x1,y1,x2,y2 format for IoU calculation
    boxes, confs, classes = [], [], []
    for (x, y, w, h), conf, cls in detections:
        boxes.append([x, y, x + w, y + h])
        confs.append(conf)
        classes.append(cls)

    boxes = np.array(boxes, dtype=float)
    confs = np.array(confs, dtype=float)

    # sort by confidence so we keep the best detection first
    order = confs.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        # calculate IoU between top box and all remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter   = inter_w * inter_h

        area_i    = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union     = area_i + area_rest - inter

        iou = np.where(union > 0, inter / union, 0.0)

        # only keep boxes that dont overlap with the current best box
        order = order[1:][iou < iou_threshold]

    return [detections[i] for i in keep]


# one colour per track ID so vehicles are easy to follow on screen
COLOURS = [
    (0, 255, 0),   (0, 128, 255), (255, 0, 128), (255, 200, 0),
    (0, 200, 255), (200, 0, 255), (255, 100, 0), (0, 255, 180),
    (180, 255, 0), (255, 0, 200), (100, 200, 255),(255, 180, 100),
]

def get_colour(track_id):
    # track_id comes back as a string from DeepSORT so we convert to int first
    return COLOURS[int(track_id) % len(COLOURS)]


class VehicleTracker:
    # main class that runs the full detection classification and tracking pipeline

    def __init__(self, custom_model_path: str, video_path: str, output_dir: str = "output_hybrid"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {self.device}")

        # load YOLOv8s for vehicle detection
        print("[INFO] Loading YOLOv8s detector ...")
        self.yolo = YOLO("yolov8s.pt")

        # only care about these COCO classes (vehicles)
        self.yolo_vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        # load the custom CNN classifier from saved checkpoint
        print("[INFO] Loading Custom CNN classifier ...")
        self.custom_cnn = CustomVehicleCNN(num_classes=4, pretrained=False)
        ckpt = torch.load(custom_model_path, map_location=self.device)
        self.custom_cnn.load_state_dict(ckpt["model_state_dict"])
        self.custom_cnn.to(self.device)
        self.custom_cnn.eval()  # disables dropout for inference

        self.cnn_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.class_names = ["Car", "Motorcycle", "Bus", "Truck"]

        # setup DeepSORT with dashcam specific parameters
        print("[INFO] Initialising DeepSORT tracker ...")
        # max_iou_distance=0.99 disables IoU association because on a moving camera
        # the kalman filter predictions are unreliable so we use appearance instead
        # max_age=30 drops tracks quickly since vehicles leave dashcam frame fast
        # n_init=3 prevents noisy night detections from spawning fake tracks
        # nn_budget=100 keeps gallery small as vehicle appearance changes fast on dashcam
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_iou_distance=0.99,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=(self.device == "cuda"),
        )

        # counters and history for stats and trajectory drawing
        self.frame_count      = 0
        self.total_detections = 0
        self.unique_tracks    = set()
        self.id_switches      = 0
        self.class_counts     = defaultdict(int)
        self.track_history    = defaultdict(list)

    def _classify(self, crop: np.ndarray):
        # runs the custom CNN on a cropped vehicle region
        # returns class index and confidence score
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return 0, 0.5
        try:
            tensor = self.cnn_transform(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():  # no gradients needed at inference time
                logits = self.custom_cnn(tensor)
                probs  = torch.softmax(logits, dim=1)
                conf, cls = torch.max(probs, dim=1)
            return cls.item(), conf.item()
        except Exception:
            return 0, 0.5  # fallback to car class if anything goes wrong

    @staticmethod
    def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45):
        # standard NMS using torch tensors
        # returns list of indices to keep
        if boxes.numel() == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort(descending=True)
        keep   = []

        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]

            # IoU of top box vs remaining boxes
            xx1 = torch.clamp(x1[rest], min=x1[i].item())
            yy1 = torch.clamp(y1[rest], min=y1[i].item())
            xx2 = torch.clamp(x2[rest], max=x2[i].item())
            yy2 = torch.clamp(y2[rest], max=y2[i].item())

            inter_w = torch.clamp(xx2 - xx1, min=0)
            inter_h = torch.clamp(yy2 - yy1, min=0)
            inter   = inter_w * inter_h

            iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
            order = rest[iou <= iou_threshold]

        return keep

    def process_video(self, confidence_threshold: float = 0.35, display: bool = False):
        # main loop that processes each frame through the full pipeline
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Video: {width}x{height} @ {fps} fps | {total_frames} frames")

        output_path = self.output_dir / "tracked_output_hybrid.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # step 1: run YOLO to get vehicle detections
            results = self.yolo(frame, verbose=False)[0]
            raw_detections = []

            for box in results.boxes:
                yolo_cls = int(box.cls[0])
                conf     = float(box.conf[0])

                if yolo_cls not in self.yolo_vehicle_classes or conf < confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                # clamp coordinates to stay within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                w, h   = x2 - x1, y2 - y1

                # skip boxes that are too small (likely noise or reflections)
                if w < 40 or h < 40:
                    continue

                # skip tall narrow boxes (not vehicle shaped, probably signs or poles)
                if (w / h) < 0.4:
                    continue

                # skip detections in sky area at the top of frame
                if y2 < int(height * 0.20):
                    continue

                # skip detections on the bonnet at the bottom of frame
                if y1 > int(height * 0.92):
                    continue

                # hard confidence floor to reduce night glare false positives
                if conf < 0.40:
                    continue

                # step 2: classify the vehicle type using the custom CNN
                crop = frame[y1:y2, x1:x2]
                cnn_cls, cnn_conf = self._classify(crop)

                # DeepSORT needs format ([left, top, w, h], confidence, class)
                raw_detections.append(([x1, y1, w, h], cnn_conf, cnn_cls))
                self.total_detections += 1

            # step 3: apply NMS to remove duplicate boxes before tracking
            # stops one vehicle getting multiple IDs from overlapping detections
            raw_detections = _nms_detections(raw_detections, iou_threshold=0.35)

            # step 4: update DeepSORT tracker with current detections
            tracks = self.tracker.update_tracks(raw_detections, frame=frame)

            # step 5: draw annotations on confirmed tracks
            active_ids = 0
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                self.unique_tracks.add(track_id)
                active_ids += 1

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # get class name from track or fall back to generic label
                det_class = getattr(track, "det_class", None)
                if det_class is not None and int(det_class) < len(self.class_names):
                    class_name = self.class_names[int(det_class)]
                else:
                    class_name = "Vehicle"

                self.class_counts[class_name] += 1

                # store centroid position for trajectory trail drawing
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.track_history[track_id].append((self.frame_count, cx, cy))

                colour = get_colour(track_id)

                # draw coloured bounding box around vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # draw label with coloured background for readability
                label = f"ID:{track_id}  {class_name}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), colour, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # draw trajectory trail using last 30 stored positions
                history = self.track_history[track_id][-30:]
                for i in range(1, len(history)):
                    cv2.line(frame,
                             (history[i-1][1], history[i-1][2]),
                             (history[i][1],   history[i][2]),
                             colour, 1)

            # draw HUD showing frame progress and track counts
            hud_lines = [
                f"Frame: {self.frame_count}/{total_frames}",
                f"Active tracks: {active_ids}",
                f"Total vehicles seen: {len(self.unique_tracks)}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(frame, line, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.putText(frame, "W1906890 | YOLOv8s + Custom CNN + DeepSORT",
                        (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            out.write(frame)

            if display:
                cv2.imshow("Hybrid Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.frame_count % 50 == 0:
                pct = self.frame_count / total_frames * 100
                print(f"  Progress: {pct:.1f}% | Tracks so far: {len(self.unique_tracks)}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"\n[DONE] Output: {output_path}")
        print(f"       Frames processed : {self.frame_count}")
        print(f"       Total detections : {self.total_detections}")
        print(f"       Unique track IDs : {len(self.unique_tracks)}")
        return output_path

    def generate_report(self):
        # saves tracking results as JSON and plain text files
        report = {
            "system": {
                "detection"     : "YOLOv8s",
                "classification": "Custom CNN (ResNet50 backbone, 4-class vehicle head)",
                "tracking"      : "DeepSORT (max_age=30, n_init=3, nn_budget=100, dashcam-optimised)",
                "dataset"       : "Trained on COCO128"
            },
            "results": {
                "frames_processed": self.frame_count,
                "total_detections": self.total_detections,
                "unique_track_ids": len(self.unique_tracks),
                "class_breakdown" : dict(self.class_counts),
            },
            "tracker_params": {
                "max_age"         : 30,
                "n_init"          : 3,
                "nn_budget"       : 100,
                "max_iou_distance": 0.99,
                "embedder"        : "mobilenet",
            }
        }

        json_path = self.output_dir / "tracking_report_hybrid.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        txt_path = self.output_dir / "tracking_report_hybrid.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TRACKING REPORT - W1906890 Zaamin Qadeer\n")
            f.write("=" * 60 + "\n\n")
            f.write("SYSTEM\n")
            f.write(f"  Detection     : YOLOv8s\n")
            f.write(f"  Classification: Custom CNN (ResNet50)\n")
            f.write(f"  Tracking      : DeepSORT (dashcam optimised)\n\n")
            f.write("RESULTS\n")
            f.write(f"  Frames processed : {self.frame_count}\n")
            f.write(f"  Total detections : {self.total_detections}\n")
            f.write(f"  Unique track IDs : {len(self.unique_tracks)}\n\n")
            f.write("VEHICLE CLASS BREAKDOWN\n")
            for cls, cnt in self.class_counts.items():
                f.write(f"  {cls:<12}: {cnt}\n")
            f.write("\nTRACKER PARAMETERS\n")
            f.write(f"  max_age          : 30\n")
            f.write(f"  n_init           : 3\n")
            f.write(f"  nn_budget        : 100\n")
            f.write(f"  max_iou_distance : 0.99\n")
            f.write(f"  embedder         : MobileNet\n")

        print(f"[INFO] Reports saved to {self.output_dir}")
        return txt_path


if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Vehicle Tracking System")
    print("YOLOv8s + Custom CNN + DeepSORT")
    print("Zaamin Qadeer W1906890")
    print("=" * 60 + "\n")

    CUSTOM_MODEL = "custom_cnn_output/best_custom_model.pth"
    VIDEO_FILE   = "traffic_video.mp4"
    CONFIDENCE   = 0.45  # higher threshold for dashcam to reduce night glare false positives
    DISPLAY_LIVE = False

    for path, name in [(CUSTOM_MODEL, "Custom CNN model"), (VIDEO_FILE, "Video file")]:
        if not Path(path).exists():
            print(f"[ERROR] {name} not found at '{path}'")
            exit(1)

    try:
        system = VehicleTracker(CUSTOM_MODEL, VIDEO_FILE)
        system.process_video(confidence_threshold=CONFIDENCE, display=DISPLAY_LIVE)
        system.generate_report()
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()