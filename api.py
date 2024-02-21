# Importing required packages
from flask import Flask, request, jsonify
import ultralytics
import sys
from dataclasses import dataclass
import supervision
from typing import List
from supervision.draw.color import ColorPalette
import base64

import numpy as np
from ultralytics import YOLO
from tqdm.notebook import tqdm
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List
import sys
import yolox
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

from ultralytics import YOLO

from tqdm.notebook import tqdm

model = YOLO('best.pt')


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

# settings
MODEL = "best.pt"
model = YOLO(MODEL)
model.fuse()

# CLASS_NAMES_DICT = model.model.names
CLASS_NAMES_DICT={}
CLASS_NAMES_DICT[0]='vehicle'
# print(CLASS_NAMES_DICT)
CLASS_ID = [0]



# Initiating a Flask application
app = Flask(__name__)


# The endpoint of our flask app
@app.route(rule="/detect", methods=["POST"])
def detect_objects_endpoint():
    # Check if request is POST
    if request.method == "POST":
        # Get video data from request
        SOURCE_VIDEO_PATH = request.files['video'].filename

    LINE_START = Point(4, 600)
    LINE_END = Point(1784, 600)

    TARGET_VIDEO_PATH = f"highway-result.mp4"
    VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    # create frame generator
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
    # create LineCounter instance
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    # line_counter1 = LineCounter(start=LINE_START1, end=LINE_END1)
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    line_annotator = LineCounterAnnotator(thickness=2, text_thickness=4, text_scale=2)

    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        # loop over video frames
        for frame in tqdm(generator, total=video_info.total_frames):
            # model prediction on single frame and conversion to supervision Detections
            results = model(frame)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections   
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # format custom labels
            labels = [
                f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            # updating line counter
            line_counter.update(detections=detections)
            # line_counter1.update(detections=detections)

            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            # show_frame_in_notebook(frame, (16, 16))
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            # line_annotator.annotate(frame=frame, line_counter=line_counter1)
            sink.write_frame(frame)

        
        # Encode the video file as base64
    with open('highway-result.mp4', 'rb') as f:
        video_data = base64.b64encode(f.read()).decode('utf-8')

    response = {'annotated_video': video_data}
    return jsonify(response)

# Running the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
