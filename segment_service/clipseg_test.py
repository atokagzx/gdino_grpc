#!/usr/bin/env python3

from modules.pb2_init import *
import time
import logging
import grpc
import cv2
import numpy as np
import threading
from typing import Generator, Tuple, Union
from modules.items_dataclasses import DetectedItem, SegmentedItem, Mask
from modules.detection_client_wrapper import DetectionClient
from modules.segmentation_client_wrapper import SegmentationClient
from modules.clipseg_client_wrapper import CLIPSegClient

class VideoCaptureBuffless(cv2.VideoCapture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("video_capture_buffless")
        self.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        super().grab()
        return super().retrieve()

class LoopSegmentation:
    def __init__(self, frames_provider: cv2.VideoCapture, detector_iterator: DetectionClient, segmentation_client: CLIPSegClient):
        self._logger = logging.getLogger("loop_segmentation")
        self._frames_provider = frames_provider
        self._detector_iterator = detector_iterator
        self._segmentation_client = segmentation_client
        self._last_result = None
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        for frame, detection_result in self._detector_iterator:
            if self._stop:
                break
            ret, new_frame = self._frames_provider.read()
            if not ret:
                self._logger.info("failed to read new frame (end of video?)")
                break
            self._detector_iterator.add_new_frame(new_frame)
            # crop frame
            crop = frame[0:100, 0:100]
            result = self._segmentation_client.process(frame, detection_result, crop)
            self._last_result = {
                'frame': result[0],
                'segmented_items': result[1]
            }

    @property
    def last_result(self):
        return self._last_result
    
    def stop(self):
        self._logger.info("stopping segmentation loop")
        self._stop = True
        self._thread.join()
        self._logger.info("segmentation loop stopped")

    @property
    def is_running(self):
        return self._thread.is_alive()

def draw_boxes(image, boxes, phrases, logits):
    for box, phrase, logit in zip(boxes, phrases, logits):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(image, phrase, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(image, f"d_score: {logit:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return image

def draw_box_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = None):
        # mask from (H, W) to (H, W, 1)
        mask = np.expand_dims(mask, axis=-1)
        if color is None:
            color = np.random.uniform(0, 1, size=3)
        rgb_mask = np.array([mask.copy() for _ in range(3)])
        colored_mask = np.concatenate(rgb_mask, axis=-1) * color
        colored_mask = np.array(colored_mask, dtype=np.uint8)
        image = cv2.addWeighted(image, 1, colored_mask, 0.4, 0)
        return image

def main(segmentation_loop: LoopSegmentation, window_name: str = 'segmented items'):
    logger = logging.getLogger("main")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cat2color = {}
    while segmentation_loop.is_running:
        if segmentation_loop.last_result is None:
            time.sleep(0.1)
            continue
        frame = segmentation_loop.last_result['frame']
        segmented_items = segmentation_loop.last_result['segmented_items']
        boxes = [item.bounding_box for item in segmented_items]
        phrases = [item.label for item in segmented_items]
        detection_confidences = [item.detection_confidence for item in segmented_items]
        frame = draw_boxes(frame, boxes, phrases, detection_confidences)
        for masks, label in zip([item.masks for item in segmented_items], phrases):
            for mask in masks:
                mask_array = mask.mask
                if label not in cat2color:
                    cat2color[label] = np.random.uniform(0, 255, size=3).astype(np.uint8)
                frame = draw_box_mask(frame, mask_array, color=cat2color[label])
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        logger.info("segmentation loop stopped")

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    grpc_channel = grpc.insecure_channel('localhost:50051')
    frames_provider = VideoCaptureBuffless(0)
    frames_provider.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    frames_provider.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, frame = frames_provider.read()
    if not ret:
        raise RuntimeError("failed to read from camera")
    detector_iterator = DetectionClient(grpc_channel=grpc_channel,
                                    first_frame=frame,
                                    prompt='monitor . cup . human . phone . pc mouse',
                                    box_threshold=0.05,
                                    text_threshold=0.3, 
                                    confidence_threshold=0.2)
    segmentor_client = CLIPSegClient(grpc_channel=grpc_channel)
    try:
        segmentation_loop = LoopSegmentation(frames_provider, detector_iterator, segmentor_client)
        main(segmentation_loop)
    except KeyboardInterrupt:
        segmentation_loop.stop()
        frames_provider.release()
        cv2.destroyAllWindows()
        