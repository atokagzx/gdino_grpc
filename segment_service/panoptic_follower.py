#!/usr/bin/env python3

from modules.pb2_init import *
import time
import logging
import grpc
import cv2
import numpy as np
import threading
from typing import Generator, Union, List, Dict, Tuple, Iterable
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

# class LoopSegmentation:
#     def __init__(self, frames_provider: cv2.VideoCapture, detector_iterator: DetectionClient, segmentation_client: SegmentationClient):
#         self._logger = logging.getLogger("loop_segmentation")
#         self._frames_provider = frames_provider
#         self._detector_iterator = detector_iterator
#         self._segmentation_client = segmentation_client
#         self._last_result = None
#         self._stop = False
#         self._thread = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()

#     def _loop(self):
#         for frame, detection_result in self._detector_iterator:
#             if self._stop:
#                 break
#             ret, new_frame = self._frames_provider.read()
#             if not ret:
#                 self._logger.info("failed to read new frame (end of video?)")
#                 break
#             self._detector_iterator.add_new_frame(new_frame)
#             result = self._segmentation_client.process(frame, detection_result)
#             self._last_result = {
#                 'frame': result[0],
#                 'segmented_items': result[1]
#             }

#     @property
#     def last_result(self):
#         return self._last_result
    
#     def stop(self):
#         self._logger.info("stopping segmentation loop")
#         self._stop = True
#         self._thread.join()
#         self._logger.info("segmentation loop stopped")

#     @property
#     def is_running(self):
#         return self._thread.is_alive()
    
def draw_boxes(image, boxes, phrases, logits):
    for box, phrase, logit in zip(boxes, phrases, logits):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(image, phrase, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(image, f"d_score: {logit:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        # add weighted mask
        mask = np.zeros(image.shape, dtype=np.uint8)
        # random_color = np.random.uniform(0, 1, size=3) * 255
        mask[y1:y2, x1:x2] = [255, 0, 255]
        image = cv2.addWeighted(image, 1, mask, 0.8, 0)
    return image

def draw_box_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = None):
        # mask from (H, W) to (H, W, 1)
        mask = np.expand_dims(mask, axis=-1)
        if color is None:
            color = np.random.uniform(0, 1, size=3)
        rgb_mask = np.array([mask.copy() for _ in range(3)])
        colored_mask = np.concatenate(rgb_mask, axis=-1) * color
        colored_mask = np.array(colored_mask, dtype=np.uint8)
        image = cv2.addWeighted(image, 1, colored_mask, 0.2, 0)
        return image

class ImagePresenter:
    def __init__(self, frames_provider:cv2.VideoCapture, detector_iterator: DetectionClient, clipseg_clien: CLIPSegClient):
        self._logger = logging.getLogger("image_presenter")
        self._frames_provider = frames_provider
        self._detector_iterator = detector_iterator
        self._clipseg_client = clipseg_clien
        self._stop = False
        self._image_whiteboard = None
        self._selected_bbox = None
        self._last_detection_result = None
        self._thread = threading.Thread(target=self._detection_processor, daemon=True)
        self._thread.start()

    def _detection_processor(self):
        for frame, detection_result in self._detector_iterator:
            if self._stop:
                break
            ret, new_frame = self._frames_provider.read()
            if not ret:
                self._logger.info("failed to read new frame (end of video?)")
                break
            self._detector_iterator.add_new_frame(new_frame)
            detected_items = [DetectedItem(
                bounding_box = [getattr(detect_item.bounding_box, field) for field in ('x1', 'y1', 'x2', 'y2')],
                label = detect_item.label,
                detection_confidence = detect_item.confidence) for detect_item in detection_result.items]
            detected_items = sorted(detected_items, key=lambda item: (item.bounding_box[2] - item.bounding_box[0]) * (item.bounding_box[3] - item.bounding_box[1])) 
            if self._selected_bbox is None:
                clipseg_mask = None
            else:
                _frame, clipseg_mask = self._segmentation_processor(frame)
            self._last_detection_result = {
                'frame': frame,
                'detection_result': detected_items,
                'clipseg_mask': clipseg_mask
            }

    def _segmentation_processor(self, frame):
        crop = self._selected_bbox['image']
        print(f"crop shape: {crop.shape}")
        # cv2.imshow("crop", crop)
        # cv2.waitKey(1)
        result = self._clipseg_client.process(frame, crop)
        return result
        # cv2.imshow("mask", result[1])
        # cv2.waitKey(1)

    # def _filter_out_non_selected(self, detected_items: Iterable[DetectedItem]) -> Iterable[DetectedItem]:
    #     if self._selected_bbox is None:
    #         return detected_items
    #     filtered_by_label = [item for item in detected_items if item.label == self._selected_bbox.label]
    #     def bbox_similarity(bbox1: DetectedItem, bbox2: DetectedItem) -> float:
    #         x1, y1, x2, y2 = bbox1.bounding_box
    #         x1_, y1_, x2_, y2_ = bbox2.bounding_box
    #         return (x1 - x1_)**2 + (y1 - y1_)**2 + (x2 - x2_)**2 + (y2 - y2_)**2
    #     filtered_by_position = sorted(filtered_by_label, key=lambda item: bbox_similarity(item, self._selected_bbox))
    #     if not len(filtered_by_position):
    #         return []
    #     similarity = bbox_similarity(filtered_by_position[0], self._selected_bbox)
    #     if similarity > 70000:
    #         # print(f"similarity is {similarity}")
    #         return []
    #     nearest_bbox = filtered_by_position[0]
    #     if self._selected_bbox:
    #         self._selected_bbox = nearest_bbox
    #     return [nearest_bbox]
        
    @property
    def last_detection_result(self) -> Dict[str, Union[np.ndarray, List[DetectedItem]]]:
        return self._last_detection_result

    @property
    def selected_bbox(self) -> Dict[str, DetectedItem]:
        return self._selected_bbox
    
    @selected_bbox.setter
    def selected_bbox(self, value: Dict[str, DetectedItem]):
        # assert isinstance(value, DetectedItem) or value is None, "selected_bbox must be of type DetectedItem or None"
        '''
        @param value: dict with keys 'item' and 'image'
        '''
        self._selected_bbox = value
    
    def stop(self):
        self._logger.info("stopping image presenter")
        self._stop = True
        self._thread.join()
        self._logger.info("image presenter stopped")

    @property
    def is_running(self):
        return self._thread.is_alive()
    

def main(image_presenter: ImagePresenter, window_name: str = 'segmented items'):
    logger = logging.getLogger("main")

    def mouse_cb(event, x, y, flags, param):
        nonlocal logger
        logger_l = logger.getChild("mouse_cb")
        if event == cv2.EVENT_LBUTTONDOWN:
            for item in segmented_items:
                if item.bounding_box[0] <= x <= item.bounding_box[2] and item.bounding_box[1] <= y <= item.bounding_box[3]:
                    image_of_selected_bbox = frame[item.bounding_box[1]:item.bounding_box[3], item.bounding_box[0]:item.bounding_box[2]]
                    image_presenter.selected_bbox = {
                        'item': item,
                        'image': image_of_selected_bbox
                    }
                    logger_l.info(f"selected bbox {item.__dict__}")
                    break
            else:
                image_presenter.selected_bbox = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_cb)
    # cat2color = {}
    while image_presenter.is_running:
        if image_presenter.last_detection_result is None:
            time.sleep(0.1)
            continue
        frame = image_presenter.last_detection_result['frame']
        segmented_items = image_presenter.last_detection_result['detection_result']
        clipseg_mask = image_presenter.last_detection_result['clipseg_mask']
        boxes = [item.bounding_box for item in segmented_items]
        phrases = [item.label for item in segmented_items]
        detection_confidences = [item.detection_confidence for item in segmented_items]
        drawing = draw_boxes(frame.copy(), boxes, phrases, detection_confidences)
        # for masks, label in zip([item.masks for item in segmented_items], phrases):
        #     for mask in masks:
        #         mask_array = mask.mask
        #         if label not in cat2color:
        #             cat2color[label] = np.random.uniform(0, 255, size=3).astype(np.uint8)
        #         frame = draw_box_mask(frame, mask_array, color=cat2color[label])
        if image_presenter.selected_bbox is not None:
            cv2.imshow("selected bbox", image_presenter.selected_bbox['image'])
        if clipseg_mask is not None:
            clipseg_mask = clipseg_mask.copy()
            # clipseg_mask is (H, W, 1), make it (H, W, 3)
            clipseg_mask = np.concatenate([clipseg_mask.copy() for _ in range(3)], axis=-1)
            clipseg_mask = clipseg_mask.astype(np.uint8)
            cv2.addWeighted(drawing, 0.1, clipseg_mask, 1, 0, drawing)
            # print(f"clipseg_mask shape: {clipseg_mask.shape}")
            # cv2.imshow("clipseg", clipseg_mask)
        cv2.imshow(window_name, drawing)
        pressed_key = cv2.waitKey(1)
        if pressed_key & 0xFF == ord('q'):
            break
        elif pressed_key & 0xFF == ord('n'):
            image_presenter.selected_bbox = None
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
                                    prompt='monitor . cup . human',
                                    box_threshold=0.05,
                                    text_threshold=0.3, 
                                    confidence_threshold=0.2)
    clipseg_client = CLIPSegClient(grpc_channel=grpc_channel)
    try:
        detection_processor = ImagePresenter(frames_provider, detector_iterator, clipseg_client)
        main(detection_processor)
    except KeyboardInterrupt:
        detection_processor.stop()
        frames_provider.release()
        cv2.destroyAllWindows()
        