#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pb2_include'))

import time
import logging
import grpc
import segmentation_pb2_grpc as pb2_grpc
import segmentation_pb2 as pb2
import cv2
import numpy as np
import threading
from typing import Generator, Tuple
from dataclasses import dataclass

class VideoCaptureBuffless(cv2.VideoCapture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("video_capture_buffless")
        self.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        super().grab()
        return super().retrieve()
    
    def __next__(self):
        self._logger.debug("getting next frame")
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame
    
    def __iter__(self):
        return self
    
    def __del__(self):
        self.release()

class DetectionClient(object):
    def __init__(self, 
                grpc_channel: grpc.Channel,
                video_iterator: Generator[np.ndarray, None, None],
                prompt, box_threshold, text_threshold, confidence_threshold):
        '''
        @param video_iterator: the video iterator to use
        @param prompt: the prompt to use for grounding
        @param box_threshold: the threshold for bounding box confidence
        @param text_threshold: the threshold for text confidence
        '''
        self._logger = logging.getLogger("detection_client")
        self._lock = threading.Lock()
        self._video_iter = video_iterator
        self._prompt = prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._confidence_threshold = confidence_threshold
        self._channel = grpc_channel
        self._stub = pb2_grpc.DetectionStub(self._channel)

    @property
    def last_frame(self) -> np.ndarray:
        '''
        Returns the last frame that was processed. 
        This used becaused it's hard to get the last frame from the video generator since it's a generator.
        @return: the last frame that was processed
        '''
        return self._last_frame
    
    def _generate_request(self) -> Generator[pb2.DetectionRequest, None, None]:
        '''
        Generates a request for the server as protobuf items.
        @return: a generator that yields protobuf items
        '''
        for frame in self._video_iter:
            self._logger.debug("generating detection request, got frame")
            self._last_frame = frame
            frame_as_byte = pb2.NumpyArray(shape=frame.shape, data=frame.tobytes())
            request = pb2.DetectionRequest(image=frame_as_byte,
                            prompt=self._prompt,
                            box_threshold=self._box_threshold,
                            text_threshold=self._text_threshold)
            self._logger.debug("yielding request to grpc iterator")
            yield request

    def __iter__(self):
        return self._generator()
    
    def __next__(self):
        return next(self._generator())

    def _generator(self) -> Generator[Tuple[np.ndarray, pb2.DetectionResult], None, None]:
        '''
        Generates a response from the server as protobuf items.
        @return: a generator that yields frame got by the video generator and the response from the server
        '''
        if self._lock.locked():
            raise RuntimeError('This class does not support multiple generators at the same time')
        with self._lock:
            response_generator = self._stub.detect(self._generate_request())
            self._logger.debug("starting detection")
            for response in response_generator:
                self._logger.debug("got detection response")
                filtered_response = self._filter_out_low_confidence(response)
                self._logger.debug("yielding response to client")
                yield self._last_frame.copy(), filtered_response

    def _filter_out_low_confidence(self, response: pb2.DetectionResult) -> pb2.DetectionResult:
        '''
        Filters out low confidence items from the response.
        @param response: the response from the server
        @return: the response with low confidence items filtered out
        '''
        items = [item for item in response.items if item.confidence > self._confidence_threshold]
        return pb2.DetectionResult(items=items)
    
    @property
    def box_threshold(self):
        return self._box_threshold
    
    @box_threshold.setter
    def box_threshold(self, value):
        self._box_threshold = value

    @property
    def text_threshold(self):
        return self._text_threshold
    
    @text_threshold.setter
    def text_threshold(self, value):
        self._text_threshold = value

    @property
    def confidence_threshold(self):
        return self._confidence_threshold
    
    @confidence_threshold.setter
    def confidence_threshold(self, value):
        self._confidence_threshold = value

    @property
    def prompt(self):
        return self._prompt
    
    @prompt.setter
    def prompt(self, value):
        self._prompt = value

    @property
    def video_iterator(self):
        return self._video_iter
    
    @video_iterator.setter
    def video_iterator(self, value):
        self._video_iter = value

class SegmentationClient:
    def __init__(self, grpc_channel: grpc.Channel, detection_iterator: Generator[Tuple[np.ndarray, pb2.DetectionResult], None, None]):
        '''
        @param grpc_channel: the grpc channel to use
        @param detection_client: the detection client to use
        '''
        self._logger = logging.getLogger("segmentation_client")
        self._lock = threading.Lock()
        self._detection_iter = detection_iterator
        self._channel = grpc_channel
        self._stub = pb2_grpc.SegmentationStub(self._channel)
        
    def _generate_request(self) -> Generator[pb2.SegmentationRequest, None, None]:
        '''
        Generates a request for the server as protobuf items.
        @return: a generator that yields protobuf items
        '''
        for frame, detection_result in self._detection_iter:
            self._logger.debug("generating segmentation request, got detection result")
            labels = [item.label for item in detection_result.items]
            bboxes = [item.bounding_box for item in detection_result.items]
            self._last_detection_result = {
                'frame': frame.copy(),
                'bboxes': [[getattr(item.bounding_box, field) for field in ('x1', 'y1', 'x2', 'y2')] for item in detection_result.items],
                'labels': labels,
                'confidences': [item.confidence for item in detection_result.items],
            }
            frame_as_byte = pb2.NumpyArray(shape=frame.shape, data=frame.tobytes())
            request = pb2.SegmentationRequest(image=frame_as_byte,
                            bounding_boxes=bboxes,
                            phrases=labels)
            yield request
    
    @staticmethod
    def _unencode_mask(mask: pb2.NumpyArray) -> np.ndarray:
        '''
        Converts RLE encoded mask to np.array mask
        @param mask: the protobuf mask encoded by RLE
        @return: the numpy array mask
        '''
        rle_encoded = np.frombuffer(mask.data, dtype=np.int64)
        image_shape = mask.shape
        mask = np.zeros((image_shape[0] * image_shape[1],), dtype=np.uint8)
        starts = rle_encoded[0::2]
        lengths = rle_encoded[1::2]
        [mask.__setitem__(slice(start, start + length), 1) for start, length in zip(starts, lengths)]
        mask = mask.reshape(image_shape)
        return mask
    
    def _process_response(self, response: pb2.SegmentationResult) -> dict:
        segmented_items = []
        for item, bbox, label, detection_confidence in zip(response.items, 
                                                        self._last_detection_result['bboxes'], 
                                                        self._last_detection_result['labels'], 
                                                        self._last_detection_result['confidences']):
            masks = [self._unencode_mask(mask.mask) for mask in item.masks]
            confidences = [mask.confidence for mask in item.masks]
            segmented_items.append(SegmentedItem(
                bounding_box = bbox,
                label=label,
                detection_confidence=detection_confidence,
                masks=tuple(Mask(mask, confidence) for mask, confidence in zip(masks, confidences))))
        return segmented_items

    
    def _generator(self) -> Generator[Tuple[np.ndarray, dict], None, None]:
        if self._lock.locked():
            raise RuntimeError('This class does not support multiple generators at the same time')
        with self._lock:
            response_generator = self._stub.segment(self._generate_request())
            self._logger.debug("starting segmentation")
            for response in response_generator:
                self._logger.debug("got segmentation response")
                segmented_items = self._process_response(response)
                self._logger.debug("processed segmentation response")
                frame = self._last_detection_result['frame']
                yield frame, segmented_items
    
    @property
    def last_detection_result(self) -> dict:
        '''
        Returns the last detection result that was processed.
        @return: the last detection result that was processed:
            {
                'frame': the frame that was processed
                'bboxes': list of bounding boxes
                'labels': list of labels
                'confidences': list of confidences
            }
        '''
        return self._last_detection_result

    def __iter__(self):
        return self._generator()
    
    def __next__(self):
        return next(self._generator())

def draw_boxes(image, boxes, phrases, logits):
    for box, phrase, logit in zip(boxes, phrases, logits):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(image, phrase, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(image, f"score: {logit:.2f}", (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image

def draw_box_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = None):
        # mask from (H, W) to (H, W, 1)
        mask = np.expand_dims(mask, axis=-1)
        if color is None:
            color = np.random.uniform(0, 1, size=3)
        rgb_mask = np.array([mask.copy() for _ in range(3)])
        colored_mask = np.concatenate(rgb_mask, axis=-1) * color
        colored_mask = np.array(colored_mask, dtype=np.uint8)
        image = cv2.addWeighted(image, 1, colored_mask, 1, 0)
        return image

@dataclass
class Mask:
    mask: np.ndarray
    confidence: float  

@dataclass
class DetectedItem:
    bounding_box: Tuple[int, int, int, int]
    label: str
    detection_confidence: float
 
@dataclass
class SegmentedItem(DetectedItem):
    masks: Tuple[Mask]

class LoopSegmentation:
    def __init__(self, segmentation_client: SegmentationClient):
        self._segmentation_client = segmentation_client
        self._last_result = None
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        for frame, segmented_items in self._segmentation_client:
            self._last_result = {
                'frame': frame,
                'segmented_items': segmented_items
            }
            if self._stop: break

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

def main(segmentation_loop: LoopSegmentation):
    logger = logging.getLogger("main")
    while segmentation_loop.is_running:
        time.sleep(0.1)
        if segmentation_loop.last_result is None:
            continue
        frame = segmentation_loop.last_result['frame']
        segmented_items = segmentation_loop.last_result['segmented_items']
        boxes = [item.bounding_box for item in segmented_items]
        phrases = [item.label for item in segmented_items]
        detection_confidences = [item.detection_confidence for item in segmented_items]
        frame = draw_boxes(frame, boxes, phrases, detection_confidences)
        for masks in [item.masks for item in segmented_items]:
            for mask in masks:
                mask_array = mask.mask
                frame = draw_box_mask(frame, mask_array, color=(0, 255, 0))
        cv2.imshow('segmented items', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        logger.info("segmentation loop stopped")

if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.DEBUG)    
        grpc_channel = grpc.insecure_channel('localhost:50051')
        video_generator = VideoCaptureBuffless(0)
        video_generator.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_generator.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        detector_client = DetectionClient(grpc_channel=grpc_channel,
                                        video_iterator=video_generator,
                                        prompt='monitor . cup . human',
                                        box_threshold=0.05,
                                        text_threshold=0.05, 
                                        confidence_threshold=0.2)
        segmentor_client = SegmentationClient(grpc_channel=grpc_channel,
                                            detection_iterator=detector_client)
        
        segmentation_loop = LoopSegmentation(segmentor_client)
        main(segmentation_loop)
    except KeyboardInterrupt:
        segmentation_loop.stop()
        video_generator.release()
        cv2.destroyAllWindows()
        