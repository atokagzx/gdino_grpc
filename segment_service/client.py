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
        self.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        super().grab()
        return super().retrieve()
    
    def __next__(self):
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
                 video_generator: Generator[np.ndarray, None, None],
                 prompt, box_threshold, text_threshold, confidence_threshold):
        '''
        @param video_generator: a generator that yields frames
        @param prompt: the prompt to use for grounding
        @param box_threshold: the threshold for bounding box confidence
        @param text_threshold: the threshold for text confidence
        '''
        self._video_generator = video_generator
        self._prompt = prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._confidence_threshold = confidence_threshold
        self._channel = grpc.insecure_channel('localhost:50051')
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
        for frame in self._video_generator:
            self._last_frame = frame
            frame_as_byte = pb2.NumpyArray(shape=frame.shape, data=frame.tobytes())
            request = pb2.DetectionRequest(image=frame_as_byte,
                            prompt=self._prompt,
                            box_threshold=self._box_threshold,
                            text_threshold=self._text_threshold)
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
        response_generator = self._stub.detect(self._generate_request())
        for response in response_generator:
            filtered_response = self._filter_out_low_confidence(response)
            yield self._last_frame, filtered_response

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
    def video_generator(self):
        return self._video_generator
    
    @video_generator.setter
    def video_generator(self, value):
        self._video_generator = value


class SegmentationClient:
    def __init__(self, detection_client: DetectionClient):
        '''
        @param detection_client: the detection client to use
        '''
        self._logger = logging.getLogger("segmentation_client")
        self._detection_client = detection_client
        self._run_thread()

    def _run_thread(self):
        self._last_result = None
        self._channel = grpc.insecure_channel('localhost:50051')
        self._stub = pb2_grpc.SegmentationStub(self._channel)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _generate_request(self) -> Generator[pb2.SegmentationRequest, None, None]:
        '''
        Generates a request for the server as protobuf items.
        @return: a generator that yields protobuf items
        '''
        for frame, detection_result in self._detection_client:
            print('segmenting')
            labels = [item.label for item in detection_result.items]
            bboxes = [item.bounding_box for item in detection_result.items]
            self._last_detection_result = {
                'frame': frame,
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
    
    def _loop(self):
        response_generator = self._stub.segment(self._generate_request())
        for response in response_generator:
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
                    masks=tuple(Mask(mask, confidence) for mask, confidence in zip(masks, confidences))
                ))
            self._last_result = {
                'frame': self._last_detection_result['frame'],
                'items': segmented_items
            }
            if self._stop: break

            
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
            
    @property
    def last_result(self) -> dict:
        '''
        Returns the last result that was processed.
        @return: the last result that was processed:
            {
                'frame': the frame that was processed
                'items': list of SegmentedItem objects
            }
        '''
        return self._last_result
    
    def stop(self):
        self._stop = True
        self._thread.join()
        self._channel.close()
    
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

def main():
    video_generator = VideoCaptureBuffless(0)
    video_generator.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_generator.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector_client = DetectionClient(video_generator,
                                      prompt='monitor . cup . human',
                                      box_threshold=0.05,
                                      text_threshold=0.05, 
                                      confidence_threshold=0.2)
    segmentor_client = SegmentationClient(detector_client)
    while True:
        result = segmentor_client.last_result
        if result is None:
            time.sleep(0.01)
            continue
        image = result['frame']
        boxes = [item.bounding_box for item in result['items']]
        phrases = [item.label for item in result['items']]
        detection_confidences = [item.detection_confidence for item in result['items']]
        image = draw_boxes(image, boxes, phrases, detection_confidences)
        for masks in [item.masks for item in result['items']]:
            for mask in masks:
                mask_array = mask.mask
                image = draw_box_mask(image, mask_array, color=(0, 255, 0))
        cv2.imshow('segmented items', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass