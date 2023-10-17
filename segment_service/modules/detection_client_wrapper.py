from modules.pb2_init import *

import logging
import grpc
import numpy as np
import queue
from typing import Generator, Tuple, Union
from modules.items_dataclasses import DetectedItem, SegmentedItem, Mask


class DetectionClient(object):
    def __init__(self, 
                first_frame: np.ndarray,
                prompt, box_threshold, text_threshold, confidence_threshold,
                grpc_channel: grpc.Channel = None):
        '''
        @param video_iterator: the video iterator to use
        @param prompt: the prompt to use for grounding
        @param box_threshold: the threshold for bounding box confidence
        @param text_threshold: the threshold for text confidence
        '''
        self._logger = logging.getLogger("detection_client")

        self._prompt = prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._confidence_threshold = confidence_threshold
        self._channel = grpc_channel
        self._stub = pb2_grpc.DetectionStub(self._channel)
        self._frames_queue = queue.Queue()
        self._frames_queue.put(first_frame)

    def add_new_frame(self, frame: np.ndarray):
        '''
        Adds a new frame to the queue to be sent to the detection server.
        @param frame: the frame to add
        '''
        self._frames_queue.put(frame)

    def _generate_request(self, frame: np.ndarray) -> pb2.DetectionRequest:
        '''
        Generates a request for the server as protobuf items.
        @param frame: the frame to generate the request for
        @return: protobuf encoded request
        '''
        self._logger.debug("generating detection request, got frame")
        frame_as_byte = pb2.NumpyArray(shape=frame.shape, data=frame.tobytes())
        request = pb2.DetectionRequest(image=frame_as_byte,
                        prompt=self._prompt,
                        box_threshold=self._box_threshold,
                        text_threshold=self._text_threshold)
        self._logger.debug("yielding request to grpc iterator")
        return request

    def _filter_out_low_confidence(self, response: pb2.DetectionResult) -> pb2.DetectionResult:
        '''
        Filters out low confidence items from the response.
        @param response: the response from the server
        @return: the response with low confidence items filtered out
        '''
        items = [item for item in response.items if item.confidence > self._confidence_threshold]
        return pb2.DetectionResult(items=items)
    
    def __iter__(self) -> Generator[Tuple[np.ndarray, pb2.DetectionResult], None, None]:
        '''
        @return: a generator that yields items encoded as protobuf
        '''
        return self
    
    def __next__(self) -> Tuple[np.ndarray, pb2.DetectionResult]:
        '''
        Gets the next frame from the queue and sends it to the server for detection.
        @return: the frame and the detection result
        '''
        frame = self._frames_queue.get(block=True)
        self._logger.debug("got frame from queue")
        request = self._generate_request(frame)
        response = self._stub.detect(request)
        self._logger.debug("got detection response")
        filtered_response = self._filter_out_low_confidence(response)
        return frame, filtered_response
    
    @property
    def box_threshold(self) -> float:
        return self._box_threshold
    
    @box_threshold.setter
    def box_threshold(self, value: float):
        self._box_threshold = value

    @property
    def text_threshold(self) -> float:
        return self._text_threshold
    
    @text_threshold.setter
    def text_threshold(self, value: float):
        self._text_threshold = value

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold
    
    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        self._confidence_threshold = value

    @property
    def prompt(self) -> str:
        return self._prompt
    
    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value