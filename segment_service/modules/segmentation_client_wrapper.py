from modules.pb2_init import *

import logging
import grpc
import numpy as np
from typing import Generator, Tuple, Union
from modules.items_dataclasses import DetectedItem, SegmentedItem, Mask


class SegmentationClient:
    def __init__(self, grpc_channel: grpc.Channel = None):
        '''
        Provides segmentation service to the server using grpc.
        @param grpc_channel: the grpc channel to use
        '''
        self._logger = logging.getLogger("segmentation_client")
        self._channel = grpc_channel
        self._stub = pb2_grpc.SegmentationStub(self._channel)
        self._request_id = None
        
    def _generate_request(self, frame: np.ndarray, detection_result: dict) -> pb2.SegmentationRequest:
        '''
        Generates a request to the server for segmentation.
        @param frame: the frame to generate the request for
        @param detection_result: the detection result to generate the request for
        @return: the request to send to the server as protobuf
        '''
        items = detection_result.items
        self._logger.debug("generating segmentation request, got detection result")
        labels = [item.label for item in items]
        bboxes = [item.bounding_box for item in items]
        frame_as_byte = pb2.NumpyArray(shape=frame.shape, data=frame.tobytes())
        request = pb2.SegmentationRequest(image=frame_as_byte,
                        bounding_boxes=bboxes,
                        phrases=labels)

        return request
    
    def _process_response(self, detection_result:pb2.DetectionResult, response: pb2.SegmentationResult) -> list:
        segmented_items = []
        for detect_item, segm_item in zip(detection_result.items, response.items):
            masks = [self._unencode_mask(mask.mask) for mask in segm_item.masks]
            confidences = [mask.confidence for mask in segm_item.masks]
            segmented_items.append(SegmentedItem(
                bounding_box = [getattr(detect_item.bounding_box, field) for field in ('x1', 'y1', 'x2', 'y2')],
                label = detect_item.label,
                detection_confidence = detect_item.confidence,
                masks=tuple(Mask(mask, confidence) for mask, confidence in zip(masks, confidences))))
        return segmented_items

    
    def process(self, frame: np.ndarray, detection_result: dict) -> Tuple[np.ndarray, list]:
        request = self._generate_request(frame, detection_result)
        self._logger.debug("starting segmentation")
        response = self._stub.segment(request)
        self._logger.debug("got segmentation response")
        segmented_items = self._process_response(detection_result, response)
        self._logger.debug("processed segmentation response")
        return frame, segmented_items

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