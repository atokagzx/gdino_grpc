from modules.pb2_init import *

import logging
import grpc
import numpy as np
from typing import Generator, Tuple, Union
from modules.items_dataclasses import DetectedItem, SegmentedItem, Mask


class CLIPSegClient:
    def __init__(self, grpc_channel: grpc.Channel = None):
        '''
        Provides segmentation service to the server using grpc.
        @param grpc_channel: the grpc channel to use
        '''
        self._logger = logging.getLogger("clipseg_client")
        self._channel = grpc_channel
        self._stub = pb2_grpc.CLIPSegmentationStub(self._channel)
        self._request_id = None
        
    def _generate_request(self, frame: np.ndarray, vis_prompt: np.ndarray) -> pb2.CLIPSegRequest:
        '''
        Generates a request to the server for segmentation.
        @param frame: the frame to generate the request for
        @param detection_result: the detection result to generate the request for
        @return: the request to send to the server as protobuf
        '''
        self._logger.debug("generating segmentation request, got detection result")
        frame_as_byte = pb2.NumpyArray(shape=frame.shape, data=frame.tobytes())
        vis_prompt_as_byte = pb2.NumpyArray(shape=vis_prompt.shape, data=vis_prompt.tobytes())
        request = pb2.CLIPSegRequest(image=frame_as_byte,
                        visual_prompt_image=vis_prompt_as_byte)

        return request
    
    def _process_response(self, response: pb2.CLIPSegResult) -> list:
        mask = np.frombuffer(response.mask.data, dtype=np.uint8).reshape(response.mask.shape)
        print(f"mask shape: {mask.shape}")
        return mask

    
    def process(self, frame: np.ndarray, vis_prompt: np.ndarray) -> Tuple[np.ndarray, list]:
        request = self._generate_request(frame, vis_prompt)
        self._logger.debug("starting segmentation")
        response = self._stub.segment(request)
        self._logger.debug("got segmentation response")
        mask = self._process_response(response)
        self._logger.debug("processed segmentation response")
        return frame, mask

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