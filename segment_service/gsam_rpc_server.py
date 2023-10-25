#!/usr/bin/env python3

from modules.pb2_init import *
import grpc
from concurrent import futures
import cv2
import numpy as np
import logging
from typing import List, Tuple
import argparse
from modules.adapters import GDINOAdapter, SAMAdapter, CLIPSegAdapter

class Detection(pb2_grpc.DetectionServicer):
    def __init__(self, detector):
        self._logger = logging.getLogger("detection_servicer")
        self._detector = detector

    def detect(self, request, context):
        self._logger.debug("received request")
        response = self._process_request(request)
        self._logger.debug("yielding response")
        return response

    def _process_request(self, request: pb2.DetectionRequest) -> pb2.DetectionResult:
        '''
        Process request from client, return detection result
        @param request: request from client: image, prompt, box_threshold, text_threshold
        @return: detection result: bounding_boxes
        '''
        image = np.frombuffer(request.image.data, dtype=np.uint8).reshape(request.image.shape)
        prompt = request.prompt
        box_threshold = request.box_threshold
        text_threshold = request.text_threshold
        detected_items_list = self._detector.find_categories(image, prompt, box_threshold, text_threshold)
        items = [pb2.DetectedItem(bounding_box=pb2.BoundingBox(**dict(zip(["x1", "y1", "x2", "y2"], box))),
                                            label=label, 
                                            confidence=confidence) for box, label, confidence in detected_items_list]
        return pb2.DetectionResult(items=items)
        
class Segmentation(pb2_grpc.SegmentationServicer):
    def __init__(self, segmentor):
        self._logger = logging.getLogger("segmentation_servicer")
        self._segmentor = segmentor

    def segment(self, request, context):
        self._logger.debug("received request")
        response = self._process_request(request)
        self._logger.debug("yielding response")
        return response

    def _process_request(self, request: pb2.SegmentationRequest) -> pb2.SegmentationResult:
        '''
        Process request from client, return segmentation result
        @param request: request from client: image, bounding_boxes, phrases
        @return: segmentation result: segmentation_masks
        '''
        image = np.frombuffer(request.image.data, dtype=np.uint8).reshape(request.image.shape)
        bboxes = [np.array([getattr(box, attr) for attr in ["x1", "y1", "x2", "y2"]]) for box in request.bounding_boxes]
        phrases = request.phrases
        segmented_items = self._segmentor.segment(image, bboxes, phrases)
        items = [pb2.SegmentedItem(masks=self._mask_to_proto(segmented_item)) for segmented_item in segmented_items]
        return pb2.SegmentationResult(items=items)

    def _mask_to_proto(self, item: List[dict]) -> List[pb2.SegmentationMask]:
        '''
        Convert a list of segmentation masks to a list of protobuf SegmentationMask
        @param item: a list of segmentation masks for an object
        @return: a list of protobuf SegmentationMask
        '''
        return [pb2.SegmentationMask(mask=pb2.NumpyArray(shape=item_mask["mask"]['shape'], data=item_mask["mask"]['rle'].tobytes()),
                                     confidence=item_mask["confidence"]) for item_mask in item]

class CLIPSegmentation(pb2_grpc.CLIPSegmentationServicer):
    def __init__(self):
        self._logger = logging.getLogger("clip_segmentation_servicer")
        self._segmentor = CLIPSegAdapter()

    def segment(self, request, context):
        self._logger.debug("received request")
        response = self._process_request(request)
        # print("shape:", len(response), response[0].shape)
        print(response[1].shape)
        mask = response[0]
        mask = mask.cpu().numpy()
        mask *= 255
        mask = mask.astype(np.uint8)
        # print(f"shape: {mask.shape}")
        # shape is (1, 720, 1280), make it (720, 1280, 1)
        mask = np.squeeze(mask, axis=0)
        mask = np.expand_dims(mask, axis=-1)
        # print(f"shape: {mask.shape}")
        
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
        self._logger.debug("yielding response")



        # sem_mask = response[1]
        # sem_mask = sem_mask.cpu().numpy()
        # sem_mask *= 20
        # sem_mask = sem_mask.astype(np.uint8)
        # cv2.imshow("sem_mask", sem_mask)
        # cv2.waitKey(1)


        return pb2.CLIPSegResult(mask=pb2.NumpyArray(shape=mask.shape, data=mask.tobytes()))
        # return pb2.CLIPSegResult(mask=pb2.NumpyArray(shape=mask.shape, data=mask.tobytes()))
                                #  sem_mask=pb2.NumpyArray(shape=sem_mask.shape, data=sem_mask.tobytes()))
    
    def _process_request(self, request: pb2.CLIPSegRequest) -> pb2.CLIPSegResult:
        image = np.frombuffer(request.image.data, dtype=np.uint8).reshape(request.image.shape)
        visual_prompt_image = np.frombuffer(request.visual_prompt_image.data, dtype=np.uint8).reshape(request.visual_prompt_image.shape)
        print(f"image shape: {image.shape}")
        print(f"visual_prompt_image shape: {visual_prompt_image.shape}")
        # phrase = request.phrase
        return self._segmentor.segment(image, visual_prompt_image, 0.01)


                
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("detection_servicer").setLevel(logging.DEBUG)
    # logging.getLogger("segmentation_servicer").setLevel(logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", type=str, default="[::]:50051")
    args = parser.parse_args()
    detector = GDINOAdapter()
    segmentor = SAMAdapter()
    detection_service = Detection(detector)
    segmentation_service = Segmentation(segmentor)
    clip_segmentation_service = CLIPSegmentation()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_DetectionServicer_to_server(detection_service, server)
    pb2_grpc.add_SegmentationServicer_to_server(segmentation_service, server)
    pb2_grpc.add_CLIPSegmentationServicer_to_server(clip_segmentation_service, server)
    logging.info(f"starting gRPC server on {args.address}")
    server.add_insecure_port(args.address)
    server.start()
    server.wait_for_termination()