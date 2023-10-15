#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pb2_include'))


import grpc
from concurrent import futures
import time
import segmentation_pb2_grpc as pb2_grpc
import segmentation_pb2 as pb2
import cv2
import numpy as np
import threading
import logging
from typing import List, Tuple

from segment_anything import build_sam, SamPredictor 
import subprocess
import os

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict 
from groundingdino.util.inference import predict as gd_predict
from huggingface_hub import hf_hub_download
from PIL import Image
import torch

class GDINOAdapter:
    def __init__(self,
                gd_ckpt_repo_id = "ShilongLiu/GroundingDINO",
                gd_ckpt_filenmae = "groundingdino_swinb_cogcoor.pth",
                gd_ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"):

        self._logger = logging.getLogger("gdino_adapter")
        self._lock = threading.Lock()
        self._model, self._device = self._load_model(gd_ckpt_repo_id, 
                                                     gd_ckpt_filenmae, 
                                                     gd_ckpt_config_filename)

    def _load_model(self, repo_id, filename, ckpt_config_filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger.info(f'using device: "{device}"')
        
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file) 
        args.device = device
        model = build_model(args)
        
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        _log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        _ = model.eval()
        model.to(device)
        self._logger.info("model loaded successfully")
        return model, device
    
    def find_categories(self, image, prompt, box_threshold, text_threshold) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        with self._lock:
            boxes, phrases, logits = self._gdino_detect(image, prompt, box_threshold, text_threshold)
            H, W = image.shape[:2]
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]).numpy()
            boxes = np.int0(boxes_xyxy)
            return list(zip(boxes, phrases, logits))

    def _gdino_detect(self, image, prompt, box_threshold=0.05, text_threshold=0.05):
        with torch.no_grad():
            transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            )
            image_source = Image.fromarray(image).convert("RGB")
            image = np.asarray(image_source)
            image_transformed, _ = transform(image_source, None)
            boxes, logits, phrases = gd_predict(
                model=self._model,
                image=image_transformed, 
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            return boxes, phrases, logits

class SAMAdapter:
    def __init__(self, sam_checkpoint_filename = "sam_vit_h_4b8939.pth"):
        self._logger = logging.getLogger("sam_adapter")
        self._lock = threading.Lock()
        self._model, self._device = self._load_model(sam_checkpoint_filename)
        self._predictor = SamPredictor(self._model)

    def _download_model(self, sam_checkpoint_filename):
        user_cache_dir = os.path.expanduser("~/.cache")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        self._logger.info(f'downloading from {url}, it may take a while')
        result = subprocess.run(['wget', url, '-P', user_cache_dir], capture_output=True, text=True, check=True)
        if result.returncode != 0:
            raise RuntimeError("failed to download sam model")
        self._logger.info("model downloaded successfully")

    def _load_model(self, sam_checkpoint_filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_cache_dir = os.path.expanduser("~/.cache")
        if os.path.exists(os.path.join(user_cache_dir, sam_checkpoint_filename)):
            self._logger.info(f'trying to load model from {os.path.join(user_cache_dir, sam_checkpoint_filename)}')
            try:
                model = build_sam(os.path.join(user_cache_dir, sam_checkpoint_filename)).to(device)
            except RuntimeError as e:
                self._logger.exception(f'failed to load model from {os.path.join(user_cache_dir, sam_checkpoint_filename)}, removing the corrupted file and retrying')
                os.remove(os.path.join(user_cache_dir, sam_checkpoint_filename))
                self._download_model(sam_checkpoint_filename)
                model = build_sam(os.path.join(user_cache_dir, sam_checkpoint_filename)).to(device)
        else:
            self._download_model(sam_checkpoint_filename)
            model = build_sam(os.path.join(user_cache_dir, sam_checkpoint_filename)).to(device)
        self._logger.info("model loaded successfully")
        return model, device
    
    def segment(self, image, bboxes: List[Tuple[int, int, int, int]],
                phrases: List[str]) -> List[List[np.ndarray]]:
        '''
        Segment the image based on the bounding boxes and phrases
        @param image: image to be segmented
        @param bboxes: bounding boxes
        @param phrases: phrases
        @return: segmentation masks: list of list of numpy arrays (each item has a list of masks))
        '''
        with self._lock:
            segmented_items = []
            self._predictor.set_image(image)
            for bbox, phrase in zip(bboxes, phrases):
                masks, confidence, _low_res_masks = self._predictor.predict(box=bbox, point_labels=[phrase])
                # apply rle to masks
                masks = [self.mask_to_rle(mask) for mask in masks]
                masks_with_confidence = [{"mask": mask, "confidence": confidence} for mask, confidence in zip(masks, confidence)]
                segmented_items.append(masks_with_confidence)
            return segmented_items
        
    @staticmethod
    def mask_to_rle(mask):
        '''
        Apply Run Length Encoding to mask
        @param mask: mask
        @return: run length encoding of mask
        '''
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return {"rle": runs, "shape": mask.shape}


class Detection(pb2_grpc.DetectionServicer):
    def __init__(self, detector):
        self._logger = logging.getLogger("detection_servicer")
        self._detector = detector

    def detect(self, request_iterator, context):
        for request in request_iterator:
            self._logger.debug("received request")
            response = self._process_request(request)
            self._logger.debug("yielding response")
            yield response

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
        items = [pb2.DetectedItem(bounding_box=pb2.BoundingBox(x1=box[0], x2=box[2], y1=box[1], y2=box[3]), 
                                            label=label, 
                                            confidence=confidence) for box, label, confidence in detected_items_list]
        return pb2.DetectionResult(items=items)
        
class Segmentation(pb2_grpc.SegmentationServicer):
    def __init__(self, segmentor):
        self._logger = logging.getLogger("segmentation_servicer")
        self._segmentor = segmentor

    def segment(self, request_iterator, context):
        for request in request_iterator:
            self._logger.debug("received request")
            response = self._process_request(request)
            self._logger.debug("yielding response")
            yield response

    def _process_request(self, request: pb2.SegmentationRequest) -> pb2.SegmentationResult:
        '''
        Process request from client, return segmentation result
        @param request: request from client: image, bounding_boxes, phrases
        @return: segmentation result: segmentation_masks
        '''
        image = np.frombuffer(request.image.data, dtype=np.uint8).reshape(request.image.shape)
        bboxes = [np.array([box.x1, box.y1, box.x2, box.y2]) for box in request.bounding_boxes]
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
    
                
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    detector = GDINOAdapter()
    segmentor = SAMAdapter()
    detection_service = Detection(detector)
    segmentation_service = Segmentation(segmentor)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_DetectionServicer_to_server(detection_service, server)
    pb2_grpc.add_SegmentationServicer_to_server(segmentation_service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()