import cv2
import numpy as np

from PIL import Image

from diffusers import StableDiffusionInpaintPipeline
from ultralytics import YOLO
from typing import List

class FaceDetailer:
    
    def __init__(self, 
                 device, 
                 yolo_url,
                 tokenizer,
                 text_encoder,
                 vae,
                 unet,
                 scheduler,
                 safety_checker,
                 feature_extractor 
                 ):
        self.device = device

        self.yolo_face = YOLO(yolo_url)
        self.inpaint_model = StableDiffusionInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor
        )
        self.inpaint_model.safety_checker = None
        self.yolo_face.to(self.device)
        self.inpaint_model.to(self.device)

    def generate_retouch(self,
                         image : List,
                         pos_prompt : str ="cute",   
                         neg_prompt : str = "ugly, deformed eyes, bad eyes, scary eyes",
                         guidance_scale=7.5,
                         strength=0.25,
                         num_inference_steps=30,
                         num_images_per_prompt=1,
                         generator=None):
        
        width, height = image[0].size
        batch_size = len(image)

        mask_img = []
        for img in image:
            mask = self.generate_mask(img, height, width)
            if np.all(mask == 0):
                mask = np.ones((height, width), dtype=np.uint8) * 255
            mask_img.append(mask)

        retouch_img = self.inpaint_model(prompt=batch_size * [pos_prompt],
                                         negative_prompt=batch_size * [neg_prompt],
                                         height=height,
                                         width=width,
                                         image=image,
                                         mask_image=mask_img,
                                         guidance_scale=guidance_scale,
                                         strength=strength,
                                         num_inference_steps=num_inference_steps,
                                         num_images_per_prompt=num_images_per_prompt,
                                         generator=generator
                                         ).images
        
        return retouch_img
        
    def generate_mask(self, image, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        results = self.yolo_face.predict(image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=cv2.FILLED)

        return Image.fromarray(mask)