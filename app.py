import os
import io
import time
import torch
import numpy as np
import json
import gc
import requests
import onnxruntime as ort
import subprocess
from io import BytesIO
from PIL import Image

from RealESRGAN import RealESRGAN
from diffusers import DDIMScheduler, AutoencoderKL, UniPCMultistepScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPTokenizer, pipeline
from instantid.pipelines.pipeline_gen import StableDiffusionInstantPipelineImg2Img

from instantid.utils.load_components import load_face_app, prepare_face_embeddings
from controlnet_aux import OpenposeDetector
from instantid.face_detailer.detailer_pipeline import FaceDetailer
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi import File, Form, UploadFile
from fastapi.responses import StreamingResponse
import asyncio
import uuid

import boto3
from datetime import datetime, timedelta

app = FastAPI()

request_queue = asyncio.Queue()
flash_queue = []
response_event = {}  
responses = {} 

def get_active_branch():
    # "git branch" 명령어 실행하여 출력 문자열을 가져옵니다.
    output = subprocess.check_output(["git", "branch"], universal_newlines=True)
    # 각 줄을 리스트에 저장
    branches = [line.strip() for line in output.splitlines()]
    # 활성 브랜치는 '*'로 시작하는 항목입니다.
    active_branch = [branch for branch in branches if branch.startswith("*")]
    if active_branch:
        # '*'를 제거한 문자열을 반환합니다.
        return active_branch[0].lstrip("* ").strip()
    else:
        return None

branch = get_active_branch()
if branch == "AI_d_02":
    STYLE="ghibli"
    print("Ghibli Style")
else:
    STYLE="disney"
    print("Disney Style")

def get_instance_info(retry_interval=3):
    """
    Keeps retrying until instance_id and region are successfully fetched via IMDSv2.
    Returns (instance_id, region)
    """
    while True:
        try:
            # 1. IMDSv2 토큰 요청
            token_resp = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=5
            )
            if token_resp.status_code != 200:
                raise Exception(f"Metadata token request failed: {token_resp.status_code}")
            token = token_resp.text
            headers = {"X-aws-ec2-metadata-token": token}

            # 2. 인스턴스 ID 요청
            id_resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id", headers=headers, timeout=5)
            if id_resp.status_code != 200:
                raise Exception(f"Instance ID request failed: {id_resp.status_code}")
            instance_id = id_resp.text

            # 3. AZ 요청
            az_resp = requests.get("http://169.254.169.254/latest/meta-data/placement/availability-zone", headers=headers, timeout=5)
            if az_resp.status_code != 200:
                raise Exception(f"Availability zone request failed: {az_resp.status_code}")
            az = az_resp.text

            # 4. region 파싱
            region = az[:-1]
            
            print(f"[IMDS] Successfully fetched instance_id={instance_id}, region={region}")
            return instance_id, region

        except Exception as e:
            print(f"[IMDS] Error fetching instance info, retrying in {retry_interval} seconds: {e}")
            time.sleep(retry_interval)
    
INSTANCE_ID, REGION = get_instance_info()

session = boto3.Session(region_name=REGION)
cloudwatch = session.client('cloudwatch')
ec2_client = session.client('ec2')
asg_client = session.client('autoscaling')

def update_ec2_tag(instance_id, key, value):
    try:
        ec2_client.create_tags(
            Resources=[instance_id],
            Tags=[
                {
                    "Key" : key,
                    "Value" : value
                }
            ]
        )
        #print(f"Updated EC2 tag : {key} = {value}")
    except Exception as e:
        print(f"Error updating EC2 tag : {e}")

def decrease_desired_capacity(auto_scaling_group_name):
    try:
        # 현재 Auto Scaling Group 정보 가져오기
        response = asg_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[auto_scaling_group_name]
        )

        if not response['AutoScalingGroups']:
            print("Auto Scaling Group을 찾을 수 없습니다.")
            return

        # 현재 Desired Capacity 확인
        current_capacity = response['AutoScalingGroups'][0]['DesiredCapacity']
        print(f"현재 Desired Capacity: {current_capacity}")

        # 최소 용량 확인
        min_capacity = response['AutoScalingGroups'][0]['MinSize']

        # 1 줄인 새 Desired Capacity 설정 (최소 값 이하로 줄이지 않음)
        new_capacity = max(current_capacity - 1, min_capacity)

        if new_capacity == current_capacity:
            print("이미 최소 용량이므로 줄일 수 없습니다.")
            return

        # Desired Capacity 업데이트
        asg_client.set_desired_capacity(
            AutoScalingGroupName=auto_scaling_group_name,
            DesiredCapacity=new_capacity,
            HonorCooldown=False  # 쿨다운 적용 여부 (기본적으로 True)
        )

        print(f"새로운 Desired Capacity: {new_capacity}")

    except Exception as e:
        print(f"오류 발생: {e}")


def load_model(model_path, device='cpu'):
    model = torch.load(model_path, map_location=device)
    return model.eval()

# 병렬로 모델 로드
def load_all_models(paths, device='cpu'):
    models = {}
    with ThreadPoolExecutor() as executor:
        # 병렬 작업 수행
        future_to_key = {executor.submit(load_model, path, device): key for key, path in paths.items()}
        for future in future_to_key:
            key = future_to_key[future]
            try:
                models[key] = future.result()  # 모델 로드 결과 저장
                print(f"{key} 로드 완료")
            except Exception as e:
                print(f"{key} 로드 중 오류 발생: {e}")
    return models

def init_components(style="ghibli"):
    global_tic = time.time()
    print("Model Loading ... ")

    #model path
    customized_pipeline_path = '/home/ec2-user/PHOTOBOOTH_MODEL/sd_15'
    tokenizer_path = os.path.join(customized_pipeline_path, 'customized', 'tokenizer')

    image_enhancer = RealESRGAN("cuda", scale=4)
    image_enhancer.load_weights('/home/ec2-user/PHOTOBOOTH_MODEL/real_esrgan/RealESRGAN_x4.pth', download=False)  

    vae_path = os.path.join(customized_pipeline_path, 'customized', f'vae/vae_{style}.pth')
    unet_path = os.path.join(customized_pipeline_path, 'customized', f'unet/unet_{style}.pth')
    text_encoder_path = os.path.join(customized_pipeline_path, 'customized', 'text_encoder/text_encoder.pth')
    controlnet_face_path = os.path.join(customized_pipeline_path, 'customized', 'controlnet_face/controlnet_face.pth')
    controlnet_tile_path = os.path.join(customized_pipeline_path, 'controlnet_tile.pth')
    controlnet_depth_path = os.path.join(customized_pipeline_path, 'controlnet_depth.pth')

    paths = {
        "vae" : vae_path,
        "unet" : unet_path,
        "text_encoder" : text_encoder_path,
        "controlnet_face" : controlnet_face_path,
        "controlnet_tile" : controlnet_tile_path,
        "controlnet_depth" : controlnet_depth_path,
    }

    models = load_all_models(paths, device="cuda")
    #load_models
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    vae = models.get("vae")
    unet = models.get("unet")
    text_encoder = models.get("text_encoder")
    controlnet_face = models.get("controlnet_face")
    controlnet_tile = models.get("controlnet_tile")
    controlnet_depth = models.get("controlnet_depth")

    face_extractor = load_face_app()

    depth_estimator = pipeline('depth-estimation')
    controlnet = MultiControlNetModel([controlnet_face, controlnet_tile, controlnet_depth])
    
    scheduler = DDIMScheduler(**{
    'beta_start': 0.00085,
    'beta_end': 0.012,
    'beta_schedule': 'linear',
    'steps_offset': 1,
    'clip_sample': False,
    })

    face_detailer_vae = AutoencoderKL.from_single_file('/home/ec2-user/PHOTOBOOTH_MODEL/sd_15/vaeFtMse840000EmaPruned_vaeFtMse840k.safetensors', torch_dtype=torch.float16).to("cuda")
    face_detailer = FaceDetailer(device="cuda", 
                                 yolo_url="/home/ec2-user/PHOTOBOOTH_MODEL/yolo/yolov8n-face.pt",
                                 tokenizer=tokenizer,
                                 text_encoder=text_encoder,
                                 vae=face_detailer_vae,
                                 unet=unet,
                                 scheduler=scheduler,
                                 safety_checker=None,
                                 feature_extractor=None
                                 )

    pipe_ip = StableDiffusionInstantPipelineImg2Img(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        image_encoder=None,
        safety_checker=None,
        feature_extractor=None,
    )

    pipe_ip.load_resampler()
    pipe_ip.to("cuda")

    pipe_ip.load_textual_inversion("Eugeoter/badhandv4")
    pipe_ip.load_textual_inversion("/home/ec2-user/PHOTOBOOTH_MODEL/sd_15/EasyNegative.safetensors")
    #pipe_ip.scheduler = DPMSolverMultistepScheduler.from_config(pipe_ip.scheduler.config)
    pipe_ip.scheduler = UniPCMultistepScheduler.from_config(pipe_ip.scheduler.config)

    if style == "ghibli":    
        pipe_ip.load_lora_weights('/home/ec2-user/PHOTOBOOTH_MODEL/sd_15/ghibli_style_offset.safetensors')

    global_toc = time.time()
    print(f"Model Loaded : {global_toc - global_tic:.2f}")

    torch.cuda.empty_cache()
    gc.collect()
    
    return face_detailer, face_extractor, depth_estimator, pipe_ip, image_enhancer

def onnx_session_control(cuda=True):
    landmark_3d_68_path = '/home/ec2-user/PHOTOBOOTH_MODEL/models/antelopev2/antelopev2/1k3d68.onnx'
    landmark_2d_106_path = '/home/ec2-user/PHOTOBOOTH_MODEL/models/antelopev2/antelopev2/2d106det.onnx'
    genderage_path = '/home/ec2-user/PHOTOBOOTH_MODEL/models/antelopev2/antelopev2/genderage.onnx'
    recognition_path = '/home/ec2-user/PHOTOBOOTH_MODEL/models/antelopev2/antelopev2/glintr100.onnx'
    detection_path = '/home/ec2-user/PHOTOBOOTH_MODEL/models/antelopev2/antelopev2/scrfd_10g_bnkps.onnx' 

    if cuda:
        face_extractor.models['landmark_3d_68'].session = ort.InferenceSession(landmark_3d_68_path, providers=['CUDAExecutionProvider'])
        face_extractor.models['landmark_2d_106'].session = ort.InferenceSession(landmark_2d_106_path, providers=['CUDAExecutionProvider'])
        face_extractor.models['genderage'].session = ort.InferenceSession(genderage_path, providers=['CUDAExecutionProvider'])
        face_extractor.models['recognition'].session = ort.InferenceSession(recognition_path, providers=['CUDAExecutionProvider'])
        face_extractor.models['detection'].session = ort.InferenceSession(detection_path, providers=["CUDAExecutionProvider"])
    else:
        del face_extractor.models['landmark_3d_68'].session
        del face_extractor.models['landmark_2d_106'].session
        del face_extractor.models['genderage'].session
        del face_extractor.models['recognition'].session
        del face_extractor.models['detection'].session
    
        torch.cuda.empty_cache()
        gc.collect()

#face_detailer, face_extractor, openpose_detector, pipe_ip = init_components()
face_detailer, face_extractor, depth_estimator, pipe_ip, image_enhancer = init_components(style=STYLE)

LAST_REQUEST_TIME = datetime.now()
update_ec2_tag(INSTANCE_ID, "status", "on-demand")
onnx_session_control(cuda=False)

def prepare_prompt(style):
    prompt = "masterpiece, best quality, high resolution, cartoon, character, beautiful, adorable, cute, perfect face, soft smiling face, perfect eyes, natural eye proportions"
    n_prompt = "<badhandv4>, easynegative, lowres, low quality, worst quality, deformed face, glitch, deformed, mutated, cross-eyed, misalinged eyes, wide-spaced pupils, ugly, disfigured, extra limb"
    ghibli = "ghibli style, ghibli animation, ghibli background, san \(mononoke hime\), howl \(howl no ugoku shiro\)"
    disney = "Disney, Disney style, Disney character, Disney animation, Disney background"

    if style == "ghibli":
        prompt += ghibli
    if style == "disney":
        prompt += disney

    return prompt, n_prompt

def prepare_conditions(src_path, face_extractor, depth_estimator, image_enhancer):
    src = Image.open(src_path).convert("RGB").resize((544, 680))

    depth = depth_estimator(src)['depth']
    depth = np.array(depth)
    depth = depth[:, :, None]
    depth = np.concatenate([depth, depth, depth], axis=2)
    controlnet_image_depth = Image.fromarray(depth)

    sr_src = image_enhancer.predict(src).resize((src.size))
    face_embedding, controlnet_image_face = prepare_face_embeddings(sr_src, face_extractor)

    return face_embedding, sr_src, controlnet_image_face.resize((src.size)), controlnet_image_depth.resize((src.size)), src

async def make_batch_files(batch):

    input_images = []
    prompts = []
    negative_prompts = []
    controlnet_image_faces = []
    controlnet_image_depths = []

    prompt, n_prompt = prepare_prompt(style=STYLE)

    face_embeddings = []
    
    onnx_session_control(cuda=True)

    async def process_photo(photo):
        src = await asyncio.to_thread(photo.resize, (544, 680))

        depth_dict = await asyncio.to_thread(lambda : depth_estimator(src))
        depth = depth_dict['depth']
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)  
        controlnet_image_depth = Image.fromarray(depth)
        
        sr_src = await asyncio.to_thread(lambda: image_enhancer.predict(src).resize(src.size))

        face_embedding, controlnet_image_face = await asyncio.to_thread(
            lambda: prepare_face_embeddings(sr_src, face_extractor)
        )

        controlnet_image_face = controlnet_image_face.resize(src.size)
        controlnet_image_depth = controlnet_image_depth.resize(src.size)
        
        
        return face_embedding, sr_src, controlnet_image_face, controlnet_image_depth, src

    tasks = [process_photo(photo) for _, photo in batch]
    results = await asyncio.gather(*tasks)
    for result in results:
        face_embedding, sr_src, controlnet_image_face, controlnet_image_depth, src = result
        input_images.append(sr_src)
        face_embeddings.append(face_embedding)
        controlnet_image_faces.append(controlnet_image_face)
        controlnet_image_depths.append(controlnet_image_depth)
        prompts.append(prompt)
        negative_prompts.append(n_prompt)
    
    onnx_session_control(cuda=False)

    face_embeddings = np.stack(face_embeddings)

    torch.cuda.empty_cache()
    gc.collect()
         
    return input_images, prompts, negative_prompts, controlnet_image_faces ,controlnet_image_depths, face_embeddings

async def face_detailer_gen(image, face_detailer):
    mask_image, retouched_image = await asyncio.to_thread(face_detailer.generate_retouch,image=image, 
                                                                                         strength=0.25, 
                                                                                         generator=torch.Generator("cuda").manual_seed(111))
    
    del mask_image

    torch.cuda.empty_cache()
    gc.collect()

    return retouched_image

async def ip_gen(input_images, prompts, negative_prompts, controlnet_image_faces ,controlnet_image_depths, face_embeddings, pipe_ip):
    if len(input_images) == 1:
        ip_image = await asyncio.to_thread(pipe_ip,
            input_image = input_images[0],
            prompt =  prompts[0],
            negative_prompt = negative_prompts[0],
            image = [controlnet_image_faces[0], input_images[0], controlnet_image_depths[0]],
            ip_adapter_image_embeds = face_embeddings[0],
            strength = 0.7,
            num_inference_steps = 30,
            guidance_scale = 7.5,
            width = input_images[0].size[0],
            height = input_images[0].size[1],
            controlnet_conditioning_scale = [0.5, 0.8, 0.3],
            ip_adapter_scale = 0.5,
            generator = torch.Generator("cuda").manual_seed(777),
            clip_skip=2,
            num_images_per_prompt=1
            )
    else:
        ip_image = await asyncio.to_thread(pipe_ip,
            input_image = input_images,
            prompt =  prompts,
            negative_prompt = negative_prompts,
            image = [controlnet_image_faces, input_images, controlnet_image_depths],
            ip_adapter_image_embeds = face_embeddings,
            strength = 0.7,
            num_inference_steps = 30,
            guidance_scale = 7.5,
            width = input_images[0].size[0],
            height = input_images[0].size[1],
            controlnet_conditioning_scale = [0.5, 0.8, 0.3],
            ip_adapter_scale = 0.5,
            generator = torch.Generator("cuda").manual_seed(777),
            clip_skip=2,
            num_images_per_prompt=1
            )
   
    torch.cuda.empty_cache()
    gc.collect()

    return ip_image.images

async def generate(input_images, prompts, negative_prompts, controlnet_image_faces ,controlnet_image_depths, face_embeddings, pipe_ip, face_detailer):
    with torch.no_grad():
        try:
            gen = await ip_gen(input_images, prompts, negative_prompts, controlnet_image_faces ,controlnet_image_depths, face_embeddings, pipe_ip)
        except Exception as e:
            print("ip_gen 오류 : ", e)
            gen = input_images
        gen = await face_detailer_gen(gen, face_detailer)

    torch.cuda.empty_cache()
    gc.collect()

    return gen

async def process_batch():
    while True:
        batch = []
        while not request_queue.empty() and len(batch) < 1:
            try:
                request = await asyncio.wait_for(request_queue.get(), timeout = 0.1)
                batch.append(request)
            except asyncio.TimeoutError:
                 break
        
        if batch:
            try:
                input_images, prompts, negative_prompts, controlnet_image_faces ,controlnet_image_depths, face_embeddings = await make_batch_files(batch)
                processed_images = await generate(input_images, prompts, negative_prompts, controlnet_image_faces ,controlnet_image_depths, face_embeddings, pipe_ip, face_detailer)
            except:
                processed_images = [photo for _, photo in batch]
            for (request_id, _), processed_image in zip(batch, processed_images):
                processed_image = processed_image.resize((544, 680))
                buffer = io.BytesIO()
                processed_image.save(buffer, format="PNG")
                buffer.seek(0)
                responses[request_id] = buffer
                response_event[request_id].set()
            print(f"Processed and stored batch of size {len(batch)}")

            torch.cuda.empty_cache()
            gc.collect()

        else:
            await asyncio.sleep(0.01)

async def flash2request_queue():
    while True:
        if flash_queue: 
            requests_to_transfer = flash_queue[:]
            flash_queue.clear()  

            for request in requests_to_transfer:
                await request_queue.put(request)
                print(f"Transferred to processing queue: {request[0]}")
        else:
            await asyncio.sleep(0.01)  # 더 짧은 대기 시간 설정

async def update_request_queue_tag():
    """매 1초마다 request_queue와 flash_queue의 요청 갯수를 EC2 태그에 업데이트하는 비동기 함수"""
    while True:
        # 현재 요청 갯수 (request_queue와 flash_queue에 남은 요청)
        pending_requests = request_queue.qsize() + len(flash_queue)
        
        # 태그 값은 문자열로 전달합니다.
        if int(os.getenv("APP_PORT", 8000)) == 8000:
            update_ec2_tag(INSTANCE_ID, "p1_queue_count", str(pending_requests));
        elif int(os.getenv("APP_PORT", 8001)) == 8001:
            update_ec2_tag(INSTANCE_ID, "p2_queue_count", str(pending_requests));
        await asyncio.sleep(1)



async def monitor_inactivity():
    global LAST_REQUEST_TIME
    while True:
        await asyncio.sleep(300)
        elapsed_time = datetime.now() - LAST_REQUEST_TIME
        if elapsed_time > timedelta(minutes=15):
            print("No reqeusts for 15 minutes. Updating status to 'Termination'")
            update_ec2_tag(INSTANCE_ID, "status", "termination")
            if int(os.getenv("APP_PORT", 8000)) == 8000:
                decrease_desired_capacity(get_active_branch().replace('d', 'p'))

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(flash2request_queue())
    asyncio.create_task(process_batch())
    asyncio.create_task(monitor_inactivity())
    asyncio.create_task(update_request_queue_tag())


@app.post('/api/photo/ai')
async def ai(
    request: Request,
    ):

    tic = time.time()

    global LAST_REQUEST_TIME
    LAST_REQUEST_TIME = datetime.now()
    update_ec2_tag(INSTANCE_ID, "status", "on-demand")

    try:
        body = await request.json()
        url = body.get("imageUrl")
        
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Fail to download image from imageUrl")
        photo = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    print(f"Request received on port : {request.url.port}")
    request_id  = str(uuid.uuid4())
    event = asyncio.Event()
    response_event[request_id] = event

    flash_queue.append((request_id, photo))
    print(f"Added to flash_queue: Request ID {request_id}")

    await event.wait()
    print(f"Request {request_id} completed!")

    buffer = responses.pop(request_id)
    response_event.pop(request_id)

    toc = time.time()
    print(f"Elapsed Time : {toc-tic:.2f} seconds")

    torch.cuda.empty_cache()
    gc.collect()
    
    return StreamingResponse(buffer, media_type="image/png")[root@ip-10-0-7-39 ~]# 
