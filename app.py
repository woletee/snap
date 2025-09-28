import asyncio
import uuid
import torch
import requests
import httpx
from io import BytesIO
from PIL import Image
from photobooth import PHOTOBOOTH
from fastapi import (
    FastAPI, Request, Response,
    File, Form, UploadFile
)
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
request_queue = asyncio.Queue()
response_event = {}
response_result = {}

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
photobooth = PHOTOBOOTH(device=device, seed=777, style="disney", model_dir="/root/photobooth/ckpt")

executor = ThreadPoolExecutor(max_workers=1)

async def generate():
    while True:
        batch = []
        while not request_queue.empty() and len(batch) < 4:
            try:
                request = await asyncio.wait_for(request_queue.get(), timeout=0.1)
                batch.append(request)
            except asyncio.TimeoutError:
                break

        if batch:
            request_ids = [request_id for request_id, _ in batch]
            images = [image for _, image in batch]

            try:
                generated_images = await asyncio.get_running_loop().run_in_executor(executor, photobooth.inference, images)
            except:
                generated_images = images

            for request_id, generated_image in zip(request_ids, generated_images):
                buffer = BytesIO()
                generated_image.save(buffer, format="PNG")
                buffer.seek(0)
                response_result[request_id] = buffer
                response_event[request_id].set()
        else:
            await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(generate())

@app.post('/api/photo/ai')
async def ai(
    request : Request
):
   
    body = await request.json()
    url = body.get('imageUrl')

    photo = await asyncio.to_thread(lambda: Image.open(BytesIO(requests.get(url).content)).convert("RGB"))

    request_id = str(uuid.uuid4())
    event = asyncio.Event()
    response_event[request_id] = event
    await request_queue.put((request_id, photo))

    await event.wait()

    buffer = response_result.pop(request_id)
    response_event.pop(request_id)

    return StreamingResponse(buffer, media_type="image/png")

