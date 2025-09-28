import cv2
import numpy as np
import math
from PIL import Image
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor

# 눈, 코, 입술, 볼 등 얼굴 영역에 대한 정보
face_areas_info = {
    'cheekbones': {
        'points': np.arange(0, 33), 
        'color': (0, 255, 255),
    },
    'right_eye': {
        'points': np.arange(33, 52), 
        'color': (255, 0, 0),
    },
    'lips': {
        'points': np.arange(52, 72), 
        'color': (255, 255, 0),
    },
    'nose': {
        'points': np.arange(72, 87), 
        'color': (0, 0, 255),
    },
    'left_eye': {
        'points': np.arange(87, 106), 
        'color': (0, 255, 0),
    },
}

# 얼굴 특징을 2D로 그리는 함수
def draw_2d_kps(image, kps, point_radius=3, write_numbers=False, overlap_image=False):
    kps = np.array(kps)
    h, w = image.shape[:2]
    
    out_img = np.zeros([h, w, 3], dtype=np.uint8)
    if overlap_image:
        out_img = image.copy()
    
    for i, (x, y) in enumerate(kps):
        x, y = int(x), int(y)

        for face_area, area_info in face_areas_info.items():       
            if i in area_info['points']:
                color = area_info['color']
                out_img = cv2.circle(out_img, (x, y), point_radius, color, -1)

                if write_numbers:
                    out_img = cv2.putText(
                        out_img,
                        f'{i}',
                        (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.3, 
                        (255, 255, 255),
                        1, 
                        cv2.LINE_AA)
                break
    return out_img

# 얼굴 특징을 5개의 주요 키포인트로 그리는 함수
def draw_five_kps(image, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w = image.shape[:2]
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)
    return out_img


def load_face_app(
        name='antelopev2',
        root='root/data/photobooth/models/antelopev2/antelopev2'
):
    face_app = FaceAnalysis(
        name=name,
        root=root,
        providers=["CUDAExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app

def prepare_face_embeddings(
        source_image: Image,
        face_app : FaceAnalysis,
        force_zero: bool = False
):
    # 얼굴 특징 정보 추출
    source_np = np.array(source_image)
    face_info = face_app.get(source_np)
    
    def process_face_features(face_info, source_np):
        keypoints = draw_five_kps(source_np, face_info["kps"])
        keypoints = draw_2d_kps(keypoints, face_info["landmark_2d_106"], overlap_image=True)
        return keypoints, face_info["embedding"].astype(np.float32)

    if not face_info:
        # 얼굴 정보가 없을 시, zero 벡터 반환 및 검은 이미지 반환
        return np.zeros((512,)), Image.fromarray(np.zeros_like(source_np, dtype=np.uint8))
    
    # 멀티스레딩을 사용하여 얼굴 특징 추출
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda face: process_face_features(face, source_np), face_info))
    
    keypoints_images, embeddings = zip(*results)
    keypoints_images = np.clip(np.sum(keypoints_images, axis=0), 0, 255).astype(np.uint8)
    embeddings = np.mean(embeddings, axis=0)

    if force_zero:
        return np.zeros((512,)), Image.fromarray(np.zeros_like(source_np, dtype=np.uint8))
    else:
        # 얼굴 특징 벡터와 키포인트 이미지 반환
        return embeddings, Image.fromarray(keypoints_images)
        
        

