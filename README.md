# Photobooth AI

Photobooth AI는 인공지능을 활용하여 사진을 자동으로 생성, 편집, 관리할 수 있는 프로젝트입니다.

## 주요 기능
- 얼굴 인식 및 처리
- 사진 자동 생성 및 저장
- 샘플 이미지 제공
- Docker를 통한 손쉬운 배포

## 폴더 구조
```
app.py                # 메인 애플리케이션 파일
Dockerfile            # Docker 이미지 빌드 파일
photobooth.py         # 포토부스 기능 구현 파일
generated/            # 생성된 이미지 저장 폴더
InstantID/            # 얼굴 인식 및 처리 관련 모듈
samples/              # 샘플 이미지 폴더
utils/                # 유틸리티 함수 모음
```

## 실행 방법

### Docker로 실행 (권장)
1. Docker 이미지 pull 및 컨테이너 실행
   ```bash
   docker run --gpus all -p 80:8080 -it <image_name>
   ```

2. 컨테이너 내부에서 애플리케이션 실행
   ```bash
   # 컨테이너 내부에서 실행
   uvicorn app:app --host 0.0.0.0 --port 8080
   ```

3. 브라우저에서 접속
   - 로컬 환경: `http://localhost`
   - 원격 서버: `http://{GPU 인스턴스 외부 IP}:80`

> **⚠️ 주의사항**  
> 이 애플리케이션은 AI 모델 실행을 위해 GPU가 필요합니다. 로컬 환경에서 실행하려면 NVIDIA GPU와 CUDA가 설치되어 있어야 합니다. GPU가 없는 환경에서는 원격 GPU 서버에 Docker 컨테이너를 띄워서 사용하는 것을 권장합니다.

### 로컬 개발 환경 실행
1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
2. 애플리케이션 실행
   ```bash
   python app.py
   ```

### Docker 컨테이너 관리
- 실행 중인 컨테이너에 접속하기
  ```bash
  docker exec -it <container_id> /bin/bash
  ```
- 컨테이너 ID 확인하기
  ```bash
  docker ps
  ```

## 저작권 및 사용 범위
본 프로젝트는 projectbuildup의 사내 전용 소프트웨어입니다. 무단 복제, 배포, 사용을 금합니다.

Copyright © 2025 projectbuildup. All rights reserved.   