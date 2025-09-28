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
1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
2. 애플리케이션 실행
   ```bash
   python app.py
   ```
3. Docker로 실행
   ```bash
   docker build -t photobooth-ai .
   docker run -p 5000:5000 photobooth-ai
   ```

## 저작권 및 사용 범위
본 프로젝트는 projectbuildup의 사내 전용 소프트웨어입니다. 무단 복제, 배포, 사용을 금합니다.

Copyright © 2025 projectbuildup. All rights reserved.   