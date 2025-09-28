# Dockerfile - 원본 이미지 기반 (가장 안전한 방법)
FROM pbup/photobooth:20250914-clean

# 작업 디렉토리로 이동
WORKDIR /root/photobooth

# 기존 소스코드 삭제
RUN rm -rf ./*

# 새 소스코드 복사
COPY . .

# Python 캐시 정리
RUN find . -name "__pycache__" -exec rm -rf {} + || true && \
    find . -name "*.pyc" -delete || true

# 모델 파일 디렉토리는 볼륨 마운트용으로 빈 디렉토리 유지
RUN mkdir -p ckpt

# 권한 설정
RUN chmod -R 755 .

# 포트 노출
EXPOSE 8080

# 애플리케이션 실행 (원본과 달리 uvicorn으로 시작)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]