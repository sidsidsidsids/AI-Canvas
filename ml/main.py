import json
from boto3 import client
from botocore.exceptions import NoCredentialsError
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from aio_pika import connect, IncomingMessage, Message
from dotenv import load_dotenv
import os
import tensorflow as tf
import tensorflow_addons as tfa
from models import Residual_Block, Pix2Pix_Generator
from utils import preprocess_edge, postprocess_result, load_and_preprocess_edge, load_model
import numpy as np
from PIL import Image
import requests
import io
from io import BytesIO
import time

# https://velog.io/@cho876/%EC%9A%94%EC%A6%98-%EB%9C%A8%EA%B3%A0%EC%9E%88%EB%8B%A4%EB%8A%94-FastAPI
# uvicorn main:app --reload
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 이름 모아둔 문자열 배열
target_name_list = ["person",           # trained for 28 epoches (with batch_size = 4) using 9996 images
                    "panda",            # trained for 180 epoches (with batch_size = 1) using 300 images
                    "car",              # trained for 19 epoches (with batch_size = 4) using 11476 images
                    "handbag",          # trained for 5 epoches (with batch_size = 4) using 138567 images
                    "shoe",             # trained for 25 epoches (with batch_size = 4) using 49825 images
                    "maple_character",  # trained for 14 epoches (with batch_size = 4) using 69372 images
                    "gemstone",         # trained for roughly 36 epoches (with batch_size = 4) using 3219 images
                    "space",            # trained for 40 epoches, (with batch_size = 4) using 4649 images
                    ]

# 모델들을 모아둘 빈 딕셔너리 생성
model_zoo = dict()

# 각각의 모델 load
for target_name in target_name_list:
    model_zoo[target_name] = load_model(target_name)
    model_zoo[target_name]._name = f"{target_name}_generator"

# 환경 변수 로드
load_dotenv()

# AWS S3 클라이언트 생성
s3 = client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    region_name=os.getenv('AWS_REGION')  # AWS 지역을 설정합니다.
)

# RabbitMQ 연결 설정
RABBITMQ_URL = os.getenv('RABBITMQ_URL')
request_queue_name = 'sketch_conversion_request_queue'
response_queue_name = 'sketch_conversion_response_queue'
demo_request_queue_name = 'demo_conversion_request_queue'
demo_response_queue_name = 'demo_conversion_response_queue'

connection = None
channel = None

@app.on_event("startup")
async def startup():
    global connection, channel
    connection = await connect(RABBITMQ_URL)
    channel = await connection.channel()
    
    # 본 서비스
    queue = await channel.declare_queue(request_queue_name, durable=True)
    await queue.consume(on_message) 
    
    # 랜디 페이지
    demo_queue = await channel.declare_queue(demo_request_queue_name, durable=True)
    await demo_queue.consume(on_demo_message)


# 메시지 처리 함수 - 오리지널 서비스
async def on_message(message: IncomingMessage):
    async with message.process():
        print("Received message: ", message.body)
        data = json.loads(message.body)
        
        sketch_url = data.get('sketchUrl')
        canvas_id = data.get('canvasId')
        profile_id = data.get('profileId')
        subject = data.get('modelName')
        
        # # 파일 경로 생성
        file_name = f"{subject}_{int(time.time())}"
        file_path = f'profile/{profile_id}/canvas/{subject}/{time.time_ns()}.JPG'

        ######## 변환 코드 들어갈 부분 ########
        sketch_response = requests.get(sketch_url)

        # 성공 시 & 그러한 모델이 있을 경우
        if sketch_response.status_code == 200 and (subject in target_name_list):
            # 이미지를 url로부터 불러와서 전처리
            sketch = load_and_preprocess_edge(sketch_response.content)

            # inference & post processing
            result = model_zoo[subject](sketch)
            result = postprocess_result(result)

            # S3에 업로드
            result = Image.fromarray(result)
            file = BytesIO()
            result.save(file, 'JPEG')
            file.seek(0)
            s3.upload_fileobj(file, os.getenv('AWS_BUCKET_NAME'), file_path)
            file.close()
        else:
            print("Failed to retrieve the image.")

        ######################################
        
        # 파일의 S3 URL 생성
        file_url = f"https://{os.getenv('AWS_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{file_path}"

        # 응답 생성
        response_data = {
            "canvasUrl": file_url,
            "canvasName": file_name,
            "canvasId": canvas_id,
            "status": "SUCCESS"
        }

        # 응답을 sketch_conversion_response_queue에 보냄
        await channel.default_exchange.publish(
            Message(body=json.dumps(response_data).encode('utf-8')),
            routing_key=response_queue_name
        )

# 데모 서비스
async def on_demo_message(message: IncomingMessage):
    async with message.process():
        print("Received demo message: ", message.body)
        data = json.loads(message.body)
        sketch = data.get("sketchUrl")
        subject = data.get("modelName")
        temp_id = data.get("tempId")

        # # 파일 경로 생성
        file_name = f"{subject}_{int(time.time())}"
        file_path = f'demo/{time.time_ns()}.JPG'

        sketch_response = requests.get(sketch)

        # 성공 시 & 그러한 모델이 있을 경우
        if sketch_response.status_code == 200 and (subject in target_name_list):
            # 이미지를 url로부터 불러와서 전처리
            sketch = load_and_preprocess_edge(sketch_response.content)

            # inference & post processing
            result = model_zoo[subject](sketch)
            result = postprocess_result(result)

            # S3에 업로드
            result = Image.fromarray(result)
            file = BytesIO()
            result.save(file, 'JPEG')
            file.seek(0)
            s3.upload_fileobj(file, os.getenv('AWS_BUCKET_NAME'), file_path)
            file.close()
        else:
            print("Failed to retrieve the image.")

        file_url = f"https://{os.getenv('AWS_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{file_path}"

                # 응답 생성
        response_data = {
            "canvasUrl": file_url,
            "tempId": temp_id,
            "status": "SUCCESS"
        }

        await channel.default_exchange.publish(
            Message(body=json.dumps(response_data).encode('utf-8')),
            routing_key=demo_response_queue_name
        )

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/temp")
async def temp(model_name: str):
    start = time.time()
    
    for i in range(1, len(os.listdir(f"./test_edges/{model_name}")) + 1):
        # preprocess edge
        edge_img = tf.io.decode_image(tf.io.read_file(f"./test_edges/{model_name}/sketch{i}.jpg"), channels=1)
        edge_img = preprocess_edge(edge_img)

        # run the generator & postprocess the result
        result = model_zoo[model_name](edge_img)
        result = postprocess_result(result)

        img = np.array(result).astype(np.uint8)
        img = Image.fromarray(np.array(img))
        img.save(f"./test_results/{model_name}/result{i}.jpg")

    end = time.time()

    print(f"{end - start:.5f} sec")

    return "success"

@app.post("/upload/")
async def create_upload_file(file: UploadFile):
    try:
        object_key = file.filename
        s3.upload_fileobj(file.file, os.getenv('AWS_BUCKET_NAME'), object_key)

        # 파일의 S3 URL 생성
        file_url = f"https://{os.getenv('AWS_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{object_key}"

        return {"filename": file.filename, "url": file_url, "message": "File uploaded successfully!"}
    except FileNotFoundError:
        return {"error": "File not found"}
    except NoCredentialsError:
        return {"error": "Credentials not available"}
