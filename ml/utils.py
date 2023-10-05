import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from models import Residual_Block, Pix2Pix_Generator
from PIL import Image
import io
from io import BytesIO


def preprocess_edge(edge_img):
    '''
    edge_img를 모델에 넣기 전 전처리 해주는 함수
        range : [0, 255] (int, uint8) => [0, 1] (tf.float32)
        shape : [W, H, C] => [1, W, H, C]
    
    <params>
        edge_img (np.array) : input 이미지가 될 numpy array
    '''

    edge_img = tf.cast(edge_img, tf.float32)
    edge_img = edge_img / 255.0
    edge_img = tf.clip_by_value(edge_img, 0, 1)
    edge_img = tf.expand_dims(edge_img, axis=0)

    return edge_img


def load_and_preprocess_edge(img_response):
    '''
    s3에서 가져온 이미지를 로드하고 model의 input으로 활용할 수 있도록 전처리 해주는 함수

    <params>
        img_response (bytes array) : s3로부터 http의 get 메서드를 통해서 받은 response의 content
    '''

    sketch = Image.open(io.BytesIO(img_response))
    sketch = sketch.resize((256, 256))

    # 배경이 투명한 이미지인 경우 처리해 주자
    if sketch.mode == "RGBA":
        white_background = Image.new("RGB", sketch.size, (255, 255, 255))
        white_background.paste(sketch, mask=sketch.split()[3])
        sketch = white_background

    sketch = sketch.convert('L')
    sketch = np.array(sketch)
    sketch = np.expand_dims(sketch, axis=-1)
    sketch = preprocess_edge(sketch)

    return sketch


def postprocess_result(result):
    '''
    model의 result를 이미지로 저장하기 쉽도록 후처리 해주는 함수
        range : [-1, 1] (tf.float32) => [0, 255] (np.uint8)
        shape : [H, W, 3] or [1, H, W, 3] => [H, W, 3]
    
    <params>
        result (tf.Tensor) : model의 output Tensor
    '''

    result = (result + 1) * 127.5
    result = tf.cast(result, tf.int32)
    result = tf.clip_by_value(result, 0, 255)
    result = np.array(result).astype(np.uint8)

    # batch dim 제거
    if len(result.shape) == 3:
        return result
    elif len(result.shape) == 4:
        return result[0]


def load_model(target_name):
    '''
    target_name에 해당하는 모델을 load한 뒤 반환해주는 함수

    <params>
        target_name (string) : 모델 이름
    '''
    
    generator = Pix2Pix_Generator(input_channels=1, output_channels=3, name=f"{target_name}_generator")
    ckpt = tf.train.Checkpoint(generator=generator)
    ckpt.restore(f"./checkpoints/{target_name}/ckpt-1")

    return generator