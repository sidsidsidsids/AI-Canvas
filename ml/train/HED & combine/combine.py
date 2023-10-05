import cv2
import os
import numpy as np

target_name = "maple_character4"

# 각 데이터들이 들어있는 디렉토리
edge_image_directory = os.path.join(os.getcwd(), f'edge_images/{target_name}')
image_directory = os.path.join(os.getcwd(), f'images/{target_name}')


# 이미지를 합쳐주는 함수
def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
    # im_A = im_A[10:-10, 10:-10]
    im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
    # im_B = im_B[10:-10, 10:-10]
    im_AB = np.concatenate([im_A, im_B], axis=1) # 가로로 합치기
    cv2.imwrite(path_AB, im_AB)


for index, image_path in enumerate(os.listdir(image_directory)):

    if (index % 100) == 0:
        print(f'processing {index + 1}th')

    image_name = image_path.split('/')[-1].split('.')[0]

    path_A = os.path.join(edge_image_directory, image_name + '_edge.jpg')
    path_B = os.path.join(image_directory, image_name + '.png')
    path_AB = os.path.join(os.getcwd(), 'combined_images/' + target_name + '/' + image_name + '.jpg')

    image_write(path_A, path_B, path_AB)