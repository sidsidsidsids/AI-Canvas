from PIL import Image
import os
import numpy as np

image_path = "./images_without_bg/sharks"
squared_image_path = "./images_squared/sharks"

index = 0
for file_name in os.listdir(image_path):
    index += 1
    if (index % 50 == 0):
        print(f"processing {index}th")

    img = Image.open(os.path.join(image_path, file_name))
    img = np.array(img)
    H, W, _ = img.shape

    if H > W:
        pad_size = (H - W) // 2

        if (H - W) % 2:
            img = np.concatenate((np.full((H, pad_size, 3), 255, dtype=np.uint8), img, np.full((H, pad_size + 1, 3), 255, dtype=np.uint8)), axis=1)
        else:
            img = np.concatenate((np.full((H, pad_size, 3), 255, dtype=np.uint8), img, np.full((H, pad_size, 3), 255, dtype=np.uint8)), axis=1)  
    else:
        pad_size = (W - H) // 2

        if (W - H) % 2:
            img = np.concatenate((np.full((pad_size, W, 3), 255, dtype=np.uint8), img, np.full((pad_size + 1, W, 3), 255, dtype=np.uint8)), axis=0)
        else:
            img = np.concatenate((np.full((pad_size, W, 3), 255, dtype=np.uint8), img, np.full((pad_size, W, 3), 255, dtype=np.uint8)), axis=0)

    img = Image.fromarray(img, "RGB")
    img = img.resize((256, 256))
    img.save(os.path.join(squared_image_path, file_name))
    