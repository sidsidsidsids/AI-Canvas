# 아이캔버스 [Backend-AI]

💡 아이들이 스케치를 그리면 생성형 AI가 이를 그림으로 변환해 주는 서비스입니다. 게임적인 요소를 더해 아이들이 스스로 자신만의 콘텐츠를 생성하면서 자신감과 창의력을 증진하고, 재미와 성취감을 경험할 수 있도록 합니다.

- 참고 문헌 : [Pix2Pix 프로젝트 페이지](https://phillipi.github.io/pix2pix/) | [Github](https://github.com/phillipi/pix2pix) | [Paper](https://arxiv.org/abs/1611.07004) | [HED paper](https://arxiv.org/abs/1504.06375)

<br>

- 데이터 준비 과정이 궁금하신 분은 [데이터 전처리 Section](#데이터-전처리)을 참고해주세요.

- 모델이 돌아가는 원리가 궁금하신 분은 [모델 소개 Section](#모델-소개)을 참고해주세요.

- 모델의 결과를 보고 싶으신 분은 [사용한 데이터 Section](#사용한-데이터)을 참고해주세요.

<br>

## 서비스 예시 화면

- 작성중

<br>

## AI 파트의 목표

🚩 궁극적인 목표 : 서비스의 핵심 기능 중 하나인 그린 스케치를 해당 그림으로 변환해주는 generator 학습 <br>

- 이를 달성하기 위해서 해야할 일
  - 데이터 수집
  - 데이터 전처리
  - 모델 학습
  - 모델 비교 & checkpoint 결정
  - FastAPI 서버에 모델 띄우기

<br>

## 데이터 전처리

- 많은 경우에 이미지 데이터와 해당하는 edge가 함께 제공되지 않습니다.
- 이미지 수가 많기에 이를 직접 그리는 것은 현실적으로 어려우니, pix2pix 저자들의 implementation을 참고, [HED(Holistically-Nested Edge Detection)](https://github.com/s9xie/hed)로 edge를 추출한 뒤, post-processing 작업을 거쳤습니다.
  - [pix2pix github](https://github.com/phillipi/pix2pix)의 Extracting Edges Section을 참고해주세요

<br>

- 결론적으로 저렇게 이어붙여서 학습 시에 불러오게 됩니다
- 예시 이미지 : [DVM Car Dataset](https://deepvisualmarketing.github.io/), bmw series 5
![combined bmw](./docs%20images/combined_bmw.jpg)

<br>

※ <b>Distribution Mismatch</b>

- 학습을 시키는 데이터는 HED에 의해 자동으로 추출된 edge인데, 실제 사용자가 이를 따라 그리기는 현실적으로 어렵습니다.
- 그래서 학습시키는 데이터와 궁극적으로 적용하고자 하는 데이터에 <b>distribution mismatch</b>가 발생하는데, 일반적으로 이는 실제 적용 시에서의 성능 저하를 초래할 수 있다는 점을 짚고 넘어가야 합니다.

<br>

- 그렇다 하더라도 저희가 궁극적으로 원하는 것은 유저가 그린 edge를 잘 변환해주는 generator이기 때문에 저희는 직접 edge를 그려서 유저가 잠재적으로 그릴 만한, 그릴 수 있는 수준의 edge를 기준으로 결과를 비교해서 모델을 선정했습니다.

| HED에 의해 자동으로 추출된 edge | 직접 그려본 edge |
| :---: | :---: |
|![edge_hed](./docs%20images/bmw_edge_hed.jpeg)|![edge_drawn](./docs%20images/bmw_edge_drawn.jpg)|

- 저자들도 마찬가지로 이를 유념하여, paper 부록에 학습 데이터에 대한 결과 뿐만 아니라, 사람이 그린 데이터에 대한 결과도 함께 첨부했습니다.

<br>

## 모델 소개
- pix2pix는 conditional GAN(Generative Adversarial Network)를 이용해서 (paired) Image-to-Image Translation 문제에 접근합니다

<br>

<b>배경 지식</b>
- (paired) Image-to-Image Translation이란 말 그대로, 어떤 한 이미지가 주어졌을 때 이를 대응하는 한 이미지로 바꾸는 것입니다
- 여기서 paired의 의미는 어떤 구조를 공유하는, (input으로 output을 어느 정도 설명 가능한) (input, output) pair가 있는 환경을 말합니다
- 예시 적용 사례
![pix2pix 예제들](./docs%20images/pix2pix%20examples.png)

<br>

- GAN이란 Generative Adversarial Network의 약자로, generator와 discriminator 두개의 neural network로 이루어져 있는데, Generative : 뭔가를 만들어내는, Adversarial : generator와 discriminator가 뭔가 서로 경쟁한다는(or 도움을 주는) 뜻을 담고 있습니다
    - generator의 목적은 사실적인 데이터(image, audio 등)를 만들어내는 것입니다
        - 여기서 사실적이라 함은, (discriminator가) 실제 데이터와 구별하기 어려운 것을 말합니다
    - discriminator의 목적은 어떤 데이터(image, audio 등)가 주어졌을 때, 이것이 generator가 만들어낸 fake 데이터인지, 혹은 real 데이터인지 구별해내는 것입니다

<br>

<b>소개</b>
- pix2pix는 input image가 주어지면 해당하는 타겟의 output image로 바꿔주는 paired Image-to-Image Translation Task를 위한 모델입니다
- 많은 이전의 GAN이 noise를 input으로 주면 output을 반환하는 것에 비해, pix2pix에서는 input으로 condition(변환하고자 하는 이미지)을 주고 별도의 noise는 주지 않습니다
    - 그래서 condition이 동일한 한 generator는 deterministic하게 됩니다
        - 즉, 같은 condition이 주어지면 같은 결과 이미지를 만들어내게 됩니다
- 저자들도 처음에는 noise도 같이 주는 방향을 고려했지만 그리 효과적이지 않아서 제외했다고 합니다
    - generator가 noise를 무시하는 쪽으로 학습하는 경향을 보였다고 합니다

>Past conditional GANs have acknowledged this and provided Gaussian noise z as an input to the generator, in addition to x (e.g., [55]). In initial experiments, we did not find this strategy effective – the generator simply learned to ignore the noise – which is consistent with Mathieu et al. (출처 : pix2pix paper)

<br>

<b>Generator</b>
![generator 사진](./docs%20images/generator.png)
- input : (H, W, 1) image tensor(흑백), 범위 : [0, 1]
- output : (H, W, 3) image tensor(컬러), 범위 : [-1, 1]
- 저자들이 이 논문을 발표할 때는 U-net 구조를 사용했지만, 저희는 이후 CycleGAN에서 사용했던 Resnet 기반의 generator를 사용했습니다
- conv + 2 Contracting Blocks + 9 Residual Blocks + 2 Expanding Blocks + conv
    - c7s1-64, d128, d256, R256 * 9, u128, u64, c7s1-3 (CycleGAN 저자들의 Notation 참고)
    - Contracting Block : conv + instance_norm ⇒ width & height를 절반으로 줄입니다
    - Residual Block : conv + instance_norm + relu + conv + instance_norm + input과의 skip_connection ⇒ width & height를 그대로 유지합니다
    - Expanding Block : transposed_conv + instance_norm ⇒ width & height를 2배로 늘립니다
    - padding : reflection_pad
    - 결론적으로 input과 output의 이미지 크기는 동일합니다
- 파라미터 수 : 11,377,155개 (condition이 흑백 edge channel 하나일 경우)

<br>

<b>Discriminator</b>
![discriminator 사진](./docs%20images/discriminator.png)
- input : (H, W, 4) image tensor(real or fake image + condition(우리의 경우 edge)), 범위 : [-1, 1]
    - 타겟 이미지 뿐만 아니라, 그 타겟을 만들어내기 위한 condition을 타겟 이미지의 채널 축에 붙입니다
    - 이는 기존의 conditional GAN에서 그랬듯, 단순히 결과 이미지만 가지고 그 이미지가 진짜인지 아닌지 구별하는 것보다, 그 결과 이미지가 저 컨디션으로부터 나왔을 때 진짜인지 가짜인지 구별하는 것이 성능에 더 좋았기에 그랬다고 합니다
- output : patchGAN output tensor, 범위 : 제한 없음, but 0에 가까울수록 discriminator는 가짜로 판단하는 것이고, 1에 가까울수록 진짜로 판단하는 것입니다
- discriminator로는 PatchGAN discriminator을 사용합니다
- 이전의 많은 GAN이 이미지 전체를 한번에 보고, 이게 real인지 fake인지 구분했다면, PatchGAN discriminator는 이미지 전체를 한번에 보지 않고, 각각 해당하는 이미지 Patch 별로 그 부분이 사실적인가(real distribution과 discriminator가 구분할 수 없는가) 아닌가(fake)를 판단합니다
    - 그리고 추후에 살펴볼 loss에서 이 정보들을 취합합니다
    - 여기서 주의할 점은, 이미지를 여러 개의 patch로 잘라서 하나하나 넣는 것이 아니라, 모두 한번에 convolution의 성질을 이용해서 진행하게 됩니다
        - output의 각 cell들이 보는 patch는 서로 겹칠 수 있습니다
- PatchGAN은 기본적으로 receptive field size 보다 더 멀리 있는 픽셀들은 서로 독립적이라고 가정을 하기 때문에 다른 말로 Markovian Discriminator, Local-patch Disctiminator라고도 불린다고 합니다

- 아래는 저자들이 테스트 해본 여러가지 receptive field size(discriminator의 output중 한 cell이 보는 입력 이미지 patch 크기, in pixel)에 대한 결과 예시입니다
![receptive field size](./docs%20images/receptive%20field%20size.png)
- 저희는 저자들의 선택을 참고해서 70을 사용했습니다
- C64 - C128 - C256 - C512 - output layer (CycleGAN 저자들의 notation 참고)
- 파라미터 수 : 2,765,633개 (condition이 흑백 edge channel 하나일 경우)

<br>
<b>loss</b>
- pix2pix는 다음 objective를 기준으로 학습합니다

![pix2pix total objective](./docs%20images/pix2pix%20total%20objective.png)

- 여기서 G는 generator이고, D는 discriminator입니다
- 구성하는 것을 두가지로 나눠보면 크게 L_cGAN과 L_L1으로 나눌 수 있습니다
    - L_cGAN
        - 이는 conditional gan에서 사용하던 loss와 동일합니다
        - 단, 저희는 cross entropy loss대신에 LSGAN에서 사용되었던 least square adversarial loss를 사용했습니다
            - 즉, generator는 input condition x(우리의 경우 sketch)를 넣어서 생성된 결과 G(x)가 discriminator에게 사실적인 이미지처럼 보이도록 학습하고
            - discriminator는 y(실제 target)은 1로, 가짜(G(x))는 0이 되도록 학습합니다
    - L_L1 loss
        - input을 generator에 넣어서 생성된 이미지와 원하는 target 사이의 pixel level에서의 L1 distance를 계산합니다
        
        ![pix2pix l1 loss](./docs%20images/pix2pix%20L1%20loss.png)
        
        - 쉽게 말하자면, model의 output과 target(이상적인 결과)의 픽셀간의 절댓값 차이(RGB 모두)를 모두 구한 후 이를 평균내면 됩니다
        - 저자들은 전체적으로 비교적 blurry한 결과를 만드는 L2(절댓값 대신 제곱합을 사용) 대신 L1 distance를 사용했습니다
        - 우리가 궁극적으로 원하는 것은 input을 target으로 잘 바꿔주는 generator이기 때문에 자연스러운 선택으로 받아들일 수 있습니다

```Python
def discriminator_loss_function(real_D_out, fake_D_out):
    '''
    LSGAN loss

    <params>
        real_D_out : 실제 이미지가 주어졌을 때, discriminator의 결과값
        fake_D_out : 가짜 이미지가 주어졌을 때, discirminator의 결과값
    '''

    # 저자들의 방식에 따라, 2로 나눔으로써 D가 배우는 속도를 늦춘다 (G가 Generator Adversarial Loss로부터 배우는 것에 비해서)
    return 0.5 * (tf.math.reduce_mean(tf.math.squared_difference(real_D_out, tf.ones_like(real_D_out))) +
                  tf.math.reduce_mean(tf.math.squared_difference(fake_D_out, tf.zeros_like(fake_D_out))))
```

```Python
def generator_adversarial_loss_function(fake_D_out):
    '''
    LSGAN loss

    <params>
        fake_D_out : 가짜 이미지가 주어졌을 때, discirminator의 결과값
    '''

    return tf.math.reduce_mean(tf.math.squared_difference(fake_D_out, tf.ones_like(fake_D_out)))
```

```Python
def generator_L1_loss_function(real_images, fake_images):
    '''
    L1 loss

    <params>
        real_images : 실제 이미지
        fake_images : generator에 의해서 생성된 이미지
    '''

    return tf.math.reduce_mean(tf.math.abs(real_images - fake_images))
```

```Python
# discriminator loss 계산
discriminator_loss = discriminator_loss_function(real_D_out, fake_D_out)

# generator loss 계산
generator_adversarial_loss = generator_adversarial_loss_function(fake_D_out)
generator_L1_loss = generator_L1_loss_function(real_image, fake_image)

generator_loss = generator_adversarial_loss + LAMBDA * generator_L1_loss
```
- L1 loss를 구하는 데는 discriminator를 이용하지 않으므로
    - discriminator_loss = LSGAN_discriminator_loss
    - generator_loss = LSGAN_generator_loss + lambda * L1_loss
        - lambda는 10을 사용했습니다

<br>

- 실제 학습은 discriminator의 파라미터를 discriminator_loss를 낮추는 방향으로 한번 업데이트 하고, 그리고 그 다음에 generator의 파라미터를 generator_loss를 낮추는 방향으로 한번 업데이트 하는 과정을 반복합니다
- 두 모델의 파라미터를 동시에 학습하지 않는 것이 중요합니다 (따로따로 번갈아가며 해야합니다)
    - 그 이유는 generator(잘 만들어내자)의 역할과 discriminator(잘 구별하자)의 역할이 어찌 보자면 서로 상반되는데, 두개의 parameter를 동시에 학습한다면 서로 절충, 타협하는 방향으로 학습할 수 있기 때문입니다
    - discriminator_loss를 낮춘다 함은 LSGAN_discriminator_loss를 낮추는 방향, 즉 가짜 이미지는 0으로 예측하려고 하고, 진짜 이미지는 1로 예측하려고 하는, 이미지를 잘 구별하려고 학습하는 과정입니다
    - 그리고 그 뒤에 generator_loss를 낮춘다 함은 LSGAN_generator_loss + lambda * L1_loss를 낮추는 방향인데
        - LSGAN_generator_loss를 다시 살펴보면 G(x), 즉 가짜이미지가 discriminator에게 1(진짜처럼 보이도록)에 가깝도록 학습합니다
            
            ![lsgan generator loss](./docs%20images/lsgan%20generator%20loss.png)
            
        - 궁극적으로 D는 사용하지 않고 좋은 G를 얻어내는 것이 목적인데, D가 왜 필요하냐?의 대답을 여기서 할 수 있다고 생각합니다
        - D는 이전에 가짜(G(x))와 진짜(y)를 구별하려고 학습했기 때문에 (비록 한번의 step이지만, 조금이라도, 계속 누적 된다면) 가짜와 진짜가 어떤 부분에서 다른지 어느 정도 알게 됩니다
        - 그래서 어떻게 구별할 수 있었는지 이 정보를 G의 파라미터에 전달(D는 이때 학습하지 않습니다)해서 G가 더 잘 만들게 되는 거라고 이해하면 좋을거 같습니다
            - 여기서 전달은 물론 back propagation을 통해 이루어집니다
    - 이런 식으로 D는 잘 구별하려고 한번 배우고, 그 다음에 어떻게 구별했는지 그 정보를 G에도 전달해서 G는 더 잘 만들게 되고, 다시 또 D는 이걸 실제와 구별해보려고 노력하고, 다시 G에게 어떻게 구별했는지 정보를 전달하고 이런 식으로 GAN은 학습하는 것입니다

<br>

## 사용한 데이터

### Cartoon set
- 출처 : [구글의 cartoon set](https://google.github.io/cartoonset/)
- 데이터 수 : 9996 (원본 10만 개 중에서 일부를 추출하여 수행)
- batch 사이즈 : 4
- 학습시킨 epoch 수 : 28
- 특이사항
  - 성능 개선을 위해서 color 정보를 condition으로 추가해 보기도 하고, 데이터의 수를 늘려보기도 하였으나(10만 개, 원본 데이터 전부), 사용자가 그린 edge에 대한 변환 성능에 이렇다 할 개선점이 보이지 않음 (학습 데이터는 굉장히 잘 변환)
- 예시 결과
![cartoon set 예시 결과 이미지](./docs%20images/cartoon_set.png)

<br>

### Panda
- 출처 : [Kaggle, Panda or Bear Image Classification](https://www.kaggle.com/datasets/mattop/panda-or-bear-image-classification)
- 데이터 수 : 300 (곰 데이터는 제외하고, 판다 데이터만 사용)
- batch 사이즈 : 1
- 학습 시킨 epoch 수 : 180
- 예시 결과
![panda 예시 결과 이미지](./docs%20images/panda.png)

### Car
- 출처 : [DVM car dataset](https://deepvisualmarketing.github.io/)
- 데이터 수 : 11476 (DVM car dataset에서 세단 형의 bmw series 5 & 7만 추출)
- batch 사이즈 : 4
- 학습 시킨 epoch 수 : 19
- 예시 결과
![car 예시 결과 이미지](./docs%20images/bmw.png)

### Handbags
- 출처 : [저자들이 사용했던 데이터셋](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)
- 데이터 수 : 138567
- batch 사이즈 : 4
- 학습 시킨 epoch 수 : 5
- 예시 결과
![handbag 예시 결과 이미지](./docs%20images/handbag.png)

### Shoes
- 출처 : [저자들이 사용했던 데이터셋](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)
- 데이터 수 : 49825
- batch 사이즈 : 4
- 학습 시킨 epoch 수 : 25
- 예시 결과
![show 예시 결과 이미지](./docs%20images/shoe.png)

### Maplestory Characters
- 출처 : [Kaggle, maplestory_characters_hd](https://www.kaggle.com/datasets/irotem98/maplestory-characters-hd)
- 데이터 수 : 69372
- batch 사이즈 : 4
- 학습 시킨 epoch 수 : 14
- 예시 결과
![maplestory character 예시 결과 이미지](./docs%20images/maple_character.png)

### Gemstone
- 출처 : [Kaggle, Gemstones Images](https://www.kaggle.com/datasets/lsind18/gemstones-images)
- 데이터 수 : 3219
- batch 사이즈 : 4
- 학습 시킨 epoch 수 : 대략 36
- 예시 결과
![gemstone 예시 결과 이미지](./docs%20images/gemstone%20result.jpg)

### Space
- 출처 : [Kaggle, Cosmos Images](https://www.kaggle.com/datasets/kimbosoek/cosmos-images)
- 데이터 수 : 4649
- batch 사이즈 : 4
- 학습 시킨 epoch 수 : 40
- 예시 결과
![space 예시 결과 이미지](./docs%20images/space%20result.jpg)