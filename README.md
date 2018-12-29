# Training-your-won-data-set-into-the-CNN-model-in-TF

VGG, Inception, Resnet 말고 !

MNIST, CIFAR - 10 말고 !

**내가 만든 CNN 모델**과 **직접 수집한 데이터 셋**을 통해 **이미지 분류 작업**을 수행해보고싶다.

## 뭐 부터 해야되지?

### 1. 어떤 이미지를 분류할 것인가??

만화 **"쿠로코의 농구"** 의 등장 인물인 **"쿠로코"** 와 **"카가미"** 를 분류해보자.


#### 쿠로코의 농구
![60136-kuroko_no_basket-blue-basketball](https://user-images.githubusercontent.com/35001605/50537882-8c231a00-0ba9-11e9-8da4-a438b7342c03.jpg)

#### 쿠로코
![65d739597a4f1bfd7085615cadf2c38e1367394101_full](https://user-images.githubusercontent.com/35001605/50537884-8fb6a100-0ba9-11e9-9d3b-ae820249f4c8.png)

#### 카가미
![kurokonobaske_a03](https://user-images.githubusercontent.com/35001605/50537883-8e857400-0ba9-11e9-9982-adedbc1e16f0.jpg)

### 2. 우선 데이터 수집부터!

정면 얼굴이 나온 사진을 위주로 쿠로코 60장, 카가미 60 장 수집

#### 데이터 셋 예시

![example_dataset](https://user-images.githubusercontent.com/35001605/50537659-f46ffc80-0ba5-11e9-9441-6f7e988447a6.PNG)

### 3. 파이썬 + 텐서플로 코딩 시작! (부제: 천리길도 한 걸음부터)

### 3.1 파이썬과 텐서플로를 이용하여 이미지 파
