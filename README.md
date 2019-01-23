# Custom-CNN-Based-Classifier-in-PyTorch

이번 튜토리얼에서는 **PyTorch** 를 사용하여 **Image Classifier** 를 만들어보도록 하겠습니다.

본 튜토리얼을 통해 다음 방법들을 배울 수 있습니다.

* **CNN(Convolutional Neural Network)** 기반의 **Image Classifer** 모델 설계 방법
* 기존의 데이터셋(MNIST, CIFAR-10 등)이 아닌 개인이 수집한 데이터셋을 처리하기 위한 **PyTorch** 의 **Dataset** 및 **DataLoader** 사용 방법

**본 튜토리얼에서는 PyTorch 의 Dataset 및 DataLoader 사용에 능숙해지기 위하여 PyTorch 의 ImageFolder 를 사용하지 않습니다.**

## Setup

### Prerequisite
본 튜토리얼에 앞서 다음의 라이브러리들을 필요로 합니다.
* PyTorch
* PIL

추가적으로 NVIDIA GPU 를 사용하신다면 CUDA 와 CUDNN 을 설치하는 것을 권장합니다.

### Custom Dataset

본 튜토리얼에서는 만화 **"쿠로코의 농구"** 의 등장 인물인 **"쿠로코"** 와 **"카가미"** 를 분류해보겠습니다.

#### Kuroko's Basketball

![60136-kuroko_no_basket-blue-basketball](https://user-images.githubusercontent.com/35001605/50537882-8c231a00-0ba9-11e9-8da4-a438b7342c03.jpg)

#### Kuroko

![65d739597a4f1bfd7085615cadf2c38e1367394101_full](https://user-images.githubusercontent.com/35001605/50537884-8fb6a100-0ba9-11e9-9d3b-ae820249f4c8.png)

#### Kagami

![kurokonobaske_a03](https://user-images.githubusercontent.com/35001605/50537883-8e857400-0ba9-11e9-9982-adedbc1e16f0.jpg)

#### Dwonload

정면 얼굴이 나온 사진을 위주로 쿠로코 60장, 카가미 60 장 수집

[다운로드 ](https://drive.google.com/open?id=1dQePxrd9xdtvLr9E-jiUb-TdyWG1EFlJ)

#### 데이터 셋 예시

```
data/
  train/
    kuroko/
      *.png
    kagami/
      *.png
  test/
    kuroko/
      *.png
    kagami/
      *.png
```

data/train/kagami

![example_dataset2](https://user-images.githubusercontent.com/35001605/51581546-d929b000-1f0b-11e9-96ff-bcf704913ec0.PNG)


### 3. 코딩 시작!

#### 3.1 프로젝트 생성
#### 3.1.1


