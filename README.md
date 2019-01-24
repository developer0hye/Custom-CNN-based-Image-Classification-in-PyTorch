# Custom-CNN-Based-Classifier-in-PyTorch

본 튜토리얼에서는 **PyTorch** 를 사용하여 **Image Classifier** 를 만들어보도록 하겠습니다.

본 튜토리얼을 통해 다음 방법들을 배울 수 있습니다.

* **CNN(Convolutional Neural Network)** 기반의 **Image Classifer** 모델 설계 방법
* 기존의 데이터셋(MNIST, CIFAR-10 등)이 아닌 Custom Dataset(개인이 수집한 데이터셋)을 처리하기 위한 **PyTorch** 의 **Dataset** 및 **DataLoader** 사용 방법

**본 튜토리얼에서는 PyTorch 의 Dataset 및 DataLoader 에 능숙해지기 위하여 PyTorch 의 ImageFolder 를 사용하지 않습니다.**

## Setup

### Prerequisite

본 튜토리얼에 앞서 다음의 라이브러리들을 필요로 합니다.
* PyTorch
* PIL

추가적으로 NVIDIA GPU 를 사용하신다면 CUDA 와 CUDNN 을 설치하는 것을 권장합니다.

### Custom Dataset

본 튜토리얼에서는 만화 **"쿠로코의 농구"** 의 등장 인물인 **"쿠로코"** 와 **"카가미"** 를 분류해보겠습니다.

![60136-kuroko_no_basket-blue-basketball](https://user-images.githubusercontent.com/35001605/50537882-8c231a00-0ba9-11e9-8da4-a438b7342c03.jpg)

#### Kuroko

![65d739597a4f1bfd7085615cadf2c38e1367394101_full](https://user-images.githubusercontent.com/35001605/50537884-8fb6a100-0ba9-11e9-9d3b-ae820249f4c8.png)

#### Kagami

![kurokonobaske_a03](https://user-images.githubusercontent.com/35001605/50537883-8e857400-0ba9-11e9-9982-adedbc1e16f0.jpg)

#### Examples

정면 얼굴이 나온 사진을 위주로 쿠로코 60장, 카가미 60 장 수집

![example_dataset3](https://user-images.githubusercontent.com/35001605/51650040-12285a00-1fca-11e9-95d1-189352ef2d58.PNG)

![example_dataset2](https://user-images.githubusercontent.com/35001605/51650039-105e9680-1fca-11e9-89be-868234ae3241.PNG)

#### Download

[다운로드 ](https://drive.google.com/open?id=1dQePxrd9xdtvLr9E-jiUb-TdyWG1EFlJ)

데이터셋은 학습을 위한 train 폴더와 성능 평가를 위한 test 폴더로 나누어져 있습니다.

train 폴더와 test 폴더 내부에는 분류하고자 하는 클래스별로 하위 폴더가 존재하며 각 하위 폴더에는 해당 클래스에 맞는 이미지들이 저장되어 있습니다.

```python
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

## Data pre-processing and data augmentation

PyTorch 에는 데이터셋에 대한 처리를 용이하게 하기 위하여 **Dataset** 과 **DataLoader** 클래스를 제공합니다.

**Dataset** 클래스는 torch.utils.data.Dataset 에 정의된 추상 클래스(Abstract class) 로써 사용자는 본인이 소유한 데이터를 읽어들이기 위하여 **Dataset** 클래스를 상속받는 클래스를 작성해야 합니다.

**DataLoader**는 **Dataset** 클래스를 상속받는 클래스에 정의된 작업에 따라 데이터를 읽어오며, 이때 설정에 따라 원하는 **배치(Batch) 크기**로 데이터를 읽어올 수 있고 병렬 처리 설정, 데이터 셔플(Shuffle) 등의 작업을 설정할 수 있습니다.

본 튜토리얼에서는 **Dataset** 클래스를 상속받는 클래스를 **Dataset** 클래스라 칭하도록 하겠습니다.

### Dataset Class

**Dataset** 클래스의 형태는 아래와 같습니다.

```python
from torch.utils.data.dataset import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, ...):
        # stuff
        
    def __getitem__(self, index):
        # stuff
        return (img, label)

    def __len__(self):
        return count # of how many examples(images?) you have
```

__init__ 함수는 



