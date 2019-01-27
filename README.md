# Custom-CNN-Based-Classifier-in-PyTorch

본 튜토리얼에서는 **PyTorch** 를 사용하여 **Image Classifier** 를 만들어보도록 하겠습니다.

본 튜토리얼을 통해 다음 방법들을 배울 수 있습니다.

* **CNN(Convolutional Neural Network)** 기반의 **Image Classifier** 모델 설계 방법
* 기존의 데이터셋(MNIST, CIFAR-10 등)이 아닌 **Custom Dataset**(개인이 수집한 데이터셋)을 처리하기 위한 **PyTorch** 의 **Dataset** 및 **DataLoader** 사용 방법

**본 튜토리얼에서는 PyTorch 의 Dataset 및 DataLoader 에 능숙해지기 위하여 PyTorch 의 ImageFolder 를 사용하지 않습니다.**

## Setup

### Requirements

본 튜토리얼에 앞서 다음의 라이브러리들을 필요로 합니다.
* PyTorch
* PIL

추가적으로 NVIDIA GPU 를 사용하신다면 CUDA 와 CUDNN 을 설치하는 것을 권장합니다.

### Custom Dataset

본 튜토리얼에서는 만화 **"쿠로코의 농구"** 의 등장 인물인 **"쿠로코"** 와 **"카가미"** 를 분류해보겠습니다.

**Kuroko's basketball**

![60136-kuroko_no_basket-blue-basketball](https://user-images.githubusercontent.com/35001605/51658134-693e2700-1fea-11e9-8045-b2d49231246f.jpg)

**Kuroko**

![65d739597a4f1bfd7085615cadf2c38e1367394101_full](https://user-images.githubusercontent.com/35001605/51658137-6c391780-1fea-11e9-9493-313b93f6166b.png)

**Kagami**

![kurokonobaske_a03](https://user-images.githubusercontent.com/35001605/51658140-6e9b7180-1fea-11e9-9804-9ff222d3d079.jpg)

#### Examples

정면 얼굴이 나온 사진을 위주로 쿠로코 60장, 카가미 60 장 수집

![example_dataset3](https://user-images.githubusercontent.com/35001605/51650040-12285a00-1fca-11e9-95d1-189352ef2d58.PNG)

![example_dataset2](https://user-images.githubusercontent.com/35001605/51650039-105e9680-1fca-11e9-89be-868234ae3241.PNG)

#### Download

[다운로드 ](https://drive.google.com/open?id=1dQePxrd9xdtvLr9E-jiUb-TdyWG1EFlJ)

데이터셋은 학습을 위한 train 폴더와 성능 평가를 위한 test 폴더로 나누어져 있습니다.

train 폴더와 test 폴더 내부에는 분류하고자 하는 클래스별로 하위 폴더가 존재하며 각 하위 폴더에는 해당 클래스에 속하는 이미지들이 저장되어 있습니다.

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

### Structure of Directory

본 튜토리얼 에서는 **Custom Dataset**에 대한 처리를 보다 쉽게 하기 위하여 다음과 같이 main.py 파일(코드를 작성할 파일)과 **Custom Dataset**이 동일한 경로에 있음을 가정하겠습니다. 

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
      
main.py
```
## Data Loading and Processing

PyTorch 에는 데이터셋에 대한 처리를 용이하게 하기 위하여 **Dataset** 과 **DataLoader** 클래스를 제공합니다.

**Dataset** 클래스는 torch.utils.data.Dataset 에 정의된 추상 클래스(Abstract class) 로써 사용자는 **Custom Dataset** 을 읽기 위하여 **Dataset** 클래스를 상속받는 클래스를 작성해야 합니다.

**DataLoader**는 **Dataset** 클래스를 상속받는 클래스에 정의된 작업에 따라 데이터를 읽어오며, 이때 설정에 따라 원하는 **배치(Batch) 크기**로 데이터를 읽어올 수 있고 병렬 처리 설정, 데이터 셔플(Shuffle) 등의 작업을 설정할 수 있습니다.

본 튜토리얼에서는 **Dataset** 클래스를 상속받는 클래스를 **Dataset** 클래스라 칭하도록 하겠습니다.

### Dataset Class & DataLoader Class

일반적인 **Dataset** 클래스의 형태는 아래와 같습니다.

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
**Custom Dataset** 을 읽기 위하여 다음의 3가지 함수를 정의해야 합니다.

* `__init__()` 

* `__getitem__()`

* `__len__()`

`__init__()` 함수는 클래스 생성자로써 데이터에 대한 Transform(데이터 형 변환, Augmentation 등)을 설정하고 데이터를 읽기 위한 기초적인 초기화 작업들을 수행하도록 정의합니다.

`__getitem__()` 함수는 **Custom Dataset** 에 존재하는 데이터를 읽고 반환하는 함수입니다. 따라서 본인이 어떤 작업을 수행하는지에 따라 반환하는 값들이 달라질 수 있습니다. 본 튜토리얼에서 구현할 작업은 **Image Classifier** 이므로 `__getitem__()` 함수는 이미지와 해당 이미지가 어떤 클래스에 속하는지에 대한 값을 반환할 것입니다.

주의할 점은 `__getitem__()` 을 통해 반환되는 값이 PyTorch 에서 처리 가능한 데이터 타입(tensor, numpy array etc.)이 아닐 경우, **DataLoader** 를 통해 데이터를 읽을 때 다음과 같은 에러가 발생될 것입니다.

`TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.PngImagePlugin.PngImageFile'>`

`__len__()` 함수는 데이터셋의 크기를 반환하는 함수입니다. **Image Classifier** 로 예를 들면, 우리가 가진 이미지의 갯수가 곧 데이터셋의 크기 입니다. 즉 50장을 가지고 있다면 `__len__()` 함수는 50 을 반환해야 합니다.

### Programming

#### modules
```python
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
```
#### Declaration & Definition classes
```python
class CustomImageDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length
```

**Image Classifer**를 위한 **Dataset** 클래스는 위와 같이 구현되며 실제 객체 생성시에는 아래와 같이 사용됩니다.


```python
custom_dataset_train = CustomImageDataset('./data/train',transforms=transfrom_train)
custom_dataset_test = CustomImageDataset('./data/test',transforms=transfrom_test)
```

CustomImageDataset 클래스의 내부를 살펴보시면 생성자(`__init__`)를 통하여 이미지들이 저장된 경로()를 입력받게 됩니다.

```python
def __init__(self, data_set_path, transforms=None):
```

