
# Pytorch에서 학습한 모델을 Tensorflow 에서 써보자!

## 목적
현재 efficientnet-pytorch에서 학습된 모델을 안드로이드 폰에서 사용하고자 한다.
pytorch는 개발이나 연구 목적으로는 좋으나 배포를 할 때는 효율적이지 않다. 그래서 pytorch 모델을 tflite 모델로 변환해 안드로이드에 서 사용하고자 한다.
과정은 크게 3단계로 이뤄진다.
1. PyTorch를 ONNX로 변환
2. ONNX를 TensorFlow FreezeGraph로 변환
3. Tensorflow FreezeGraph를 Tensorflow Lite로 변환

---
## 1. PyTorch 모델을 ONNX 모델로 저장하기
[공식문서](https://pytorch.org/docs/master/onnx.html, "공식문서")

### 필수 패키지 설치

#### 1. 가상환경 세팅
```
conda create -n 'env_name' python=3.7
```

#### 2. 패키지 설치
```
pip install efficientnet_pytorch onnx
pip install torch torchvision
#before installing tensorflow, check your cuda and cudnn version
pip install tensorflow-gpu
# check compatable addon version 
#https://github.com/tensorflow/addons#python-op-compatibility-matrix
pip install tensorflow-addons
```
제대로 설치되었는지 확인하기 위해 다음 커맨드라인을 실행

```
python
import tensorflow as tf
import torch
import onnx
from onnx_tf.backend import prepare
```
## 모델 로드 및 컨버트
```

```

## Problems
**P1.** On cuda 10.0 (compatable tensorflow-gpu version 2.0), not support swish activation function.  
**S1-1.**
use ```.set_swish(memory_efficient=False)```
``` 
model = Efficient.from_name(model_name='eficientnet-b0')
model.set_swish(memory_efficient=False)
torch.onnx.export(model, torch.rand(10, 3, 224, 224), "EfficientNet-B0.onnx")
```

**S1-2.**
Update to cuda 10.2
---



## Reference
[1] https://towardsdatascience.com/converting-a-simple-deep-learning-model-from-pytorch-to-tensorflow-b6b353351f5d
[2] https://colab.research.google.com/drive/1MwFVErmqU9Z6cTDWLoTvLgrAEBRZUEsA#forceEdit=true&sandboxMode=true&scrollTo=ZCYajzak-LUK
[3] https://pytorch.org/docs/master/onnx.html
[4] https://github.com/onnx/tutorials

