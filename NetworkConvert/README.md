
# Pytorch에서 학습한 모델을 Tensorflow 에서 써보자!

## 목적
현재 [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch, "Efficient-PyTorch")에서 학습된 모델을 안드로이드 폰에서 사용하고자 한다.
pytorch는 개발이나 연구 목적으로는 좋으나 배포를 할 때는 효율적이지 않다. 그래서 pytorch 모델을 tflite 모델로 변환해 안드로이드에서 사용하고자 한다.
과정은 크게 3단계로 이뤄진다.
1. PyTorch를 ONNX로 변환
2. ONNX를 TensorFlow FreezeGraph로 변환
3. Tensorflow FreezeGraph를 Tensorflow Lite로 변환

---
## 1. PyTorch 모델을 ONNX 모델로 저장하기
[공식문서](https://pytorch.org/docs/master/onnx.html, "공식문서")

### 필수 패키지 설치

#### 1. 가상환경 세팅 및 접속
```
conda create -n 'env_name' python=3.7
source activate 'env_name'
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
아무런 에러 메시지가 없으면 제대로 설치된 것이다.  

## 모델 로드 및 컨버트
※주의! https://github.com/lukemelas/EfficientNet-PyTorch#example-export-to-onnx 에서 나온 예제는 잘못되었다. 
Issue 중 ONNX can't export SwishImplementation의 [저자의 코멘트](https://github.com/lukemelas/EfficientNet-PyTorch/issues/91#issuecomment-542994572, "저자의 코멘트")를 보면 ```.set_swish(memory_efficient=False)```를 사용하라고 한다. 현재 example에서는 업데이트가 안된 상태이다.

```
import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_name(model_name='efficientnet-b0')
model.set_swish(memory_efficient=False)
torch.onnx.export(model, torch.rand(10,3,240,240), "EfficientNet-B0.onnx")
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

## Reference
[1] https://towardsdatascience.com/converting-a-simple-deep-learning-model-from-pytorch-to-tensorflow-b6b353351f5d  
[2] https://colab.research.google.com/drive/1MwFVErmqU9Z6cTDWLoTvLgrAEBRZUEsA#forceEdit=true&sandboxMode=true&scrollTo=ZCYajzak-LUK  
[3] https://pytorch.org/docs/master/onnx.html  
[4] https://github.com/onnx/tutorials  

