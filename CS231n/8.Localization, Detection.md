# Computer Vision Tasks

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/30b6bccc-633a-4d16-b70b-01da3ad1ddc2" height="80%" width="80%"></p>

Detection 부터는 여러 object가 있고 각각의 object를 찾아낸다. Segmentation은 각 object를 형상대로 구분하게 된다. 

## Classification + Localization 
Classification같은 경우 C개의 class가 있을 때 input으로 이미지를 받고 output으로 class label을 준다. 평가 지표는 정확도가 된다. 
Localization은 input으로 이미지가 들어오면 output은 label이 아니라 box이다. x, y 좌표 기준 width, height 이렇게 4개의 값(x, y, w, h)을 output으로 준다. 평가 지표는 Intersection Over Union(IOU)로 box와 object가 겹치는 비율이다. 

따라서 결국 Classification과 Localization을 통합하면 label과 box 위치 정보 모두 얻게 되는 것이다. 
Localization에는 크게 2가지 방법이 있다. 

### Idea #1 : Localization as Regression

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1a9b23b1-7ed2-412f-9308-7e32fda09763" height="80%" width="80%"></p>

첫 번째 방법은 Localization을 Regression으로 간주하는 것이다. 이 방법은 1개 또는 n개의 object를 Localization할 때 굳이 detection을 사용하지 않아도 된다. 방법은 위 사진처럼 이미지를 넣고 결과로 4개 수로 이루어진 좌표와 실제 정답 box 좌표를 비교해서 loss를 구하게 된다. 그리고 이 값을 역전파에서 update해서 학습하는 방식이다. 단계별로 보면 아래와 같다. 

#### Step 1 : Train (or download) a classification model (AlexNet, VGG, GoogleNet)  

#### Step 2 : Attach new fully-connected "regression head" to the network

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/8f0acc9d-cf79-4bbd-a59f-53de9d59b680" height="70%" width="70%"></p>

위 사진 처럼 결과물이 box의 좌표로 오도록 하는 regression head가 추가되었다. 

#### Step 3 : Train the regression head only with SGD and L2 loss
3 단계에서는 추가된 regression head부분에 대해서만 학습을 시켜준다. 

#### Step 4 : At test time use both heads 
마지막 4 단계에서는 Classification head와 regression head 모두를 이용해서 결과를 산출한다. 


### Per-class vs class agnostic regression
regression head에는 2가지 방법이 있다. Per-class의 경우는 class에 특화된 방법이고, class agnostic은 각각의 class와 무관하게 범용적인 방법이라 할 수 있다. class agnostic의 경우 class에 특화되지 않기 때문에 결과로 그냥 하나의 box 4개의 숫자만 나오게 되고, class에 특화된 class specific한 방법은 각 클래스당 한 box로 결과가 나와 총 `4 X C` class의 수에 4를 곱한 숫자가 나오게 된다.
이 두가지 방법은 loss를 구하는데 약간의 차이가 있을 뿐 유사한 방법이다. 

regression head를 적용해야하는 위치 또한 2가지 방법이 있는데 이 2가지 방법 모두 통한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/8d51fb1d-a6d6-4879-a4b1-881b67afd8da" height="80%" width="80%"></p>


첫 번째 방법으로는 마지막 conv layer 뒤에다 붙여주는 방식으로 Overfeat이나 VGG를 사용하는 경우 이 방법을 사용하고, FC layer 뒷 단에 붙여주는 경우도 있다. DeepPose나 R-CNN의 경우 이 방법을 사용하고 어떤 경우든 다 잘 동작한다.  

결론적으로 object의 수가 정해진 경우에는 regression만으로도 잘 동작하게 된다. 이럴 때는 굳이 detection을 쓰지 않아도 된다. 


### Idea #2 : Sliding Window 
Sliding Window는 Regression과 비슷하게 여전히 Classification head와 regression head 두 개로 분리해서 진행하지만, 이미지의 한 군데가 아닌 여러군데를 여러번 돌리고, 최종적으로 합쳐주는 기법이다. 
또한 편의성을 위해 fully-connected layer를 convolutional layer로 변환해서 연산을 진행한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/65d10336-72b3-4392-b7b9-a551e8f7694c" height="80%" width="80%"></p>

위 사진에서 검은색 박스는 Sliding Window가 되고 이 Sliding Window를 통해 위 사진 기준으로는 왼쪽 위를 먼저 보게 된다. 그 다음 regression head에 의해 만들어진 빨간색 박스와 함께 Classification score에 고양이로 분류되는 정도를 계산한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4e6befdd-3606-43c4-a0b6-1e8aa9115c4d" height="60%" width="60%"></p>

이런식으로 위 사진과 같이 Sliding Window를 계속 움직여 주면서 score를 계산해나간다. 

실제 사용할 때는 수백게의 Sliding Window를 사용하게 될 수도 있는데 Sliding Window를 사용한 연산이 복잡해서 이를 효율적으로 하기 위한 방법이 FC를 convolutional layer로 바꾸는 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/7c70144c-3e0b-4697-b7a6-02ae1979107d" height="80%" width="80%"></p>

이렇게 conv layer를 사용하는데, 기존에 FC에서 4096 개의 원소로 구성된 벡터를 벡터가 아닌 또 하나의 convolutional feature map(입력으로부터 커널을 사용하여 합성곱 연산을 통해 나온 결과)으로 생각하는 것이다.  
따라서 1 X 1 차원을 추가하여 conv layer로 만들어버린다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/accc489d-8a2a-4958-b7cd-e653d2303c97" height="80%" width="80%"></p>


## Object Detection 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/9095b001-1cd8-4641-bf64-e0a12a420db6" height="80%" width="80%"></p>

Object Detection 한 이미지 내에 불특정 개수의 여러 Object를 인식하는 기술이다. 처음 나온 아이디어는 기존 regression을 활용하는 것인데 문제는 Object Detection은 불특정 개수이기 때문에 고양이가 한 마리가 있을 수도 있고 여러마리가 있을 수도 있다. 따라서 이미지에 따라 도출되는 number의 수가 달라지기 때문에 regression은 적합하지가 않다. 그럼에도 불구하고 나중에 나오는 YOLO라는 모델은 regression을 사용해서 Detection을 하게 된다. 

결국 일반적으로는 regression이 적합하지 않기 때문에 Detection을 Classification으로 간주하는 접근 방법이 나오게 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/ffedd649-9a07-46b6-8e9d-e57e69318f25" height="160" width="400"><img src="https://github.com/em-1001/AI/assets/80628552/520c3fae-aca2-419b-ae7e-0c3d361d3129" height="160" width="400"></p>

그래서 위 사진과 같이 박스의 위치에 따라 고양이 인지 아님 강아지인지 분류를 하게 된다. 
이렇게 Classification으로 간주했을 때의 문제는 다양한 크기의 윈도우를 가지고 다양한 이미지의 전 영역에 해야하기 때문에 테스트의 수가 매우 많다는 것이다. 하지만 conv net 같은 무거운 classifier가 아닌 이상 그냥 시도하는 것이 해결책이었는데 실제로 이 방법이 잘 통했었다. 

대표적으로 HOG같은 경우 매우 빠른 선형 분류로 이미지를 최대한 다양한 해상도에서 돌려서 잘 분류를 하였다. 
HOG에 대한 후속 연구로는 DPM이 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3895d734-a0da-43c1-9363-3efdacc9a950" height="80%" width="80%"></p>

DPM은 여전히 HOG를 기반으로 동작을 하고 특징이 특정 부분부분 사람의 경우 머리 다리 등 부분에 대한 템플릿을 가지고 있어고 이 템플릿들은 각각이 변형된 형태까지 다 기지고 있어서 당시 수준으로는 잘 동작을 하였다. 하지만 나중에 논문으로 나온 것이 이 방법이 결국 이름만 다를 뿐 CNN을 사용하는 것과 유사하다는 것이다. 

이렇듯 Classification으로 간주했을 때 발생하는 두 번째 문제는 CNN과 같은 무거은 classifier를 사용해야 한다는 것이다. 
이런 경우 모든 영역과 scale을 다 보는 것은 연산적으로 매우 무겁다. 
이 문제의 해결책은 모든 영역을 다 보기보다 의심되는 영역만 조사하는 것이다. 그래서 의심되는 영역을 추천한다고 해서 Region Proposal 방식이라고 한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/70e4124a-fe18-4a08-a749-4a4233ca544c" height="80%" width="80%"></p>

그래서 Region Proposal은 결국 object를 포함하고 있을 거 같은 blobby한 부분 즉, 뭉쳐있고, 유사한 색이나 텍스쳐를 가지고 있는 부분을 말하고, 클래스와 무관한 blobby한 부분만 찾는 detector가 된다. 
이미지를 보면 강아지이던 고양이던 class에 신경쓰지 않고 코, 눈 등을 잡아내는 것을 볼 수 있다. 장점은 매우 빠르다는 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/5dbb8711-83d7-41cf-9683-9f169501fc9a" height="80%" width="80%"></p>

Region Proposal의 방식은 여러가지가 있는데 대표적으로 Selective Search는 이미지의 픽셀에서 시작해서 색상과 텍스쳐가 유사한 픽셀들끼리 묶게되고 이것들을 알고리즘을 활용해서 merge하여 큰 blob을 만들어 낸다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/ef89942c-654b-459a-8b36-02cc7e4f5107" height="80%" width="80%"></p>

Selective Search이외에도 수많은 방법이 있다. 추천되는 방식으로는 EdgeBoxes라고 한다. 

### R-CNN

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3b888c04-9eee-4f2d-8457-59467634f746" height="70%" width="70%"></p>

결론적으로 지금까지 본 Region Proposal과 CNN을 결합한 것이 R-CNN이다. 
사진을 보면 input image를 받아서 관심있는 지역(box) ROI를 뽑아낸다. 수는 대략 2천개를 뽑아내고, 이렇게 뽑은 각각의 box들은 각기 다른 크기와 위치에 해당된다. 그 다음 이 box들을 Warping하는 작업을 거친다. 일반적으로 CNN에 들어가는 정사각형 크기로 warp를 시켜주고 이렇게 만들어진 각각의 ROI를 CNN으로 학습시킨다. 위 사진에는 Classification head와 regression head로 두 개의 head가 존재하는 것을 볼 수 있는데 Classification head을 이용해서는 SVM을 통해 분류를 하고 regression head에서는 bounding box를 추출해 낸다. 


#### Step 1 : Train (or Download) a classification model for ImageNet(AlexNet)

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e089f991-ae42-489d-beb5-65e2f1d23a8a" height="60%" width="60%"></p>

#### Step 2 : Fine-tune model for detection
- Instead of 1000 ImageNet classes, want 20 object classes + background
- Throw away final fully-connected layer, reinitialize from scratch
- Keep training model using positive / negative regions from detection images

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/f4f50758-1107-4602-8de7-492087cb505e" height="60%" width="60%"></p>

우리가 가져야 하는 클래스의 수는 21개(20개의 object 클래스 + 배경)이다. 원래라면 사진의 파란색 레이어가 4096 X 1000의 크기를 가졌는데, 마지막 FC layer를 제거하고 4096 X 21이 되도록 변형하였다. 

#### Step 3 : Extract features 
- Extract region proposals for all images
- For each region : warp to CNN input size, run forward CNN, save pool5 feature to disk
- Have a big hard drive : feature are ~200GB for PASCAL dataset

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/ca165438-e130-46ca-97a5-8b8fe4055c7c" height="60%" width="60%"></p>

이제 feature를 추출하는데 우선 region proposals들을 추출하고, 각각의 region에 대해 CNN에 들아갈 사이즈에 맞게 Warp를 시켜준다. 그리고 CNN을 돌린 뒤 pool5(AlexNet의 5번재 pooling layer)의 feature를 디스크에 저장한다. 

#### Step 4 : Train one binary SVM per class to classify region feature 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4eaad553-9c13-4b0e-88da-34c778ce8784" height="60%" width="60%"></p>

클래스당 하나의 binary SVM을 이용해 region feature들을 분류한다. binary SVM이기 때문에 고양이 이면 고양이 이다/아니다 이런식으로 모든 클래스에 대해 반복을 한다. 
위 사진은 강아지의 경우 binary SVM에서 어떤 사진이 negative이고 어떤 사진이 positive인지 분류한 예시이다. 

#### Step 5 : For each class, train a linear regression model to map from cached features to offsets to GT boxes to make up for " slightly wrong" proposals

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/534eb2e3-0540-4e36-a3d7-384efa46e43b" height="60%" width="60%"></p>

bbox regression이라고도 하며, region proposals이 항상 정확한 것은 아니기 때문에, cache해놓은 feature로 부터의 regression을 이용해서 region proposals의 정확도를 높여주는 역할을 한다. 
이 과정을 통해 mAP(mean Average Precision)이 3~4% 높아지는 것을 확인할 수 있다. 

Detection 관련 dataset으로는 PASCAL VOC, ImageNet Detection, MS-COCO 등이 있다. 
최근엔 google에서 Open Image Dataset을 내기도 했다. 

### Detection Evaluation 
Detection을 Evaluation하는 방법으로는 앞서 언급한 mAP가 쓰인다. 기본적으로 각각의 클래스에 대해 average precision을 따로 구하고 이 값들을 평균을 내서 값을 구한다. 

### R-CNN Problems 
지금까지 R-CNN에 대해 살펴보았는데 R-CNN은 몇가지 단점을 가지고 있다. 
1. 테스트시 매우 느리다는 것이다. 각각의 region proposal에 대해 forward pass로 CNN을 돌려야 하기 때문에 매우 무겁고 느리다.
2. SVM과 regression이 오프라인으로 학습되기 때문에 바로바로 업데이트가 안된다.
3. 앞서 step에서 살펴보았듯이 다단계 training pipeline이 매우 복잡하다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/82411719-ea5d-4fbe-b93b-9d2c67f3f4ce" height="60%" width="60%"></p>

이런 약점들을 염두하고 나온 것이 Fast R-CNN이라는 것이다. Fast R-CNN의 아이디어는 CNN을 돌리는 것과 region을 추출하는 것의 순서를 바꾼것이다. 지금까지 살펴본 R-CNN은 region proposal을 먼저 추출하고 CNN을 돌렸는데 여기선 CNN을 먼저 돌린다. 

우선 이미지가 들어오면 CNN을 돌려서 고해상도의 feature map을 생성해내고, 이에 대해 region proposal을 적용해서 Regions of Interest(ROI)를 추출해낸다. 이렇게 추출한 ROI들을 ROI Pooling이라는 기법을 이용해 FC로 넘겨주고 FC에서는 classifier header와 regressor header로 전달해 준다. 

이렇게 하면 각각의 region들을 forward pass로 CNN을 돌려야 한다는 첫 번째 문제를 region proposal을 선택하기 전에 CNN을 돌림으로서 해결할 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/c523b28a-e7b2-43ed-a861-6c533be31565" height="60%" width="60%"></p>

앞선 사진은 test time일 때의 경우였고 이제 training time일 때의 경우를 살펴보면, 두 번째 단점인 SVM과 regression이 오프라인으로 학습되어 CNN이 즉각적으로 update 되지 않는다는 문제와 세 번째 문제인 training pipeline이 복잡하다는 문제를 전체 시스템을 end-to-end로 학습시킬 수 있다는 점으로 해결하였다. 

### ROI Pooling 


앞서 Fast R-CNN에서 언급된 ROI Pooling 에 대해 좀 더 상세히 설명해보면 고해상도의 input이 있고 region proposal이 되어있을 때, conv와  pooling을 거쳐서 high resolution의 features를 얻는다. 여기서 문제는 Fully-connected layer에서는 low resolution conv feature를 원한다는 것이다. 
그래서 이 상충을 해결하기 위해 ROI Pooling을 이용한다. 방법은 원본 이미지의 region proposal을 conv feature map으로 projection하고 그 영역을 grid로 나눈다. 그리고 이를 max pooling하여 결과적으로 ROI feature를 추출하게 된다. 이렇게 max pooling을 했을 때의 장점은 역전파에서 아무런 문제가 없다는 것이다. 아래 이미지는 위 과정을 단계별로 표현한 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/695dbccc-9bcd-4631-82cc-ef443c697c81" height="200" width="460"><img src="https://github.com/em-1001/AI/assets/80628552/21397475-e309-4739-ae19-8531ccc509a4" height="200" width="460"></p>  
<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/075de9be-3b22-409d-bd05-52386fb63638" height="200" width="460"><img src="https://github.com/em-1001/AI/assets/80628552/168174de-8d0e-4e86-9411-c644979477c8" height="200" width="460"></p>


<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/a574a8fe-fc70-4e89-ab45-bd59a3a6e944" height="60%" width="60%"></p>

그러나 이 Fast R-CNN에도 치명적인 문제가 있다. Fast R-CNN의 이미지당 test time경과 시간은 region proposal을 포함하지 않은 것이고, selective search와 같은 region proposal을 포함하게 되면, 무려 2초나 걸려서 실제로는 사용이 힘들어 진다. 
이에 대한 해결책은 앞서 CNN을 이용해 regression과 classification을 모두 한 것 처럼 region proposal에도 CNN을 적용하자는 아이디어에서 시작하여 Faster R-CNN이라는 모델이 생겨났다. 

### Faster R-CNN

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/c7b87039-01e0-4a32-b9ea-bd067490ef0c" height="40%" width="40%"></p>

지금까지는 region proposal을 외부에서 진행했다면, Faster R-CNN에서는 그럴 필요가 없어졌다. 그림에서 보다싶이 Region Proposal Network(RPN)라는 것을 만들어서 RPN을 제일 마지막에 있는 conv layer에 삽입한다. 이 RPN도 당연히 CNN이다. RPN이후의 과정(ROI pooling, header...)은 fast R-CNN과 동일하다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/26cc4161-bdc0-4cd1-bb2b-68edefc60168" height="40%" width="40%"></p>

RPN에 대해 좀 더 살펴보면, 전단계에서 CNN을 거쳐 나온 feature map에서 3X3의 sliding window를 통해 slide를 한다. 그렇게 하므로서 RPN을 생성하는 것인데 RPN은 object냐 아니냐를 분별해주고, object의 location을 regressing bbox로 구해준다. 
sliding window의 위치는 이미지에서 localization 관련 정보를 제공하게 되고, box regression은 sliding window을 고정해주는 역할을 한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/d73826a6-fcf7-4c6f-85cc-5601e7835d41" height="40%" width="40%"></p>

여기서 새로 나오는 개념이 N anchor Box라는 것인데, anchor라는 것은 sliding window내에 존재하는 local window라고 생각할 수 있고, 특정 위치의 sliding window 마다 위(사진에서는 9개) anchor 9개는 각기 다른 크기와 비율을 갖는다.
RPN에서는 각각의 anchor가 전경인지 배경인지(object인지 아닌지)를 분별한다음 전경인 경우 해당 anchor를 object bounding box에 fit하도록 stretch시켜준다. 
그렇게 해서 regression은 anchor box로 부터 얼마나 떨어져 있냐의 값을 알려주고 classification에서는 각각의 anchor가 object인지 확률을 제공한다. 

결국 각기 다른 크기와 비율을 갖는 anchor box들을 original image에 투영하게 된다. 이 때 feature map의 포인트에 대응하는 이미지 내의 point에 paste한다. 
앞서 fast R-CNN에서 이미지에서 feature map으로 투영한 것과는 반대인 것을 알 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/f48181fb-0730-4c54-a22f-c6e5f7988aaa" height="40%" width="40%"></p>

정리하면 위 사진과 같이 하나의 거대한 네트워크 안에서 총 4개의 loss를 갖게 된다. 
RPN에서는 object냐 아니냐와 anchor box로 부터의 거리 이렇게 2개의 loss가 나오고 , ROI Pooling에서는 해당 object가 어떤 class인지와 region proposal 상에서 correction을 위한 loss 이렇게 2개가 나오게 된다. 

결과적으로 2초 걸리던 것이 0.2로 매우 단축되는 성능향상을 보인다. 

### YOLO 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/095131bd-4066-4ddc-91e0-5c49d585cea4" height="40%" width="40%"></p>

앞서 Detection을 Regression을 간주하면 정해진 수에 대한 detection만 가능해서 적절치 않다고 했었는데, YOLO는 Detection을 Regression으로 간주한 모델이다. 
이미지를 일반적으로 7 X 7의 Grid로 나누게 되고 각각의 Grid에서 2가지를 예측하는데 4개의 좌표와 하나의 score인 confidence를 가지는 B개의 Box들 그리고, Class Score를 나타내는 C 개의 숫자들을 예측하게 하는 것이고, 이렇게 하면 이미지로부터의 regression이 $7 * 7 * (5 * B + C)$ 의 tensor가 된다.

YOLO는 굉장히 빠르지만 정확도에서 다소 떨어지는 면이 있다. 

# Object Detection Code links
#### R-CNN 
(Cafffe + MATLAB) : https://github.com/rbgirshick/rcnn  

#### Fast R-CNN
(Cafffe + MATLAB) : https://github.com/rbgirshick/fast-rcnn

#### Faster R-CNN
(Cafffe + Python) : https://github.com/rbgirshick/py-faster-rcnn

#### YOLO
https://pjreddie.com/darknet/yolo/  



# Reference
https://www.youtube.com/watch?v=y1dBz6QPxBc&list=PL1Kb3QTCLIVtyOuMgyVgT-OeW0PYXl3j5&index=7    
Object Detection Tutorial : https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection  
