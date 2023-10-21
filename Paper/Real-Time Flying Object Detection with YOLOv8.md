# Background
## YOLOv1
YOLOv1이 사용하는 네트워크에 이미지를 통과시키면 결과로 SxS 그리드 셀의 클래스 확률 C와 예측된 바운딩 박스 B, 그리고 Confidence Score가 주어진다. 여기서 SxS로 나눈 그리드 셀 중 물체의 중앙과 가장 가까운 셀이 객체를 탐지하는 역할을 하게된다. 그리고 각 셀은 바운딩 박스 B와 분류한 클래스의 확률인 C를 예측한다. 

**바운딩 박스 B** 는 X, Y 좌표, 가로, 세로 크기 정보와 Confidence Score (Score)수치를 가지고 있다. Score는 B가 물체를 영역으로 잡고 있는지와 클래스를 잘 예측하였는지를 나타낸다. 본 논문에서는 Score를 간단하게 **Pr(Object) ∗ IOU** 로 정의하고 있는데, **Pr(Object)** 는 바운딩 박스 안에 물체가 존재할 확률이다. 만약 바운딩 박스가 배경만을 영역으로 잡고 있다면 Pr(Object)의 값이 0이므로 Score는 0이된다.

**클래스 확률 C** 는 그리드 셀 안에 있는 그림의 분류 확률을 나타낸다. 기호로는 **Pr(Class_i |Object)** 로 표현하며 B가 배경이 아닌 객체를 포함하는 경우의 각 클래스의 조건부 확률이다. B가 배경을 예측했다면 확률은 0이 된다. 최종적으로 클래스 조건부 확률 C와 각 바운딩 박스의 Confidence 예측 값을 곱하면 각 박스의 클래스별 Confidence Score 수치를 구할 수 있다.

$$Pr(Class_i |Object) * Pr(Object) * IOU = Pr(Class_i) * IOU$$

### YOLOv1 Loss Function
YOLOv1은 Training Network를 학습하기 위해 손실 함수를 설계하기 전 다음과 같은 원칙을 만들었다. 

1. 이미지를 분류하는 classifier 문제를 바운딩 박스를 만드는 regression문제로 생각한다.
2. 바운딩 박스를 잘 그렸는지 평가하는 Localization Error와 박스 안의 물체를 잘 분류했는지 평가하는 Classification Error의 패널티를 다르게 평가한다. 특히, 박스 안의 물체가 없는 경우에는 Confidence Score를 0으로 만들기 위해 Localization Error 에 더 높은 패널티를 부과한다.
3. 많은 바운딩 박스중에 IOU 수치가 가장 높게 생성된 바운딩 박스만 학습에 참여한다. 이는 바운딩 박스를 잘 만드는 셀은 더욱 학습을 잘하도록 높은 Confidence Score를 주고 나머지 셀은 바운딩 박스를 잘 만들지 못하더라도 나중에 Non-max suppression을 통해 최적화 하기 위함이다.

YOLO는 1번 원칙을 지키기 위해 Loss Function 에서 **Sum-Squared Error(SSD)** 를 이용한다. 그리고 2번 원칙을 만족하기 위해서 $λ_{coord}$ 와 $λ_{noobj}$ 두 개의 변수를 이용한다. 본 논문에서는 $λ_{coord} = 5, λ_{noobj} = 0.5$ 로 설정하였다. 아래는 YOLOv1의 Loss Function이다. 



$$λ_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_{i j} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]$$

$$+λ_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_{i j} \left[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$

$$+\sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{obj}_{i j} \left(C_i - \hat{C}_i\right)^2$$

$$+λ_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^B 𝟙^{noobj}_{i j} \left(C_i - \hat{C}_i\right)^2$$

$$+\sum_{i=0}^{S^2} 𝟙^{obj}_ {i} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$$  

$S$ : 그리드 셀의 크기를 의미한다. 행렬 이기 때문에 전체 그리드 셀의 개수는 S² 가 된다.      
$B$ : $S_i$ 셀의 바운딩 박스를 의미한다.    
$C$ : 각 그리드 셀이 구분한 클래스와 같다.   
$λ_{coord}$ : 5로 설정된 λ_coord 변수로서 Localization 에러에 5배 더 높은 패널티를 부여하기 위해서 사용한다.    
$𝟙^{obj}_ {i j}$ : i번째 셀의 j번 바운딩 박스만을 학습하겠다는 의미로 사용하지만 모든 셀에 대해서 바운딩 박스 학습이 일어나지 않고 각 객체마다 IOU가 가장 높은 바운딩 박스인 경우에만 패널티를 부과해서 학습을 더 잘하도록 유도한다.  
$λ_{noobj}$ : 해당 셀에 객체가 존재하지 않는 경우, 즉 배경인 경우에는 바운딩 박스 학습에 영향을 미치지 않도록 0.5의 가중치를 곱해주어서 패널티를 낮춘다.    
$𝟙^{noobj}_{i j}$ : i번째 셀과 j번째 바운딩 박스에 객체가 없는 경우에 수행 한다는 의미이다.

3,4 번째 항은 각각 bouding box가 객체를 포함할 때와 배경일 때의 confidence error를 계산하고 마지막 5번째 항은 bouding box와 관계없이 각 셀마다 클래스를 분류하기 위한 오차이다. 



# Real-Time Flying Object Detection with YOLOv8
본 논문은 현재 state-of-the-art인 YOLOv8을 이용한 비행체 탐지모델을 제안한다. 일반적으로 Real-time object detection은 object의 공간적 사이즈(spatial sizes), 종횡비(aspect ratios), 모델의 추론 속도(inference
speed), 그리고 noise 등의 변수로 어려움이 있었다. 비행체는 위치(location), 크기(scale), 회전(rotation), 궤도(trajectory)가 매우 빠르게 변하기 때문에 앞선 문제들은 비행체를 탐지하는데 더욱 부각된다. 그렇기에 비행체의 이러한 변수에 대해 thorough하고 빠른 추론속도를 갖는 모델이 중요했다. 

<img src="https://github.com/em-1001/AI/assets/80628552/7c8e5c53-3e12-46fa-813f-6698c1b06538" height="80%" width="80%">

본 논문에서는 dataset중 80%를 train, 20%을 validation으로 나누었다. 각 dataset의 이미지는 class number가 label되어있고, bounding box 가장자리의 좌표를 표시해놨다. 하나의 이미지에는 평균적으로 1.6개의 object가 있고, median image
ratio는 416x416이다. 이미지는 auto orientation으로 전처리 되었으며, augmentations은 적용하지 않았다. 

##  Model Choice and Evaluation
논문에서는 우선 YOLOv8의 small, medium, and large 버전에 대해 최적의 inference speed와 mAP50-95를 갖는 모델 버전을 선택했고, 이후에  hyper parameters를 최적화 했다. 
VOLOv8의 각 버전에 대한 정보는 아래와 같다.   
nano (yolov8n), small (yolov8s), medium (yolov8m), large (yolov8l), and extra large (yolov8x)

|Model|Input size(pixels)|mAP50-95|params(M)|FLOPS(B)|
|-|-|-|-|-|
|YOLOv8n|640|37.3|3.2|8.7|
|YOLOv8s|640|44.9|11.2|28.6|
|YOLOv8m|640|50.2|25.9|78.9|
|YOLOv8l|640|52.9|43.7|165.2|
|YOLOv8x|640|53.9|68.2|257.8|

small, medium, large 모델들의 parameters는 각각 (11151080, 25879480, & 43660680)이고, layers는 (225,295, & 365)이다. 본 논문에서 학습시킨 결과 small과 medium 사이에서는 mAP50-95에서 큰 성능향상이 있었으나 medium, large 사이에서는 그렇지 않았다고 한다. 또한 validation set에 대해서 small, medium, and large의 inference speed는 각각 4.1, 5.7, and 9.3 milliseconds 였다고 한다. 원래 목표였던 average inference speed는 30 to 60 frames for 1080p였고, medium size
model을 multiple 1080p HD videos에서 테스트해본 결과 average total speed (pre-proccess speed(0.5ms) + inference speed(17.25ms) + post-process speed(2ms)) of 19.75 ms(50 frames per second)로 목표에 적합하여 모델을 medium size로 결정하고 hyper-parameters 튜닝을 진행했다고 한다. 

## Loss Function and Update Rule
본 논문에서 제안하는 Loss Function을 일반화하면 아래와 같다. 

$$L(θ) = \frac{λ_{box}}{N_{pos}}L_{box}(θ) + \frac{λ_{cls}}{N_{pos}}L_{cls}(θ) + \frac{λ_{dfl}}{N_{pos}}L_{dfl}(θ) + φ||θ||^2_2$$

$$V^t = \beta V^{t-1} + ∇_θ L(θ^{t-1})$$

$$θ^t = θ^{t-1} - ηV^t$$

첫 번째 식은 일반화된 Loss Function으로 box loss, classification loss, distribution focal loss 각각의 Loss들을 합하고, weight decay인 $φ$를 활용해 마지막 항에서 regularization을 한다. 두 번째 식은 momentum $β$를 이용한 velocity term이다. 세 번째 식은 가중치 업데이트로 $η$는 learning rate이다. 

YOLOv8의 loss function을 자세히 살펴보면 아래와 같다. 

$$L = \frac{λ_ {box}}{N_ {pos}} \sum_ {x, y} 𝟙_ {c^{\star}_ {x, y}} \left[1 - q_ {x,y} + \frac{||b_ {x, y} - \hat{b}_ {x, y}||^2_2}{ρ^2} + α_ {x, y} v_ {x, y}\right]$$

$$+\frac{λ_ {cls}}{N_ {pos}} \sum _{x,y} \sum _{c \in classes} y _c log(\hat{y} _c) + (1 - y _c) log(1 - \hat{y} _c)$$ 

$$+\frac{λ_{dfl}}{N_{pos}} \sum_{x,y} 𝟙_{c^{\star}_ {x, y}} \left[ -(q_ {(x,y)+1} - q_{x,y})log(\hat{q}_ {x,y}) + (q_{x,y} - q_{(x,y)-1})log(\hat{q}_{(x,y)+1})\right]$$

$where:$

$$q_{x,y} = IOU_{x,y} = \frac{\hat{β}_ {x,y} ∩ β_{x,y}}{\hat{β}_ {x,y} ∪ β_{x,y}}$$

$$v_{x,y} = \frac{4}{π^2}(arctan(\frac{w_{x,y}}{h_{x,y}}) - arctan(\frac{\hat{w}_ {x,y}}{\hat{h}_{x,y}}))^2$$

$$α_{x,y} = \frac{v}{1 - q_{x,y}}$$

$$\hat{y}_c = σ(·)$$

$$\hat{q}_{x,y} = softmax(·)$$

$and:$



box loss : https://arxiv.org/abs/1911.08287  
class loss : standard binary cross entropy  
distribution focal loss :  https://arxiv.org/abs/2006.04388  


# Reference 
## Web Link 
One-stage object detection : https://machinethink.net/blog/object-detection/  
YOLOv5 : https://blog.roboflow.com/yolov5-improvements-and-evaluation/   
YOLOv8 : https://blog.roboflow.com/whats-new-in-yolov8/       
mAP : https://blog.roboflow.com/mean-average-precision/  
　　　https://a292run.tistory.com/entry/mean-Average-PrecisionmAP-%EA%B3%84%EC%82%B0%ED%95%98%EA%B8%B0-1   
　　　https://donologue.tistory.com/405    
SiLU : https://tae-jun.tistory.com/10     
Weight Decay, BN : https://blog.janestreet.com/l2-regularization-and-batch-norm/  
focal Loss : https://gaussian37.github.io/dl-concept-focal_loss/  
cross entropy : https://velog.io/@rcchun/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%81%AC%EB%A1%9C%EC%8A%A4-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BCcross-entropy  
DIOU, CIOU : https://melona94.tistory.com/3

## Paper 
Real-Time Flying Object Detection with YOLOv8 : https://arxiv.org/pdf/2305.09972.pdf   
YOLO : https://arxiv.org/pdf/1506.02640.pdf    
YOLOv2 : https://arxiv.org/pdf/1612.08242.pdf    
YOLOv3 : https://arxiv.org/pdf/1804.02767.pdf  
YOLOv4 : https://arxiv.org/pdf/2004.10934.pdf   
YOLOv6 : https://arxiv.org/pdf/2209.02976.pdf  
YOLOv7 : https://arxiv.org/pdf/2207.02696.pdf  
