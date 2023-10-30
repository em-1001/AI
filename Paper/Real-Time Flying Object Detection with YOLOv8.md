# Background
## Cross Entropy
Cross Entropy 는 정보 이론에서 파생된 개념으로, 확률 분포의 차이를 측정하는 지표이다. 두 확률 분포 간의 유사성을 평가하거나, 분류 문제에서 예측값과 실제값 간의 차이를 계산하는 데 사용된다. 

데이터 확률 분포를 $P(x)$, 모델이 추정하는 확률 분포를 $Q(x)$라고 할 때, 두 확률 분포 $P$와 $Q$의 차이를 측정하는 지표인 Cross Entropy는 아래와 같이 표현된다. 

$$H(p, q) = H(p) + D_{KL}(p||q) = -\sum_{i=0}^{n} p(x_i)\log{(q(x_i))}$$

일반적인 Classification 문제에서 주로 cross entropy loss를 사용한다. 이때 True distribution $P$는 one-hot 인코딩된 vector(Ground Truth)를 사용한다. Prediction distribution $Q$ 는 모델의 예측 값으로 softmax layer를 거친 후의 값이고, 클래스 별 확률 값을 모두 합치면 1이 된다. 

예를 들어 $P = [0, 1, 0]$,  $Q = [0.2, 0.7, 0.1]$ 일 때, cross entropy loss 결과는 아래와 같다.

$$
\begin{aligned}
H(P,Q)&=-\sum P(x)log Q(x)\\
&=-(0 \cdot \log{0.2} + 1 \cdot \log{0.7} + 0 \cdot \log{0.1})\\
&=-\log{0.7}
\end{aligned}$$

이진 분류 문제에서의 cross entropy는 다음과 같이 표현된다. 

$$H(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N (y_i \log{(\hat{y}_i)} + (1-y_i) \log{(1 - \hat{y}_i)})$$

$y_i$는 실제 클래스를 나타내는 값(0 또는 1)이고, $\hat{y}_i$는 모델의 예측 확률을 나타낸다.   

위 식은 다음과 같이 유도된다.
1. 정보 이론적 관점  
이진 분류 문제에서, 실제값 $y$가 1이라면 모델은 1로 예측해야 하며, cross entropy는 $-log(\hat{y})$가 되야한다.
반대로 실제값 $y$가 0이라면 모델은 0으로 예측해야 하며, cross entropy는 $-log(1 - \hat{y})$가 된다.
2. 평균화 및 합산  
이러한 관점에서 전체 데이터셋에 대해 각각의 경우(실제값이 1 또는 0인 경우)의 교차 엔트로피를 합산하여 평균을 취한 것이 최종적인 이진 분류 문제의 교차 엔트로피 손실 함수의 식이 된다.


위 식은 실제 분포(Ground Truth)인 $y_i$와 모델의 예측인 $\hat{y}_i$사이의 정보량을 측정하고 모델이 Ground Truth와 일치할수록, Cross Entropy의 값은 작아진다.  

## GIoU, DIoU,  CIoU
일반적으로 IoU-based loss는 다음과 같이 표현된다. 

$$L = 1 - IoU + \mathcal{R}(B, B^{gt})$$

여기서 $R(B, B^{gt})$는  predicted box $B$와 target box $B^{gt}$에 대한 penalty term이다.  
$1 - IoU$로만 Loss를 구할 경우 box가 겹치지 않는 case에 대해서 어느 정도의 오차로 교집합이 생기지 않은 것인지 알 수 없어서 gradient vanishing 문제가 발생했다. 이러한 문제를 해결하기 위해 penalty term을 추가한 것이다. 

### Generalized-IoU(GIoU)
Generalized-IoU(GIoU) 의 경우 Loss는 다음과 같이 계산된다. 

$$L_{GIoU} = 1 - IoU + \frac{|C - B ∪ B^{gt}|}{|C|}$$

여기서 $C$는 $B$와 $B^{gt}$를 모두 포함하는 최소 크기의 Box를 의미한다. Generalized-IoU는 겹치지 않는 박스에 대한 gradient vanishing 문제는 개선했지만 horizontal과 vertical에 대해서 에러가 크다. 이는 target box와 수평, 수직선을 이루는 Anchor box에 대해서는 $|C - B ∪ B^{gt}|$가 매우 작거나 0에 가까워서 IoU와 비슷하게 동작하기 때문이다. 또한 겹치지 않는 box에 대해서 일단 predicted box의 크기를 매우 키우고 IoU를 늘리는 동작 특성 때문에 수렴 속도가 매우 느리다. 

### Distance-IoU(DIoU)
GIoU가 면적 기반의 penalty term을 부여했다면, DIoU는 거리 기반의 penalty term을 부여한다. 
DIoU의 penalty term은 다음과 같다. 

$$\mathcal{R}_{DIoU} = \frac{\rho^2(b, b^{gt})}{c^2}$$

$\rho^2$는 Euclidean거리이며 $c$는 $B$와 $B^{gt}$를 포함하는 가장 작은 Box의 대각선 거리이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4abe5f78-388b-459f-a3f4-95e41a5fdb0a" height="30%" width="30%"></p>

DIoU Loss는 두 개의 box가 완벽히 일치하면 0, 매우 멀어지면 $L_{GIoU} = L_{DIoU} \mapsto 2$가 된다. 이는 IoU가 0이 되고, penalty term이 1에 가깝게 되기 때문이다. Distance-IoU는 두 box의 중심 거리를 직접적으로 줄이기 때문에 GIoU에 비해 수렴이 빠르고, 거리기반이므로 수평, 수직방향에서 또한 수렴이 빠르다. 

### Complete-IoU(CIoU)
DIoU, CIoU를 제안한 논문에서 말하는 성공적인 Bounding Box Regression을 위한 3가지 조건은 overlap area, central point
distance, aspect ratio이다. 이 중 overlap area, central point는 DIoU에서 이미 고려했고 여기에 aspect ratio를 고려한 penalty term을 추가한 것이 CIoU이다. CIoU penalty term는 다음과 같이 정의된다. 

$$\mathcal{R}_{CIoU} = \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

$$v = \frac{4}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}})^2$$

$$\alpha = \frac{v}{(1 - IoU) + v}$$

$v$의 경우 bbox는 직사각형이고 $\arctan{\frac{w}{h}} = \theta$이므로 $\theta$의 차이를 통해 aspect ratio를 구하게 된다. 이때 $v$에 $\frac{2}{π}$가 곱해지는 이유는 $\arctan$ 함수의 최대치가 $\frac{2}{π}$ 이므로 scale을 조정해주기 위해서이다. 

$\alpha$는 trade-off 파라미터로 IoU가 큰 box에 대해 더 큰 penalty를 주게 된다. 

CIoU에 대해 최적화를 수행하면 아래와 같은 기울기를 얻게 된다. 이때, $w, h$는 모두 0과 1사이로 값이 작아 gradient explosion을 유발할 수 있다. 따라서 실제 구현 시에는 $\frac{1}{w^2 + h^2} = 1$로 설정한다. 

$$\frac{\partial v}{\partial w} = \frac{8}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}}) \times \frac{h}{w^2 + h^2}$$ 

$$\frac{\partial v}{\partial h} = -\frac{8}{π^2}(\arctan{\frac{w^{gt}}{h^{gt}}} - \arctan{\frac{w}{h}}) \times \frac{w}{w^2 + h^2}$$ 

## QFL, DFL, GFL
One-stage detector는 기본적으로 object detection을 dense classification과 localization (i.e., bounding box regression)을 통해서 한다. classification의 경우 Focal Loss로 최적화 되고, box location은 Dirac delta distribution으로 학습된다. QFL, DFL를 제안한 논문에서 말하는 기존 방식의 문제는 크게 두 가지이다. 
1. 학습, 추론 시 quality estimation과 classification의 비일관성   
학습시 classification score 와 centerness(또는 iou)score 가 별개로 학습되지만 inference 시에는 nms전에 두 score를 join해서 사용(element wise multiplication)한다. 이러한 두 score의 비일관성은 성능저하로 이어질 수 있고 논문에서는 두 score를 train, test 모두에서 joint해주어 둘 사이의 상관성을 크게 갖도록 유도했다. 
2. Dirac delta distribution의 Inflexible  
기존 방식들은 positive sample 위치에만 box gt를 할당해 regression 하는 방식을 취하는데 이는 dirac delta distribution으로 볼 수 있다. dirac delta distribution는 물체의 occlusion, shadow, blur등으로 인한 물체 경계 불분명 등의 문제를 잘 커버하지 못한다.

### Quality Focal Loss(QFL)
training과 test 시의 inconsistency를 해결하기 위해서 supervision을 기존의 one-hot label에서 float target $y ∈ [0, 1]$이 가능하도록 soften하였다. 참고로 여기서 $y=0$은 0 quality score를 갖는 negative samples을, $0 < y ≤ 1$는 target IoU score $y$를 갖는 positive samples을 의미하게 된다. 논문에서 제안하는 QFL는 다음과 같이 계산된다. 

$$QFL(\sigma) = -|y - \sigma|^{\beta}((1-y)\log{(1 - \sigma)} + y \log{(\sigma)})$$

기존의 Focal Loss의 경우 $\{ 1, 0 \}$의 discrete labels만 지원하였지만 QFL에서 사용하는 새로운 label은 decimals을 포함하므로 위와 같은 식이 나왔고 기존 FL에서 바뀐 부분은 다음과 같다. 

1. 기존 cross entropy part인 $-\log{(p_t)}$가 complete version인 $-((1-y)\log{(1 - \sigma)} + y \log{(\sigma)})$로 확장되었다.   
2. scaling factor인 $(1-p_t)^{\gamma}$가 estimation $\sigma$와 continuous labe $y$ 사이의 absolute distance인 $|y - \sigma|^{\beta}$로 변경되었다. ($| · |$는 non-negativit를 보장한다.)

### Distribution Focal Loss
기존 bounding box regression의 경우 앞서 언급했듯이 Dirac delta distribution $\delta(x - y)$를 이용해서 regression되었다. 
이는 주로 fully connected layers를 통해 implemented되며 아래와 같다. 

$$y=\int_{-\infty}^{+\infty} \delta(x - y)x \ dx$$

논문에서는 Dirac delta나 Gaussian 대신에 General distribution $P(x)$을 직접 학습하는 것을 제안한다. label $y$의 범위는 $y_0 ≤ y ≤ y_n, n \in \mathbb{N}^+$에 속하고 estimated value인 $\hat{y}$를 다음과 같이 계산한다. 물론 $\hat{y}$도 $y_0 ≤ \hat{y} ≤ y_n$를 만족한다. 

$$\hat{y}=\int_{-\infty}^{+\infty} P(x)x \ dx = \int_{y_0}^{y_n} P(x)x \ dx$$

이때 학습하는 label의 분포가 연속적이지 않고 이산적이므로 위 식을 아래와 같이 나타낼 수 있다. 

$$\hat{y} = \sum_{i=0}^{n} P(y_i)y_i$$

이때 intervals $∆$은 간단하게 $∆ = 1$로 하고, $\sum P(y_i) = 1$을 만족한다. 
위 식에서 $y_i$는 object 중심으로부터 각 변까지의 거리의 discrete한 값이고 $P(y_i)$는 네트워크가 추론한 현 anchor에서 boundary까지의 거리가 $y_i$일 확률이다. 따라서 DFL은 object boundary 까지의 거리를 직접적으로 추론하는 것이 아니라 각 거리에 대한 확률 값이 있으면 이 값들의 기댓값으로 추론하는 것이다. 

$P(x)$는 softmax $S(\cdot)$을 통해 쉽게 구해질 수 있다. 또한 $P(y_i)$를 간단하게 $S_i$로 표현한다. DFL을 통한 학습은 $P(x)$의 형태가 target인 $y$에 가까운 값이 높은 probabilities를 갖도록 유도한다. 따라서 DFL은 target $y$에 가장 가까운 두 값 $y_i, y_{i+1}$ ($y_i ≤ y ≤ y_{i+1}$)의 probabilities를 높임으로서 네트워크가 빠르게 label $y$에 집중할 수 있도록 한다. 
DFL은 QFL의 complete cross entropy part를 이용하여 다음과 같이 계산된다. 

$$DFL(S_i, S_{i+1}) = -((y_{i+1} - y) \log{(S_i)} + (y - y_i) \log{(S_{i+1})})$$

DFL의 global minimum solution은 $S_i = \frac{y_{i+1} - y}{y_{i+1} - y_i}, \ S_{i+1} = \frac{y - y_i}{y_{i+1} - y_i}$가 되고 이를 통해 계산한 estimated regression target $\hat{y}$는 corresponding labe $y$에 무한히 가깝다.    
i.e $\hat{y} = \sum P(y_j)y_j = S_i y_i + S_{i+1} y_{i+1} = \frac{y_{i+1} - y}{y_{i+1} - y_i} y_i + \frac{y - y_i}{y_{i+1} - y_i} y_{i+1} = y$

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

$$+\frac{λ_{dfl}}{N_{pos}} \sum_{x,y} 𝟙_{c^{\star}_ {x, y}} - \left[(q_ {(x,y)+1} - q_{x,y})log(\hat{q}_ {x,y}) + (q_{x,y} - q_{(x,y)-1})log(\hat{q}_{(x,y)+1})\right]$$

$where:$

$$q_{x,y} = IOU_{x,y} = \frac{\hat{β}_ {x,y} ∩ β_{x,y}}{\hat{β}_ {x,y} ∪ β_{x,y}}$$

$$v_{x,y} = \frac{4}{π^2}(\arctan{(\frac{w_{x,y}}{h_{x,y}})} - \arctan{(\frac{\hat{w}_ {x,y}}{\hat{h}_{x,y}})})^2$$

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
SiLU : https://tae-jun.tistory.com/10     
Weight Decay, BN : https://blog.janestreet.com/l2-regularization-and-batch-norm/  
Focal Loss : https://gaussian37.github.io/dl-concept-focal_loss/  
　　　　 　https://woochan-autobiography.tistory.com/929  
Cross Entropy : https://sosoeasy.tistory.com/351  
DIOU, CIOU : https://hongl.tistory.com/215    
QFL, DFL : https://pajamacoder.tistory.com/m/74  
YOLOv8 Loss 수정 : https://velog.io/@easyssun/YOLOv8-%EB%AA%A8%EB%8D%B8-loss-function-%EC%88%98%EC%A0%95  

## Paper 
Real-Time Flying Object Detection with YOLOv8 : https://arxiv.org/pdf/2305.09972.pdf   
YOLO : https://arxiv.org/pdf/1506.02640.pdf    
YOLOv2 : https://arxiv.org/pdf/1612.08242.pdf    
YOLOv3 : https://arxiv.org/pdf/1804.02767.pdf  
YOLOv4 : https://arxiv.org/pdf/2004.10934.pdf   
YOLOv6 : https://arxiv.org/pdf/2209.02976.pdf  
YOLOv7 : https://arxiv.org/pdf/2207.02696.pdf  
