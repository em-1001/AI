# Activation Functions
앞선 강의에서도 언급했던 것처럼, input이 들어오면 가중치와 곱해지고, 비선형 함수인 활성함수를 거쳐 해당 데이터의 활성화여부를 결정해준다고 배웠다. 
활성화 함수에는 여러가지가 있고, 각 활성함수의 문제점과 이를 어떻게 개선해나갔는지에 대해 알아볼 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/f78e4ffc-7678-46ff-9ee6-d6d16bd09f59" height="70%" width="70%"></p>

활성함수는 기본적으로 input을 특정 범위의 출력으로 변환해주는 단순한 기능도 하지만, output에 비선형성을 부여하기도 한다. 
활성함수는 미분이 가능해야 하며 그렇지 않다면 역전파를 실패하게 된다. 

비선형성이 필요한 이유는 아래와 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3f5148ac-27aa-4fd4-b1e6-37b2b9fe3b42" height="70%" width="70%"></p>

위의 그림과 같이 circle 또는 elliptical 같은 non-linearity data가 있다고 가정하고, data point가 positive class or nagative class에 속하는지 분류하는 작업을 생각해보면 이런 경우엔 선형 모델을 적용하는 것이 불가능하다. 

정리하면 활성함수를 사용하는 이유는 이렇다. 만약 우리가 그냥 선형 함수 $h(x) = cx$ 를 활성함수로 사용한다면, multiple layer의 결과는 $h(h(h(x)))$가 될 것이고 이는 $y = c^3x$가 된다. 만약 $c^3$을 $a$로 치환한다면 이는 다시 선형 함수가 되어버리기 때문에 layer가 의미가 없어진다. 

### Sigmoid 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/69847ed5-8d1a-4143-9f43-9ff25654ef89" height="40%" width="40%"></p>

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

Sigmoid의 문제는 아래와 같은 것들이 있다. 

#### Vanishing Gradient
첫 번째는 Vanishing Gradient 문제이다. 
Sigmoid는 범위가 [0, 1]인데 0의 가까운 값에서만 simgoid가 active된다고 볼 수 있다. 
그래프를 보면 알 수 있듯이 만약 W가 10이거나 -10인 경우(매우 작거나, 매우 클 때)에는 1 또는 0의 값으로 수렴하게 된다. 
이렇게 되면 도함수가 0에 가까워지며 작아지므로 error가 sigmoid activated neural networks에서 back propagation하는 동안, 기울기 저하가 발생하고, gradient가 사라지게 된다. 
초기 레이어에 대한 gradient 값이 줄어들고 해당 레이어는 제대로 학습할 수 없게된다. 
즉, 네트워크의 깊이와 값이 0으로 이동하는 활성화로 인해 기울기가 사라지는 경향이 있다. 

#### Not zero-centered
sigmoid의 결과는 0 중심이 아니다.  모든 값은 0 이상의 값을 가져서 slow convergence를 가져오게 된다. 
Zero-centered란 그래프의 중심이 0인 상태로, 함숫값이 음수 혹은 양수에만 치우쳐 존재하지 않고 실수 전체에서 고르게 나타나는 형태를 의미한다. 하지만 시그모이드는 위의 그래프에서도 볼 수 있듯이 함숫값이 항상 0~1이기 때문에 함숫값이 양수만 존재한다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/78ec8259-443a-4151-a9ca-4cb6a9df1583" height="70%" width="70%"></p>

이것의 문제점은 위 사진에서 볼 수 있다. 위의 예시를 통해 이것의 문제점을 알아보자. Neural networks에서 input은 예전 layer의 결과값이라고 생각하면 된다. (activation function을 통과한 결과 값을 다음 layer에 넘기기 때문) 그런데 시그모이드 함수의 결과값은 항상 양수이기 때문에 시그모이드 함수를 한번 거친 이후로는 이 input 값이 항상 양수가 된다. 이렇게 되면 backpropagation을 할 때 문제가 생긴다. backpropagation을 하게 되면 chain rule을 사용하는데, $\frac{dL}{dw} = \frac{dL}{df} * \frac{df}{dw}$ 의 계산 과정을 거칠 때 $]frac{df}{dw} = x$이므로 $\frac{dL}{dw} = \frac{dL}{df} * x$ 가 된다. 이렇게 되면 input값인 x(local gradient)가 항상 양수이기 때문에 $\frac{dL}{dw}$ 와 $\frac{dL}{df}$의 부호는 같을 수밖에 없다.

w가 2차원이라고 하면 좌표 평면에서 부호를 따져봤을 때 부호가 같은 부분은 1,3사분면이다. 따라서 가상 최적 w는 파란색 벡터임에도 불구하고, 빨간색 벡터처럼 지그재그의 형태로 기울기가 업데이트(학습) 될 수밖에 없다. 그렇기 때문에 zero-centered가 필요하다. input x의 부호가 다양해야 기울기를 업데이트 할 때 적절한 방향으로 진행할 수 있기 때문이다.


#### Compute expensive
보통 지수 함수 $exp()$의 경우 연산이 굉장히 커서, 성능이 저하 된다고 볼 수 있다.  


### tanh

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/76404bc1-1928-4635-b9d9-0d469ad07b2f" height="40%" width="40%"></p>

$$tanh(x)$$

$tanh$의 경우 data가 0을 중심으로 한다. 이는 입력 데이터의 평균이 0 근처에 있음을 의미하고, Not zero-centered 문제를 해결할 수 있다. 따라서 zero centered 이지만 saturated 될 때 gradient는 여전히 죽는 문제는 남아있다. 


### ReLU

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/21663338-9508-4f42-a08c-bdd42ae25146" height="40%" width="40%"></p>

$$max(0, x)$$

#### The best activation function
복잡한 계산을 필요로 하지 않기 때문에, 계산 효율 좋다.(sigmoid나 tanh보다 수렴 속도가 약 6배 빠름)
또한 x가 양수이면 saturation 되지 않는다. 

ReLU 활성화 함수를 사용할 때의 이점을 고려하는 또 다른 중요한 속성은 sparsity(희소성)이다. 
일반적으로 대부분의 항목이 0인 행렬을 sparse matrix(희소 행렬)이라고 하며 마찬가지로 일부 가중치가 0인 신경망에서 이와 같은 속성을 원한다. 

sparsity은 종종 더 나은 predictive power와 overfitting/noise가 적은 간결한 모델을 생성한다. 
sparse network에서는 뉴런이 실제로 문제의 의미 있는 측면을 처리할 가능성이 더 큰데, 예를 들어, 이미지에서 사람의 얼굴을 감지하는 모델에는 귀를 식별할 수 있는 뉴런이 있을 수 있으며, 이미지가 얼굴이 아니고 배나 산인 경우 활성화되지 않아야 한다. 

ReLU는 모든 음수 입력에 대해 출력 0을 제공하기 때문에 주어진 단위가 전혀 활성화되지 않아 네트워크가 sparse해질 가능성이 있다. 

#### Exploding Gradient
ReLU 활성화 함수에는 exploding gradient과 같은 몇 가지 문제가 있다. 

exploding gradient는 vanishing gradient의 반대 개념이다. 
큰 오류 gradient가 누적되어서 훈련 중에 신경망 모델 가중치가 매우 크게 업데이트되는 경우에 발생한다. 

또한, 모든 음수 값에 대해 0이 되는 점이 단점이 되기도 한다. 

이 문제를 dead ReLU라고 하며, ReLU neuron이 음수 쪽에 붙어 있고 항상 0을 출력하면 "dying"한다고 본다. 
음수 값에서 ReLU의 gradient 범위도 0이다. neuron이 음수가 되면 다시 살아날 가능성이 거의 없다. 
이러한 뉴런은 input을 구별하는 데 아무런 역할도 하지 않으며 본질적으로 쓸모가 없게된다. 
dying하는 문제는 learning rate가 너무 높거나 negative bias가 클 때 발생하기 쉽다. 


<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/9831c709-6974-45b8-b64a-e69ebb57a288" height="60%" width="60%"></p>

- 초기화를 잘못해서 가중치 평면이 data cloud에서 멀리 떨어진 경우
- Learning rate가 지나치게 높은 경우, 가중치가 날뛰게 되며 ReLU가 데이터의 manifold를 벗어나게 됨
- 학습 다 시켜놓은 네트워크를 살펴보면 10-20%는 dead ReLU가 되어있는데, 이정도는 괜찮다.
- 그림에서 초록빨강 평면은 ReLU의 입력 행렬을 표현한 것
- 그래서 초기화시에 positive biases를 추가해주는 경우 많다. (active 될 확률 높이기)
- 대부분은 zero-bias로 사용

### Leaky ReLU

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/6812503e-bdc4-43df-b2a9-83f199296877" height="40%" width="40%"></p>

$$f(x) = max(0.01x, x)$$

Leaky ReLU는 ReLU Function의 Dying 문제를 극복한 ReLU Activation Function의 확장이다. 
Dying ReLU 문제는 제공된 모든 입력에 대해 비활성화되어 Neural Network의 성능에 영향을 미친다. 
이 문제를 해결하기 위해 ReLU Activation Function과 달리 Negative 입력에 대한 음의 기울기가 작게 조정된 Leaky ReLU가 있다. 

Leaky ReLU의 한계는 복잡한 분류에 사용할 수 없는 선형 곡선을 가지고 있다는 것이다. 

- zero-mean
- 음의 영역에서도 이제 saturation 되지 않음
- dead ReLU도 없음

### PReLU

$$f(x) = max(\alpha x, x)$$

- Leaky ReLU와 비슷하지만 기울기 alpha (파라미터)로 결정됨


### ELU

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/8f195da2-e21f-4eb4-8aeb-fc6807dc0c7a" height="40%" width="40%"></p>

$$
f(x)=
\begin{cases}
x & if \ x > 0 \\
\alpha(exp(x)-1 & if \ x \le 0 \\ 
\end{cases}
$$

- zero-mean에 가까운 출력값
- 음에서 saturation. 하지만 saturation이 노이즈에 강인하다고 생각
- dying ReLU문제 극복(음수 값의 기울기가 0이 아니라서)
- ReLU, sigmoid, Hyperbolic Tangent보다 높은 accuracy를 갖는다.

ReLU와 달리 ELU는 음숫값을 가지므로 함수의 평균이 0으로 이동하고, 이 점에서 더 빠르게 convergence한다고 주장된다.
ELU 함수는 지수함수여서 계산 속도는 느리지만 더 빠른 convergence로 빠르게 학습되는 원리이다. 


### Maxout Neuron

$$max(w_1^Tx + b_1, w_2^Tx + b_2)$$

- 기본형식을 정의하지 않음
- 두개의 선형함수 중 큰 값을 선택 -> ReLU와 leaky RELU의 일반화 버전
- 선형이기에 saturation안되고 gradient 안죽음
- W1, W2 때문에 파라미터 수 두배됨

여러 선형 함수를 사용하여 함수를 근사하는 것을 piece-wise linear approximation(PWL) 라고한다. 

### TLDR: In practice:
- Use **ReLU**. Be careful with your learning rates
- Try out **Leaky ReLU / Maxout / ELU**
- Try out tanh but don’t expect much
- **Don’t** use sigmoid



# Data Preprocessing

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/97d5d794-cd77-441c-84c7-9294dce58f2e" height="70%" width="70%"></p>

- zero-mean으로 만들고 표준편차로 normalize
- 이미지의 경우 스케일이 어느정도 맞춰져 있어서(0~255) zero-mean만 해준다.

해주는 이유는 앞서 입력이 전부 positive한 경우를 방지하는 것과 같음. 학습최적화를 위한 것. 
평균값은 전체 training data에서 계산하여 test 데이터에도 동일하게 적용. 
RGB별로 각각 평균을 내는 경우도 있다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e22093d7-705e-4cce-8142-19190769c51a" height="70%" width="70%"></p>

위 사진의 경우 주성분결정(PCA)과 Whitening 을 해주는 것인데, 먼저 PCA는 두 번째 그림으로 데이터의 차원을 줄여주는 역할을 하고, 
Whitening은 이미지 인접한 픽셀간의 중복성을 줄여주는 역할을 한다. 하지만 이 역시도 이미지를 대상으로 할 때는 큰 의미가 없다.

결론적으로 이미지는 zero-centered만 신경 써주면 된다. 


# Weight Initialization
## Zero initialization
모든 가중치를 0으로 초기화하면 뉴런이 훈련 중에 동일한 feature를 학습하게 된다. 
backpropagation도 동일한 연산이 진행될 것이다. 

일정한 초기화 방식은 성능이 매우 좋지 않다. 
두 개의 hidden units이 있는 신경망을 고려하고 모든 bias를 0으로 초기화하고 weight을 일정한 α로 초기화한다고 가정하자. 
이 네트워크에 input $(x_1, x_2)$를 전달하면 두 hidden units의 출력은 relu $(αx_1 + αx_2)$가 된다. 
따라서 두 hidden units은 cost에 동일한 영향을 미치므로 동일한 기울기가 발생된다. 
그러므로 두 뉴런은 훈련 전반에 걸쳐 대칭적으로 진화하여 서로 다른 뉴런이 서로 다른 것을 학습하는 것을 막아버린다.

## Random Initialization
가중치에 임의의 값을 할당하는 것이 0을 할당하는 것보다 낫다. 
하지만 가중치가 아주 크거나 작은 값으로 초기화되면 두 가지 문제에 직면하게 된다. 

### i) Vanishing Gradient
가중치가 단위 행렬보다 약간 작게 초기화되는 경우를 가정해보자. 

$$
W^{[1]} = W^{[2]} = ... = W^{[L-1]} = 
\begin{bmatrix}
0.5&0\\
0&0.5\\ 
\end{bmatrix}
$$

이것은 $y = W[L]0.5^{L-1}x$ 로 단순화 되고 활성화 $\alpha[l]$의 값은 $[l]$에 따라 기하급수적으로 감소한다. 
이런 activations가 backward propagation에서는 vanishing gradient 문제에 직면한다. 
parameter에 대한 cost의 gradient가 너무 작아서 최솟값에 도달하기 전 cost가 convergence하게된다.

### ii) Exploading Gradient
가중치가 매우 높은 값으로 초기화되면 exploding gradient 문제에 직면한다. 
모든 가중치가 단위 행렬보다 약간 높게 초기화되는 경우를 가정해보자. 

$$
W^{[1]} = W^{[2]} = ... = W^{[L-1]} = 
\begin{bmatrix}
1.5&0\\
0&1.5\\ 
\end{bmatrix}
$$

이것은 $y = W[L]1.5^{L-1}x$ 로 단순화 되고 활성화 $\alpha[l]$의 값은 $[l]$에 따라 기하급수적으로 증가한다. 
똑같이 backpropagation에 사용되면 exploding gradient 문제가 발생한다. 
즉, parameter에 대한 cost의 gradient가 너무 크면, cost가 최솟값을 중심으로 진동하는 overshooting이 발생되어 모든 neuron이 saturated 된다. 

## New initialization technique
network의 activation이 Vanishing Gradient와 Exploading Gradient 되는 것을 방지하기 위해 다음과 같은 규칙을 따른다. 
1. activations의 평균은 0이 되어야 한다.
2. activation의 분산은 모든 layer에서 동일하게 유지되어야 한다.

### Xavier Initialization

$$\frac{1}{\sqrt{n}}$$

각 layer의 활성화값을 더 광범위하게 분포시킬 목적으로 weight의 적절한 분포를 찾으려 시도했다.
또한, tanh 또는 sigmoid로 활성화 되는 초깃값을 위해 이 방법을 주로 사용한다. 

이전 layer의 neuron의 개수가 $n$ 이라면 표준편차가 $\frac{1}{\sqrt{n}}$인 분포를 사용하는 개념이다. 
너무 크지도 않고 작지도 않은 weight을 사용하여 gradient가 vanishing하거나 exploding하는 문제를 막는다.
Random initialization에 위에 주어진 값을 곱하기만 하면 된다. 

Xavier Initialization을 사용하게 된다면, 뉴런의 개수가 많을수록 초깃값으로 설정하는 weight이 더 좁게 퍼짐을 알 수 있다.
```py
node_num = 100 # 이전의 neuron 수
w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
```

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/970b3756-d8b4-424c-83fc-c6b85b695d17" height="50%" width="50%"></p>

데이터가 더 적당히 넓게 펴지므로 sigmoid 함수를 사용할 때도 표현을 제한 받지 않고 학습이 가능하다.
또한, neuron의 개수에 따라 weight이 초기화되기 때문에 고정된 표준편차를 사용할 때보다 더 robust한 성질을 가진다. 


### He Initialization

$$\frac{2}{\sqrt{n}}$$

He Initialization는 ReLU를 위해 만들어진 초기화 방법이다.
He Initialization는 앞 layer의 neuron이 $n$ 개일 때, 표준편차가 $\frac{2}{\sqrt{n}}$인 정규분포를 사용한다.

ReLU는 음의 영역이 0이라서 활성화되는 영역을 더 넓게 분포시키기 위해 Xavier보다 2배의 계수가 필요하다고 해석하면 된다. 
Random initialization에 위에 주어진 값을 곱하기만 하면 된다.

activation function으로 ReLU를 이용한 경우의 activation 분포를 살펴보면

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/790b9e86-ccc5-41df-a23e-8ce5df04ca61" height="70%" width="70%"></p>

He Initialization은 모든 layer에서 균일하게 분포되어 있음을 알 수 있다. 
그래서 backpropagation 때도 적절한 값이 나올 것이라 기대할 수 있다.


# Batch Normalization
수십 개의 layer가 있는 DNN을 훈련하는 것은 학습 알고리즘의 초기 무작위 가중치와 구성들에 민감할 수 있다.
그 이유는 가중치가 업데이트 될 때, 각 mini-batch 이후에 네트워크의 깊은 layer에 대한 입력 분포가 변할 수 있기 때문인데, 
이런 변화를 “internal covariate shift”이라고도 한다. 

Batch Normalization은 각 미니 배치 계층에 대한 입력을 표준화하는 매우 깊은 신경망을 훈련하는 기술이다. 
이는 training process를 안정화하고 DNN을 훈련하는 데 필요한 train epoch 수를 획기적으로 줄이는 효과가 있다.

Batch Normalization을 사용하면 Weight Initialization에 너무 의존하지 않아도 문제를 해결할 수 있다. 
Batch Normalization는 기본적으로 Vanishing Gradient 문제가 발생하지 않도록 하는 방법 중 하나인데, 지금까지는 이런 문제가 발생하지 않게 하기 위해서 활성화함수를 바꾸거나 가중치 초기화를 고려했었는데, Batch Normalization은 이러한 간접적 방법이 아니라 학습하는 방법 자체를 안정화 시키는 근본적인 방법이라 할 수 있다. 

Batch Normalization의 주요 아이디어는 학습에 있어서 불안정화의 이유가 각 layer들을 거치면서 입력값의 분포가 달라지는 현상이 발생하기 때문이라 생각하여 각 layer를 거칠 때마다 이 값들을 정규화 하자는 것에서 나왔다. 

각 layer에서 Batch Normalization을 통해 정규화를 해줘도 여전히 미분가능해서 순전파와 역전파에 아무 문제가 없다. 
Batch Normalization을 적용하는 방식은 mini-batch를 뽑아서 N(batch 내의 수) by D(feature 수)로 이루어진 input X가 있을 때 이 batch에 대해 평균과 분산을 구하고 정규화를 해주는 것이다. 

Batch Normalization은 일반적으로 Fully Connected(Convolutional) layer와 활성화 함수 사이에 위치하게 된다. 

`Fully Connected -> Batch Normalization -> activation function -> Fully Connected -> Batch Normalization  ...`

Batch Normalization는 Vanishing Gradient 문제가 발생하지 않도록 아래 2가지 단계를 취한다. 

1. Normalize

$$\widehat{x}^{(k)} = \frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}$$

2. 정규화 조정 

$$y^{(k)} = \gamma^{(k)}\widehat{x}^{(k)} + \beta^{(k)}$$

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4c1657b1-5843-4fcc-bf2d-4c6ac2490e7f" height="50%" width="50%"></p>

배치 정규화는 간단히 말하자면 미니배치의 평균과 분산을 이용해서 정규화 한 뒤에, scale 및 shift 를 감마(γ) 값, 베타(β) 값을 통해 실행한다. 이 때 감마와 베타 값은 학습 가능한 변수이다. 즉, Backpropagation을 통해서 학습이 된다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3bb91de2-1336-40b6-b0b8-0ae2a8503157" height="60%" width="60%"></p>

위는 감마, 베타에 대한 Backpropatation 수식이다. 

이렇게 정규화 된 값을 활성화 함수의 입력으로 사용하고, 최종 출력 값을 다음 레이어의 입력으로 사용하는 것이다.
기존 output = g(Z), Z = WX + b 식은 **output = g(BN(Z)), Z = WX + b** 로 변경되는 것이다.

위 식에서 입실론($\epsilon$)은 계산할 때 0으로 나눠지는 문제가 발생하는 것을 막기 위한 수치적 안정성을 보장하기 위한 아주 작은 숫자이다. 감마($\gamma$)값은 Scale 에 대한 값이며, 베타($\beta$)값은 Shift Transform 에 대한 값이다. 이들은 데이터를 계속 정규화 하게 되면 활성화 함수의 비선형 같은 성질을 잃게 되는데 이러한 문제를 완화하기 위함이다. 예를 들면 아래 그림과 같이 Sigmoid 함수가 있을 때, 입력 값이 N(0, 1) 이므로 95% 의 입력 값은 Sigmoid 함수 그래프의 중간 (x = (-1.96, 1.96) 구간)에 속하고 이 부분이 선형이기 때문이다. 그래서 비선형 성질을 잃게 되는 것이며, 이러한 성질을 보존하기 위하여 Scale 및 Shift 연산을 수행하는 것이다. 이 값들은 학습을 통해서 값을 결정하게 된다. 이렇게 결정된 값을 통해 Batch Normalization을 아에 하지 않을 것인지에 대한 판단도 할 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/ac31e105-d2f2-4299-a86e-0fadb5113c42" height="40%" width="40%"></p>

주의할 점은 Batch Normalization은 train과 test 시 조금 다르게 동작하는데, train 시에는 batch를 기준으로 mean값을 구하고 test를 할 때는 전체를 기준으로 mean을 구한다. 

# Babysitting the Learning Process
1. 데이터 전처리
2. 아키텍쳐 선택    
Hidden layer, 뉴런의 수, loss function 등을 정한다. 
4. Sanity Check       
loss값이 잘 나오는지 확인. regularization 값을 조금 올리면서 loss가 증가하는지 확인한다. 
6. 이제 학습을 시키는데, 먼저 작은 데이터 셋을 넣는다.   
작은 데이터 셋이기 때문에 100% Overfitting이 나오고 Overfitting이 제대로 되면 모델이 잘 동작한다는 것을 의미한다.
7. regularization, learning rate 값을 찾는다.    
적절한 값들을 넣어 찾아줘야 한다.
<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/9a1e8180-b1bb-4ba2-be37-1497c1e61238" height="60%" width="60%"></p>  

Learning rate가 1e-6일 때 loss가 바뀌는 것을 볼 수 있으나 rate가 너무 작기 때문에 cost 값이 매우 조금씩 떨어진다. 
하지만 train은 서서히 증가하므로 조금이나마 훈련이 되면서 accuracy가 증가한다.   

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/705d7158-abd6-4d15-b29e-efb92e3b2387" height="60%" width="60%"></p>  

learning rate가 1e6일 때는 cost가 nan이 된다. 이는 값이 너무 커서 튕겨저 나간것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/d744b7eb-d27b-476b-9e8a-70ceeacad482" height="60%" width="60%"></p> 

3e-3일 때도 inf로 튕겨저 나간다. 이러한 실험을 통해 1e-3 ~ 1e-5의 값이 적당하다는 것을 추측할 수 있다. 

# Hyperparameter Optimization
최적의 하이퍼 파라미터를 찾는 방법은 아래와 같은 것들이 있다. 이 중 강의에서는 위 2 가지를 소개한다.   
- Grid Search
- Random Search
- Bayesian Optimization

처음은 값을 넓은 범위에서 좁은 범위로 줄여나가는 방법을 사용한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/bc31c582-c1d8-40e7-ac4d-b4553270699b" height="70%" width="70%"></p> 

`10**uniform(-5, 5)`일 때의 case를 보면 val_acc가 48%가 나오는 지점에서 nice한 영역이 나온다. 
lr값이 e-04, reg는 e-01이다. 이제 범위를 좁혀보자. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/c42cea93-1be7-46b2-bda4-c41409ce0f29" height="70%" width="70%"></p> 

Reg는 -4에서 0, lr는 -3에서 -4로 수정했다. 이제는 best지점이 53%지점으로 변경되었다. 이런식으로 좁혀나간다. 
하지만 53%지점은 잘 나왔다고 할 수 없다. -3부터 -4까지다 보니까 범위에 들어가지 않는 값이 있을 수도 있기 때문이다. 
처음에 nice한 지점이 lr이 e-04, reg가 e-01이어서 값을 저렇게 수정했지만 이게 값을 ‘모두’ 충족시키지 않을 수도 있다.
그러므로 값을 조금씩 줄여가며 하는것이 좋다. 

### Random Search vs. Grid Search

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/a8fe0cfe-f841-4b9b-a879-afa080eef635" height="60%" width="60%"></p> 

앞서 설명한 방식으로 찾는 방법은 Random Search, Grid Search로 두 가지가 있다. 위 사진을 통해 알 수 있듯이 Grid Search는 일정한 간격이기 때문에 찾지 못할 수도 있다. 랜덤은 말 그대로 랜덤으로 떨어지기에 더 좋은 값의 영역에 접근할 확률이 좋다고 한다. 그래서 보통 random search를 사용한다. 

이렇게 찾아야 하는 하이퍼파라미터는 네트워크 아키텍처, learning rate, regularization 등이 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/c130d893-0283-4dcb-b887-25a3702c15a1" height="70%" width="70%"></p> 




# Reference 
https://moordo91.tistory.com/32     
https://velog.io/@cha-suyeon/DL-%EA%B0%80%EC%A4%91%EC%B9%98-%EC%B4%88%EA%B8%B0%ED%99%94Weight-Initialization-    
https://velog.io/@cha-suyeon/cs231n-6%EA%B0%95-%EC%A0%95%EB%A6%AC-Training-Neural-Networks-I        
https://say-young.tistory.com/entry/CS231n-Lecture-6-Training-Neural-Networks-I  
https://eehoeskrap.tistory.com/430    
https://lsjsj92.tistory.com/404    
logistic regression : https://acdongpgm.tistory.com/109

