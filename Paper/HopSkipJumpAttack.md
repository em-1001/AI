# HopSkipJumpAttack
## Background
### p - norm
원래의 이미지에 perturbation을 넣어서 Adversarial example을 만들게 되는데 이러한 perturbation의 크기에 대한 지표로 p-norm이 쓰인다. 
p-norm은 특정한 벡터의 크기를 판단하는 기준으로 사용할 수 있다. 
이러한 p-norm을 통해서 perturbation의 크기를 제한할 수 있다. 
특정한 벡터 $x$가 있을 때 그 $x$의 p-norm값은 아래와 같이 계산할 수 있다. 이때 p의 값에 따라서 $L_1-norm$인지, $L_2-norm$인지, $L_{\infty}-norm$인지 등을 정할 수 있다. 

$$||x||_p = (|x_1|^p + |x_2|^p + \cdots + |x_n|^p)^{1/p}$$

$L_n-norm$에 대한 예시는 아래와 같다. 

$$
\begin{aligned}
&||x||_0 = |x_1|^0 + |x_2|^0 + \cdots + |x_n|^0 \le \epsilon \\  
\\
&||x||_1 = |x_1| + |x_2| + \cdots + |x_n| \le \epsilon \\ 
\\
&||x||_2 = (x_1^2 + x_2^2 + \cdots + x_n^2)^{1/2} \le \epsilon \\ 
\\
&||x|| _{\infty} = \max  \left\lbrace |x_1|, |x_2|, ..., |x_n| \right\rbrace  \le \epsilon
\end{aligned}
$$

이미지 같은 경우 매우 고차원상의 벡터로 표현된다. 이때 흑백 이미지의 경우는 총 필셀의 개수가 차원의 크기가 될 것이다. 예를 들어서 $L_0-norm$이라 하면 각각의 축 즉 각각의 픽셀에 대해서 변경된 필셀의 개수를 구하면 된다. 이때 이 값을 입실론으로 제한하면 총 몇개까지가 바뀔 수 있는 것인지를 정하게 된다. 예를 들어 입실론이 1이면 오직 하나의 픽셀만 바꿔서 공격을 수행하는 것이다. 이 외에도 $L_1$은 Manhattan distance $L_2$는 Euclidean Distance가 된다. 마지막으로 $L_{\infty}$의 경우에는 가장 값이 많이 변경된 필셀의 변경된 값을 의미한다. 쉽게 말하면 모든 필셀에 대해서 입실론크기까지만 변경될 수 있다는 것이다. 

참고로 $L_2$와 같은 경우 1, 2, 4, 8.. 과 같은 값으로 입실론을 제한하고, 
$L_{\infty}$의 경우는 4/255, 8/255, 16/255.. 와 같은 값들로 주로 설정한다. 


### Whilte Box & Black Box
적대적공격에는 공격 유형을 Whilte Box 공격과 Black Box 공격으로 나눌 수 있는데 흔히 이를 Threat Model이라 한다. 이는 공격자가 어디까지 알고있는지를 기준으로 공격을 구분한 것인데, 먼저 Whilte Box 공격의 경우 말 그대로 공격자가 모델에 대해 완전히 알고 있는 경우를 말한다. 만약 공격자가 output layer까지만 접근할 수 있다고 하면 이를 Score-based threat model(output layer에서 classify 결과로 나온 각각의 class에 대한 확률값을 알 수 있는 경우)이라 한다. 또한 예측된 레이블 하나에 대해서만 접근이 가능한 경우(쉽게 말하면 결과로 나온 classify 확률들 중 가장 확률값이 높은 하나의 class에 대해서만 알려주는 경우)는 Decision-based threat model이라 한다.  

### Whilte Box Setting 
Whilte Box 공격에서는 모델의 정보(네트워크 구조, 가중치 등)이 모두 공격자에게 드러난 경우다. 
이런 경우엔 입력 값에 대한 gradient를 구할 수 있다. adversarial attack의 경우는 이미 학습이 되어있는 네트워크에서 입력값을 바꾸어 공격자가 의도한 결과를 내도록 하는 것이기 때문에 가중치 값은 그대로 둔 상태에서 입력값의 gradient를 구해고 gradient descent를 하여 loss함수를 최소화 하는 방식으로 공격하게 된다. 

### PGD Attack 
Whilte Box 공격의 대표적인 예로 Projected Gradient Descent(PGD) Attack이 있다. 
이때 PGD는 Gradient Descen를 수행함에 있어서 특정한 범위 제한이 있는 경우 범위 제한을 유지할 수 있도록 
Projection을 시키는 방식으로 update를 하는 방법이다. 
PGD Attack은 다음과 같은 방식으로 진행된다.  

$$PGD \ Attack : x^{t+1} = \prod_{x+S} (x^t + \alpha * sign(\triangledown_x L(\theta, x, y)))$$

$\theta$ : the parameters of a model   
$x$ : the input to the model   
$y$ : the targets associated with $x$    
$J(\theta, x, y)$ : the cost used to train the neural network    
Constraint of perturbation : $||\delta||_{\infty} = \max_i|\delta_i| \le \epsilon$

이때 $L$(Loss)값은 일반적으로 cross entropy loss를 사용한다. 그러면 $\triangledown_x L(\theta, x, y)$는 특정한 모델의 cross entropy loss에 대해 $x$의 gradient를 구하게 된다. 그리고 이러한 loss를 증가시키는 방향으로 update를 시키게 된다. 그렇기 때문에 식에서도 $+$방향으로 update가 진행되는 것이다. 또한 식에 $sign$이 있는 이유는 PGD Attack이 $L_{\infty}$ Attack이기 때문이다. 즉 각각의 픽셀마다 $\alpha$만큼 update가 수행될 수 있도록 만드는 것이다. 다만 $L_{\infty}$ norm 상에서 크기를 입실론만큼 제한했다고 하면 입실론 범위 안에 들어와야 하기 때문에 매step마다 projection($\prod$)을 시켜서 입실론 범위안에 들어올 수 있도록 만드는 것이다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/2d21aa97-7199-4a3d-988d-e5025bfdb607" height="35%" width="35%"></p>


위 사진과 같이 2차원 상의 데이터 분류 모델이 있다고 가정하고 주황색 X의 Loss를 증가시키는 것이 목적이라고 하자. 현재 주황색 X는 class 0으로 분류가 되어 있다. $L_{\infty}$ Attack을 한다고 하면 각각의 축 마다 최대 
$\epsilon$크기만큼 바뀔 수 있기 때문에 위 사진과 같은 범위 안에 데이터가 존재할 수 있다. 
이때 현재 주황색 X를 기준으로 cross entropy loss를 증가시키는 방향으로 각각의 축에 대해 $\alpha$만큼 이동한다고 하면 $x_1$축에 대해서는 오른쪽으로 가고 $x_2$축에 대해서는 아래쪽으로 갈 것이다. 이런 식으로 update를 진행하여 X가 class 1으로 분류되도록 만들게 된다. 

### Carlini-Wagner Attack
Whilte Box 공격의 다른 예로 Carlini-Wagner Attack이 있다. CW Attack도 PGD와 마찬가지로 optimization문제를 해결하는데, 이때
misclassification loss를 추가해줘서 공격을 수행한다. 공격에 사용되는 목적 함수(objective function)은 다음과 같다. 

$$minimize \ \ D(x, x+\delta) + c \cdot f(x + \delta)$$

$$such \ that \ \  x + \delta \in [0, 1]^n$$

여기서 $D$는 원본 이미지 $x$와 Adversarial example $x + \delta$의 거리가 가까워 지도록 하여 눈에 띄지 않도록 하고, $c \cdot f(x + \delta)$는 공격 수행을 위한 loss이다. 또한 이러한 distance loss $D$와 공격 loss $f$중 어떤 것이 더 중요한지 가중치를 정하기 위해 하이퍼 파라미터로 $c$가 존재한다. 그리고 Adversarial example $x + \delta$는 항상 0부터 1사이로 정규화된 값의 범위 안에 들어갈 수 있도록 한다. 예를 들어 픽셀의 값이 0 ~ 255라고 하면 그 값이 0 ~ 1사이로 정규화를 진행한다. 

$L_2$에서 CW Attack은 다음과 같은 목적함수로 구성된다. 

$$minimize \ \ ||\frac{1}{2}(tanh(w) + 1) - x||_2^2 + c \cdot f(\frac{1}{2}(tanh(w) + 1))$$

$$f(x') = \max(\max \left\lbrace Z(x')_i : i \neq t \right\rbrace - Z(x')_t, -k)$$

$w$는 update를 하기 위한 Adversarial example인데, 여기에 $tanh$를 취하고 + 1을 더한 뒤 $\frac{1}{2}$을 곱하면 항상 0~1사이의 값을 취하게 된다. 그렇게 만들어진 Adversarial example을 원본 이미지 $x$와 유사함 값을 갖도록 만들면서 동시에 공격자가 원하는 class로 분류되도록 하기 위해서 $f(\frac{1}{2}(tanh(w) + 1))$를 적용한다. 
이때 $f$는 구체적으로 2번째 수식으로 정의할 수 있는데, $Z$는 로짓값으로 CW Attack의 특징은 최종적으로 나온 확률값을 이용하는 것이 아니라 로짓값을 이용한 차이값을 구한다는 것이 특징이다. 또한 $k$는 얼마나 강력한 perturbation을 넣을 것인지를 결정하는 하이퍼 파라미터이다.  

cw attack은 별도의 projection과 같은 테크닉을 이용해서 perturbation의 크기를 제한하지는 않고 $\delta$값이 애초에 작아질 수 있도록 설정한 뒤에 gradient descent를 이용해서 $w$를 update하는 방식으로 공격을 수행한다.


### Black Box Setting 
Black Box 공격은 공격자가 모델의 내부 파라미터에 대해 모르는 경우를 말하는데 이 중에서도 Decision-based(Hard-label) attack은 가장 확률값이 높은 하나의 class에 대해서만 관찰할 수 있는 경우이다. 이런 상황에서 공격자는 입력 $x$에 대해서 역전파를 하여 gradient를 구할 수 없기 때문에 gradient를 예측하거나 하는 방식으로 공격을 수행한다. 


### Transfer-based Black-box Attack 
Black Box 공격 중 가장 대표적인 것으로 Transfer-based Black-box Attack이 있다. Adversarial example은 모델 사이에서 전송 가능한(transferable) 특징이 있는데 이러한 transferability를 이용한 공격 방법은 다음과 같다. 

1. 공격자가 개인적으로 공격 대상 모델(black-box)와 유사한 대체 모델(substitute model)을 학습한다.
2. 자신의 대체 모델에 대하여 white-box 공격을 수행해 adversarial example을 생성한다.
3. 해당 adversarial example을 공격 대상인 black-box 모델에 넣어 최종적으로 공격을 수행한다.

이러한 Transfer-based Attack이 가능한 이유는 유사한 학습 데이터 세트로 학습한 모델은 유사한 decision boundary를 가지기 때문으로 추측된다. 그래서 이러한 transferability를 잘 활용한 공격 기법은 추가적인 쿼리(query)를 줘서 black-box 모델과 더욱 유사한 surrogate 모델을 만들어 공격하는 것이다. 

또한 transferability가 왜 존재하는가에 대해서 다른 관점의 분석은 Adversarial perturbation을 non-robust feature로 이해할 수 있으며 하나의 데이터 셋은 robust feature와 non-robust feature로 구성된다는 것이다. 이때 유사한 학습 데이터 셋들로 학습을 진행한 다양한 모델들은 generalized non-robust feature를 학습하기 때문에 transferability가 존재할 수 있다는 것이다. 이때 non-robust feature는 작은 크기의 노이즈를 섞었을 때 쉽게 변경될 수 있는 feature를 말하는데 쉽게 변경될 수 있지만 이 또한 일반화가 잘 되어있기 때문에 실제로 모델의 성능을 올리기에 유의미한 역할을 수행할 수 있다는 것이다. 

이러한 Transfer-based Attack에 대한 대표적인 방어기법은 ICLR에 발표된 Ensemble adversarial training을 통해 막을 수 있는데, 이는 특정한 모델을 학습할 때 Adversarial example을 만들고 그것들을 다시 학습 데이터로 활용하여 Ensemble 기법으로 높은 방어율을 보일 수 있다.   


### Decision-based Attack
#### Boundary Attack 
Decision-based Attack의 초기 공격 기법으로는 Boundary Attack이 있다. Boundary Attack은 다음과 같은 방식으로 동작한다. 

Initialization : Boundary Attack을 수행할 때 처음에 adversarial한 상태에서 시작한다. 즉 고양이를 강아지로 분류하게 만들려고 한다면 처음부터 강아지로 분류되는 이미지를 준비하는 것이다. 그렇게 해서 adversarial한 상태는 유지하면서 강아지에서 시작해서 고양이 가까워지도록 하는 것이다. 방법은 아래와 같다. 

1. Gaussian distribution $\eta_i^k \sim N(0, 1)$에서 랜덤하게 노이즈를 sampling하고, 이렇게 sampling한 위치로 이동할 수 있도록 한다. 이때 rescaling과 clipping을 이용해서 이동한 위치가 valid할 수 있도록 한다. 
2. 이동한 위치에서 original example까지 직선을 그었을때 그 직선과 L2 boundary와 만나는 지점으로 projection시킬 수 있도록 한다. 즉 original example이 초기 adversarial example과 같은 거리값을 가지는 경계로 projection을 시키는 것이다.
3. 이후에는 조금씩 original 쪽으로 파고들 수 있도록 해주고 이 과정들을 반복한다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4decd625-c83d-470d-a5c4-c68160714f07" height="45%" width="45%"></p>

랜덤한 노이즈를 섞어서 original image에 가까워질 수 있도록 만들고 이때 original쪽으로 파고들어갈 수 있으면 가고 아니면 해당 방향으로는 이동하지 않는 방식이다. 이러한 과정을 rejection sampling이라 부를 수 있는데 이는 proposal distribution에서 많은 횟수로 sampling 해보고 그 decision 결과에 따라서 accept / reject을 결정하는 방식을 이용하는 것이다. 이는 나중에 나올 HopSkipJumpAttack에 비해 수렴성 보장의 근거가 부족하다는 단점이 존재한다. 

#### Low-Frequency Boundary Attack (LF-BA)
Boundary Attack의 성능을 향상시키기 위한 많은 방법이 제안되었는데 그 중 하나가 LF-BA이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/faa84978-2d35-4ac4-8544-e523b74545f0" height="45%" width="45%"></p>

LF-BA는 기존 Boundary Attack에서 sampling하는 노이즈를 low-frequency데이터 형태로 바꿔주는 방식이다.
즉 노이즈 $\eta$를 sampling하는 과정에서 high-frequency성분이 제외된 random noise $(IDCT_r(N(0,1)^{d \times d})$ 가 사용될 수 있도록 한 것이다. 이떄 정확히 이미지의 크기($d \times d$)만큼 노이즈 데이터를 sampling한 뒤에 거기에서 저주파 성분만 남도록 잘라내고 그 상태에서 다시 IDCT를 수행해서 저주파 노이즈가 sampling될 수 있도록 한다. 

이렇게 만들어진 저주파 노이즈가 섞인 위치에서 안쪽으로 파고들 수 있도록 한다. 이렇게 한 결과 안쪽으로 파고드는 step이 훨씬 더 자주 성공할 수 있었고 결과적으로 original image에 더 빠르게 근접할 수 있었고 이렇게 쿼리 효율성이 증가하였다. 



# Reference
## Web Links
https://www.youtube.com/watch?v=KbelFArAgNQ&list=PLRx0vPvlEmdADpce8aoBhNnDaaHQN1Typ&index=28

## Papers
HopSkipJumpAttack : https://arxiv.org/pdf/1904.02144.pdf   
PGD Attack : https://arxiv.org/pdf/1706.06083.pdf  
Carlini-Wagner Attack : https://arxiv.org/pdf/1608.04644.pdf  
Low Frequency Adversarial Perturbation : http://proceedings.mlr.press/v115/guo20a/guo20a.pdf  

