# Background
## p - norm
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


## Whilte Box & Black Box
적대적공격에는 공격 유형을 Whilte Box 공격과 Black Box 공격으로 나눌 수 있는데 흔히 이를 Threat Model이라 한다. 이는 공격자가 어디까지 알고있는지를 기준으로 공격을 구분한 것인데, 먼저 Whilte Box 공격의 경우 말 그대로 공격자가 모델에 대해 완전히 알고 있는 경우를 말한다. 만약 공격자가 output layer까지만 접근할 수 있다고 하면 이를 Score-based threat model(output layer에서 classify 결과로 나온 각각의 class에 대한 확률값을 알 수 있는 경우)이라 한다. 또한 예측된 레이블 하나에 대해서만 접근이 가능한 경우(쉽게 말하면 결과로 나온 classify 확률들 중 가장 확률값이 높은 하나의 class에 대해서만 알려주는 경우)는 Decision-based threat model이라 한다.  

## Whilte Box Setting 
Whilte Box 공격에서는 모델의 정보(네트워크 구조, 가중치 등)이 모두 공격자에게 드러난 경우다. 
이런 경우엔 입력 값에 대한 gradient를 구할 수 있다. adversarial attack의 경우는 이미 학습이 되어있는 네트워크에서 입력값을 바꾸어 공격자가 의도한 결과를 내도록 하는 것이기 때문에 가중치 값은 그대로 둔 상태에서 입력값의 gradient를 구해고 gradient descent를 하여 loss함수를 최소화 하는 방식으로 공격하게 된다. 

## PGD Attack 
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


## Black Box Setting 
Black Box 공격은 공격자가 모델의 내부 파라미터에 대해 모르는 경우를 말하는데 이 중에서도 Decision-based(Hard-label) attack은 가장 확률값이 높은 하나의 class에 대해서만 관찰할 수 있는 경우이다. 이런 상황에서 공격자는 입력 $x$에 대해서 역전파를 하여 gradient를 구할 수 없기 때문에 gradient를 예측하거나 하는 방식으로 공격을 수행한다. 


## Transfer-based Black-box Attack 
Black Box 공격 중 가장 대표적인 것으로 Transfer-based Black-box Attack이 있다. Adversarial example은 모델 사이에서 전송 가능한(transferable) 특징이 있는데 이러한 transferability를 이용한 공격 방법은 다음과 같다. 

1. 공격자가 개인적으로 공격 대상 모델(black-box)와 유사한 대체 모델(substitute model)을 학습한다.
2. 자신의 대체 모델에 대하여 white-box 공격을 수행해 adversarial example을 생성한다.
3. 해당 adversarial example을 공격 대상인 black-box 모델에 넣어 최종적으로 공격을 수행한다.

이러한 Transfer-based Attack이 가능한 이유는 유사한 학습 데이터 세트로 학습한 모델은 유사한 decision boundary를 가지기 때문으로 추측된다. 그래서 이러한 transferability를 잘 활용한 공격 기법은 추가적인 쿼리(query)를 줘서 black-box 모델과 더욱 유사한 surrogate 모델을 만들어 공격하는 것이다. 

또한 transferability가 왜 존재하는가에 대해서 다른 관점의 분석은 Adversarial perturbation을 non-robust feature로 이해할 수 있으며 하나의 데이터 셋은 robust feature와 non-robust feature로 구성된다는 것이다. 이때 유사한 학습 데이터 셋들로 학습을 진행한 다양한 모델들은 generalized non-robust feature를 학습하기 때문에 transferability가 존재할 수 있다는 것이다. 이때 non-robust feature는 작은 크기의 노이즈를 섞었을 때 쉽게 변경될 수 있는 feature를 말하는데 쉽게 변경될 수 있지만 이 또한 일반화가 잘 되어있기 때문에 실제로 모델의 성능을 올리기에 유의미한 역할을 수행할 수 있다는 것이다. 

이러한 Transfer-based Attack에 대한 대표적인 방어기법은 ICLR에 발표된 Ensemble adversarial training을 통해 막을 수 있는데, 이는 특정한 모델을 학습할 때 Adversarial example을 만들고 그것들을 다시 학습 데이터로 활용하여 Ensemble 기법으로 높은 방어율을 보일 수 있다.   


## Decision-based Attack
### Boundary Attack 
Decision-based Attack의 초기 공격 기법으로는 Boundary Attack이 있다. Boundary Attack은 다음과 같은 방식으로 동작한다. 

Initialization : Boundary Attack을 수행할 때 처음에 adversarial한 상태에서 시작한다. 즉 고양이를 강아지로 분류하게 만들려고 한다면 처음부터 강아지로 분류되는 이미지를 준비하는 것이다. 그렇게 해서 adversarial한 상태는 유지하면서 강아지에서 시작해서 고양이 가까워지도록 하는 것이다. 방법은 아래와 같다. 

1. Gaussian distribution $\eta_i^k \sim N(0, 1)$에서 랜덤하게 노이즈를 sampling하고, 이렇게 sampling한 위치로 이동할 수 있도록 한다. 이때 rescaling과 clipping을 이용해서 이동한 위치가 valid할 수 있도록 한다. 
2. 이동한 위치에서 original example까지 직선을 그었을때 그 직선과 L2 boundary와 만나는 지점으로 projection시킬 수 있도록 한다. 즉 original example이 초기 adversarial example과 같은 거리값을 가지는 경계로 projection을 시키는 것이다.
3. 이후에는 조금씩 original 쪽으로 파고들 수 있도록 해주고 이 과정들을 반복한다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4decd625-c83d-470d-a5c4-c68160714f07" height="45%" width="45%"></p>

랜덤한 노이즈를 섞어서 original image에 가까워질 수 있도록 만들고 이때 original쪽으로 파고들어갈 수 있으면 가고 아니면 해당 방향으로는 이동하지 않는 방식이다. 이러한 과정을 rejection sampling이라 부를 수 있는데 이는 proposal distribution에서 많은 횟수로 sampling 해보고 그 decision 결과에 따라서 accept / reject을 결정하는 방식을 이용하는 것이다. 이는 나중에 나올 HopSkipJumpAttack에 비해 수렴성 보장의 근거가 부족하다는 단점이 존재한다. 

### Low-Frequency Boundary Attack (LF-BA)
Boundary Attack의 성능을 향상시키기 위한 많은 방법이 제안되었는데 그 중 하나가 LF-BA이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/da8fd154-3b7e-4315-9e28-3b958e358f59" height="45%" width="45%"></p>

LF-BA는 기존 Boundary Attack에서 sampling하는 노이즈를 low-frequency데이터 형태로 바꿔주는 방식이다.
즉 노이즈 $\eta$를 sampling하는 과정에서 high-frequency성분이 제외된 random noise $(IDCT_r(N(0,1)^{d \times d})$ 가 사용될 수 있도록 한 것이다. 이떄 정확히 이미지의 크기($d \times d$)만큼 노이즈 데이터를 sampling한 뒤에 거기에서 저주파 성분만 남도록 잘라내고 그 상태에서 다시 IDCT를 수행해서 저주파 노이즈가 sampling될 수 있도록 한다. 

이렇게 만들어진 저주파 노이즈가 섞인 위치에서 안쪽으로 파고들 수 있도록 한다. 이렇게 한 결과 안쪽으로 파고드는 step이 훨씬 더 자주 성공할 수 있었고 결과적으로 original image에 더 빠르게 근접할 수 있었고 이렇게 쿼리 효율성이 증가하였다. 


### Opt Attack 
Opt Attack은 Boundary Attack보다 좀 더 효율적이라고 알려져있다. Opt Attack은 기존 hard-label black-box attack을 reformulating해서 최적화 문제의 형태를 조금 바꾸었고 이렇게 조금 쉬운 형태로 바꾸고 최적화를 진행한 결과 좀 더 좋은 결과가 나왔다고 한다. 

함수 $g(\theta)$를 새롭게 제안하는데 인자로 받는 $\theta$는 방향성을 알려주는 벡터이다. 
$g(\theta)$는 특정 $\theta$ 방향으로 이동했을 때 가장 가까이에 있는 adversarial example의 거리를 의미한다. 
Opt Attack을 제안한 논문에서는 이 $g$값을 최소화하는 $\theta$를 찾는 문제로 기존의 Decision-based Attack의 문제를 변형하였다. 

$$
\theta^* = arg \underset{\theta}\min g(\theta) 　\ where 　 g(\theta) = arg \underset{\lambda > 0}\min(f(x_0 + \lambda \frac{\theta}{||\theta||}) \neq y_0)
$$

식을 보면 $\frac{\theta}{||\theta||}$로 $\theta$를 norm값으로 나누어서 단위 방향 벡터로 만들어주고 해당 방향으로 이동했을 때 원본 class($y_0$)가 아닌 다른 class가 나오는 최소한의 거리 $\lambda$를 찾고 그 거리값이 함수 $g$의 값이 될 수 있도록 하고 그러한 함수 $g$의 값이 최소화 되도록 하는 $\theta$를 찾는 문제가 된다. 

즉, 어떤 방향으로 이동했을 때 가장 짧은 거리로 적대적 예제를 만들 수 있는지를 찾는 것이다. 
이러한 Opt Attack은 원래 문제를 직접적으로 해결하기 보다는 더 해결하기 쉬운 문제로 변형하여 최적화 문제를 해결한다. 
다만 이후에 나올 HopSkipJumpAttack은 문제를 직접적으로 해결한다. 


### Limited Attack 
Limited Attack은 다음과 같은 상황에서 적용 가능한 공격 메서드를 제안한다. 
1. Query-limited setting : 공격자가 오직 한정된 개수의 쿼리(query)만 날릴 수 있다.
2. Partial-information setting : top-k개의 클래스에 대한 확률(probability)을 알 수 있다.
3. Label-only setting : top-k개의 클래스에 대한 레이블(label)정보만을 알 수 있다.

일반적으로 Label-only setting은 top-1개의 클래스 이름만을 알려주는 경우가 많고 그렇기 때문에 매우 현실적이고 어려운 문제 상황이다. 
Decision-based setting, Hard-label setting과 같은 의미로 사용되는 경우가 많다. 

Limited Attack은 블랙박스 공격 상황에서 기울기(gradient)를 예측하는 방법을 사용한다. 
Limited Attack을 제안한 논문에서는 $L_{\infty}$ 거리로 제한된 공격상황에서 사용 가능한 메서드를 제안했다. 
우선 기울기를 예측하기 위해서는 아래의 NES Gradient Estimate를 사용한다. 
<br>
<br>
<br>
**Algorithm 1** NES Gradient Estimate    
　**Input :** Classifier $P(y|x)$ for class $y$, image $x$    
　**Output :** Estimate of $∇P(y|x)$    
　**Parameters :** Search variance $\sigma$, number of samples $n$, image dimensionality $N$    
$\quad g ← O_n$    
　**for** $i = 1$ **to** $n$ **do**    
$\quad\quad u_i ← N(O_N, I_{N \cdot N})$    
$\quad\quad g ← g + P(y|x + \sigma \cdot u_i) \cdot u_i$  
$\quad\quad g ← g - P(y|x - \sigma \cdot u_i) \cdot u_i$    
　**end for**    
　**return** $\frac{1}{2n\sigma}g$  
<br>
<br>
위 알고리즘에서는 gradient를 예측하기 위해 n번 만큼의 반복을 하는데 반복문에서 매번 gaussian distribution에서 노이즈를 샘플링하고 그러한 노이즈에 델타만큼 곱한 값을 이미지 $x$에 대해 더해주고 그렇게 노이즈가 섞인 이미지에 대해 classifier $P$로 확률값을 구해준다.
이때 해당 노이즈가 확률값을 바꾸는데에 있어서 얼마나 많은 영향을 미쳤는지를 가중치에 따라 구해줄 필요가 있기 때문에 해당 값에 노이즈 값을 곱해준다. 이렇게 얻어진 값 $P(y|x + \sigma \cdot u_i) \cdot u_i$는 gradient의 추정치로 사용할 수 있게 된다. 
마찬가지로 해당 노이즈를 빼준다음에도 동일하게 확률값을 구해서 gradient 추정치를 구해주고 이러한 과정을 반복한다. 
당연히 이러한 과정을 많이 반복할수록 더 정확한 gradient값을 예측할 수 있다. 
물론 실제 공격상황에서는 노이즈를 샘플링할 때마다 매번 쿼리를 날려줘야 하기 때문에 $n$값을 무작적 키우는 것은 불가능하다. 
최종적으로 이렇게 예측된 gradient를 통해서 PGD Attack을 수행할 수 있다. 

이러한 Partial-information setting에서의 공격에 대한 아이디어는 다음과 같다. 
1. 적대적인 상태를 유지하면서 입실론의 최소 크기를 찾는다. 

$$\epsilon_t = \min \epsilon^{'} 　 s.t. \ rank \ \left(y_{adv} | \prod_{\epsilon^{'}} (x^{(t-1)})\right) < k$$

현재 단계의 적대적 예제($x^{(t-1)}$)가 있다고 했을 때 이를 입실론 범위 안으로 projection($\prod_{\epsilon^{'}}$)시켜서 작은 크기의 perturbation을 갖는 적대적 예제가 만들어질 수 있도록 한다. 이때 이러한 적대적 예제에 대해서 class 정보를 구하고 만약 targeted attack이라고 하면 목표로 하는 target class가 상위 $k$개의 class 안에 들어가 있다면 공격 성공으로 보고 그러한 공격 성공을 유지한 상태에서 최대한 작은 입실론을 찾도록 만든다. 

2. 입실론 범위를 유지하면서 target class의 확률값을 최대로 높인다.

$$x^{(t)} = arg \underset{x{'}}\max P(y_{adv} | \prod_{\epsilon_{t - 1}} (x^{'}))$$

이 과정을 거치는 이유는 입실론 범위를 유지하는 perturbation중에서 target class에 대한 확률값을 최대한으로 높여줘야 다시 1번 단계로 돌아갔을 때 입실론을 더욱더 작게 만들 수 있기 때문이다. 이렇게 1번과 2번 과정을 반복적으로 수행하여 최적화 한다. 

결과적으로 이렇게 입실론 범위가 줄어들어야 더욱 눈에 띄지 않고 강력한 적대적 예제를 만들 수 있는 것이다. 
다만 이때 2번에서 확률값을 높이는 적대적 예제를 찾기 위해서는 PGD Attack을 수행할 수 있는데 PGD Attack을 수행하려면 적대적 예제의 gradient를 구해야 하는데 이러한 gradient를 직접적으로 구할 수 없기 때문에 아까 설명한 Gradient Estimate 알고리즘을 사용해서 기울기를 예측하는 것이다. 


# HopSkipJumpAttack
HopSkipJumpAttack은 이전까지의 SOTA(State-of-the-art) Decision-based attack들 보다 더 높은 효율성을 보여주었다. 
HopSkipJumpAttack은 아래와 같은 3가지 step을 반복적으로 수행한다. 

1. Perfrom a binary search to find the boundary
2. Estimate the gradient direction at the boundary point
3. Geometric progression (Step-size search)


<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/66b1b42f-48bb-4740-97d6-9d4bc4ac1b53" height="55%" width="55%"></p>

a step 에서는 boundary에 도달할 때까지 이진탐색을 수행한다. 이때 $x^*$는 original image를 의미하고 $\tilde{x}_t$는 현재의 적대적 예제를 의미한다. 그래서 적대적 예제가 original image와 최대한 가까워질 수 있도록 
적대적 예제와 원본 이미지를 잇는 직선 상에서 최대한 Decision boundary위에 올라갈 수 있도록 이진탐색을 진행한다. 참고로 사진에서 파란색 영역은 adversarial 영역이고 빨간색은 non-adversarial 영역이다. 
이진탐색이후 b step에서는 Decision boundary위에서 gradient의 방향성을 예측할 수 있도록 한다. 이를 통해 사진에서 처럼 direction vector를 얻을 수 있게 되고 해당 방향으로 adversarial example을 이동시킨다. 
이때 이동시킬 때 여전히 파란색 영역 위에 있도록 하기 위해서 정해진 step size만큼 이동하도록 한다. 이러한 과정을 c step의 Geometric progression이라고 한다. 이후부터는 다시 a step부터 과정을 반복한다. 

이렇게 위의 과정을 반복하므로써 $x^* $에 최대한 가까워 질 수 있도록 하는 것이다. 참고로 $\tilde{x}_t$ 에서 $x^* $으로 가까워지도록 이진탐색을 진행할 때는 최대한 Decision boundary에 가깝도록 $x_t$를 구하지만 실제로는 여전히 파란색 영역에 속해있도록 해줘야 한다. 


### Contribution
HopSkipJumpAttack을 제안한 논문에서는 gradient의 direction을 예측하는 참신한 방법을 제안했다. direction을 예측할 때는 Decision boundary위에서 수행할 수 있도록 하고, 이때 model의 decision에만 접근이 가능한 상황에서 예측을 수행할 수 있기 때문에 실제 black-box attack에 효과적이다. 또한 실제로 이진탐색을 진행할 때도 정확히 Decision boundary위에 adversarial example을 올려놓기 어렵기 때문에 실제 구현상에서 발생할 수 있는 오차를 control하기 위한 방법도 제안하였다. 
HopSkipJumpAttack은 이전까지의 다른 매서드와 비교할 때 상당히 query-efficient가 좋다. 나아가서 adversarial attack에 대한 지금까지의 방어기법인 defensive distillation, region-based classifiction, adversarial training, input binarization 등에 대한 효율성에 대해 테스트할 수 있다고 주장한다. 마지막으로 HopSkipJumpAttack은 $L_2$와 $L_{\infty}$ 모두 지원한다. 

### Query-efficient
앞서 언급한 query-efficient가 중요한 이유는 black-box에서의 가장 큰 어려움이 모델에 많은 쿼리를 날려야 한다는 점이다. 즉 쿼리 수가 매우 많이 요구되기 때문에 비용이 큰 것인데, 이전까지 제안된 공격기법들은 상대적으로 많은양의 쿼리가 요구되었다. 추가적으로 다양한 Decision-based Attack은 쿼리를 많이 날리되 매번 날리는 쿼리마다 그 이미지의 차이가 크지 않고 유사한 이미지를 여러번 날리기 때문에 deep learning 운영자의 입장에서는 비슷한 쿼리가 계속 날라오면 이를 일종의 공격으로서 받아들일 수도 있다. 이러한 측면에서 보았을 때도 query의 수가 적을수록 더 유리한 공격이다. 


### White-box Notations 
우선 White-box상황일 때를 가정한다고 하면 아래와 같이 각각의 수학적 용어를 정의할 수 있다. 

Label set(m개의 class) : $[m] = \left\lbrace1, ...., m\right\rbrace$  
Output vector(각각의 class에 대한 결과값) : $y = (F_1(x), ..., F_m(x))$  
The classifier : $C(x) := arg \underset{c \in [m]}\max F_c(x)$  

The objective of the attacker:  

$$
S_{x^{\star}}(x^{'}) :=
\begin{cases}  
\underset{c \neq c^{\star}}\max F_c(x^{'}) - F_{c^{\star}}(x^{'}) & (Untargeted) \\
F_{c^{†}}(x^{'}) - \underset{c \neq c^{†}} \max F_c(x^{'}) & (Targeted)
\end{cases}
$$

Adversarial example : $x^{'}$　　Target class : $c^{†}$　　Original class : $c^{\star}$

공격자의 objective function은 CW loss와 유사한 형태로 Untargeted의 경우 original class에 대해서는 그 출력값( $F_{c^{\star}}(x^{'})$ )을 감소시키고 original class가 아닌 다른 class 중에서 가장 높은 classify결과를 갖는 class에 대해서는 그 출력값( $\underset{c \neq c^{\star}}\max F_c(x^{'})$ )을 높이는 방향으로 공격을 수행한다. 
Targeted의 경우는 의도했던 Target class에 대해서는 출력값( $F_{c^{†}}(x^{'})$ )을 높이고 Target class가 아닌 다른 class중에서 가장 높은 값을 갖는 class는 출력값( $\underset{c \neq c^{†}} \max F_c(x^{'})$ )을 낮추는 방향으로 공격한다. 
실제 공격자는 이러한 방식으로 최종적인 $S_{x^{\star}}(x^{'})$ 값이 커지도록 공격을 수행한다. 

위와 같은 공격은 공격자가 Output vector인 함수 $F$에 접근할 수 있다는 가정이 필요해서 White-box상황인 경우 위와 같은 공격이 가능하다. 


### Decision-based Attack Notations
The decision of whether the attack is a success for not:  

$$
\phi_{x^{\star}}(x^{'}) := sign(S_{x^{\star}}(x^{'})) = 
\begin{cases} 
1 & \mbox{if } S_{x^{\star}}(x^{'}) > 0 \\  
-1 & \mbox{otherwise }
\end{cases}
$$

앞서 확인했듯이 $S_{x^{\star}}(x^{'})$의 값이 0보다 크면 공격성공으로 보고 $S_{x^{\star}}(x^{'})$에 대해서 부호값만 취한 것이 $\phi_{x^{\star}}(x^{'})$가 된다. 
그래서 실제 Decision-based Attack상황에서 공격자가 알 수 있는 정보는 class정보가 바뀌었는지, 바뀌지 않았는지 둘 중 하나이기 때문에 이러한 $\phi$함수에 대해서만 접근이 가능한 것이라고 이해할 수 있다.
즉 공격 성공 여부를 $\phi$함수로 알 수 있다. 

The objective of the attacker in a decision-based attack:

$$\underset{x^{'}}\min d(x^{'}, x^{\star})　such \ that　\phi_{x^{\star}}(x^{'}) = 1$$

실제 Decision-based Attack의 objective function은 위와 같다. 
$\phi_{x^{\star}}(x^{'}) = 1$인 상태를 유지하면서 최대한 Original image와 가까운 Adversarial example을 찾도록 만든다. 

이때 distance metric인 $d$는 $L_p$-nroms을 통해 정의할 수 있다. 

### An Iterative Algorithm for $L_2$ Distance 
$L_2$ Distance에서의 공격 알고리즘을 하나의 식으로 정리하면 다음과 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/5b12f039-a24c-4159-9e81-7a1f9b78aa1e" height="15%" width="15%"></p>

$x_t$는 boundary위에 올라가 있다고 가정을 하고 다음단계의 adversarial example은 다음과 같이 정의할 수 있다. 

$$x_{t+1} = \alpha_t x^{\star} + (1 - \alpha_t) \left\lbrace x_t + \zeta_t \frac{\triangledown S_{x^{\star}}(x_t)}{||\triangledown S_{x^{\star}}(x_t)||_2} \right\rbrace$$

boundary위에 올라가 있는 $x_t$에서 gradient direction을 구하는데 이때 gradient라고 하는 것은 공격자의 원래 objective라 할 수 있는 함수 $S$에 대한 gradient를 말한다. 
이러한 gradient에 대해서 $L_2$ norm값으로 나누어( $\frac{\triangledown S_{x^{\star}}(x_t)}{||\triangledown S_{x^{\star}}(x_t)||_ 2}$ ) 주어서 단위 벡터로 만들고 해당 방향으로 $\zeta_t$만큼 이동하도록 한다. 
이때 $\zeta_ t$는 step size로 $x_t$에서 얼마만큼 이동해서 $\tilde{x}_ {t+1}$를 얻을 수 있는지를 결정한다. 결론적으로 $\tilde{x}_ {t+1} = x_t + \zeta_t \frac{\triangledown S_{x^{\star}}(x_t)}{||\triangledown S_{x^{\star}}(x_t)||_ 2}$ 가 된다. 이후 $\tilde{x}_ {t+1}$에서 $x^{\star}$에 가까워질 수 있도록 binary search를 진행하기 때문에 $\tilde{x}_ {t+1}$와 $x^{\star}$ 사이의 interpolation ( $\alpha_ t x^{\star} + (1-\alpha_t)\tilde{x}_ {t+1}$ )으로 표현할 수 있다. 이때 최대한 $\alpha_t$값이 커질 수 있도록 binary search를 진행하므로써 original image에 가까운 $x_{t+1}$을 얻을 수 있다. 

그렇다면 이러한 과정을 반복했을 때 $x^{\star}$에 충분히 가까워질 수 있는가에 대한 답은 Decision boundary가 어떻게 형성되었는지에 따라서 여부가 달라진다. 단 본 논문에서는 최소한 local minima로 수렴할 수 있다고 아래와 같이 수렴성을 증명한다. 

$$
\begin{aligned}
r(x_t, x^{\star}) :&= \cos \angle(x_t - x^{\star}, \triangledown S_{x^{\star}}(x_t)) \\  
&= \frac{ \left\langle x_t - x^{\star}, \triangledown S_{x^{\star}}(x_t) \right\rangle }{ ||x_t - x^{\star}||_ 2 ||S_{x^{\star}}(x_t)||_ 2 }
\end{aligned}
$$

수렴성을 증명하기 위해서 함수 $r$을 정의하는데 이는 $x_t$에서 $x^{\star}$를 뺀 vector와 gradient $S$의 방향성이 얼마나 유사한지를 평가하는 함수이다. 위 그림을 함께 보면 $x_t - x^{\star}$를 계산한 vector와 gradient direction vector의 방향성이 비슷하면 $x_t$가 충분히 수렴했다는 것을 의미함을 알 수 있다. 이렇게 방향성이 같아지는 $x_t$ point를 stationary point라고 하고 여기서는 위 알고리즘을 반복하여 수행해도 여전히 같은 자리로 가게 된다. 이렇게 되면 최적화가 왼료되었다고 볼 수 있고 이러한 stationary point는 $r(x_t, x^{\star}) = 1$이 되는 지점이 된다. 


### Lipschitz-Continuous Function
앞서말한 $r(x_t, x^{\star}) = 1$이 가능한 것은 립시츠 연속 함수라는 조건일 때 가능한 것이다. 립시츠 연속 함수는 연속적이고 미분 가능하며 어떠한 두 점을 잡아도 기울기가 $K$이하인 함수를 말한다. 
쉽게 말하면 급격한 변화 없이 ($K$ 만큼) 전반적으로 물 흐르듯 완만한 기울기를 가지는 함수 형태이다. 

$K$-Lipschitz 함수는 다음과 같이 정의된다. 

$$\frac{|f(x_1) - f(x_2)|}{|x_1 - x_2|} \leq K　 for \ all \ x_1 \ and \ x_2$$

만약 파라미터에 대한 비용함수가 립시츠 연속이라면 상대적으로 안정적인 수렴이 가능하다. 

HopSkipJumpAttack의 수렴성(Convergence)증명은 본 논문에 자세히 설명되어 있고 증명이 완료된 내용은 다음과 같다. 

**[Theorem]** 만약 함수 $S$가 Lipschitz-Continuous gradient라고 하면, $L_2$에서의 HopSkipJumpAttack은 적절한 step size를 사용했을 때 saddle point나 local minimum로 수렴할 수 있다. 
이때 step size($\zeta_t$)는 다음과 같이 정의된다.

$$\zeta_t = ||x_t - x^{\star}||_2 t^{-q} 　for \ some　q \in (\frac{1}{2}, 1)$$


### Gradient Estimation 
이렇게 적대적 예제를 update했을 때 stationary point에 수렴이 가능하다는 것은 증명이 되었는데 한 가지 문제가 있다. 실제 black-box attack 상황에서는 함수 $S$의 gradient($\triangledown S$)에 접근할 수 없다는 것이다. 
따라서 decision boundary위에서 gradient direction값을 예측해야하는데 해당 아이디어는 다음과 같다. 

The approximated direction of the gradient $\triangledown S_{x^{\star}}(x_t)$ :   

$$\tilde{\triangledown S}(x_t, \delta) := \frac{1}{B}\sum_{b=1}^B \phi_{x^{\star}}(x_t + \delta u_b)u_b$$

limited attack의 공식과 유사한데 본 논문에서는 monte carlo method를 이용해서 gradient의 방향성을 예측할 수 있다고 한다. 공격자 입장에서는 함수 $\phi$에 대해서는 접근이 가능하므로 현재 decision boundary위에 올라가 있는 $x_t$가 있다고 했을 때 여기에 random unit인 $u_b$를 뽑아서 해당 random noise를 $\delta$만큼 반영할 수 있도록 하고 해당 위치에서 공격 성공인지 아닌지에 대한 값을 아래 사진 처럼 +1 또는 -1로 얻을 수 있을 것이다. 따라서 어떠한 방향으로 갔을 때 공격을 성공한다는 정보를 이용할 수 있고 공격 성공시 $\phi$를 거쳐서 나온 +1값에 다시 $u_b$를 곱해주어 해당 노이즈를 반영할 수 있도록 해준다. 이러한 과정을 $B$만큼 반복을 하고 해당 값들의 평균을 구해서 gradient의 direction값을 예측할 수 있다.


실제 논문에서 증명된 결과에 따르면 $x_t$가 $S(x_t)$의 boundary위에 올라가 있을 때 direction 예측값인 $\tilde{\triangledown S}(x_t, \delta)$은 실제 $\triangledown S(x_t)$의 direction값과 같아질 수 있다고 한다. 

$$\lim_{\delta \to 0} \cos \angle \left(\mathbb{E} [\tilde{\triangledown S}(x_t, \delta)], \triangledown S_{x^{\star}}(x_t)\right) = 1$$

이를 수식으로 표현하면 위와 같고 이 때의 조건은 $x_t$가 정확히 boundary위에 있어야 하며 noise 크기인 $\delta$가 0에 가까워져야 한다는 것이다. 

다만 실제 구현상에서는 문제가 발생할 수 있는데 binary search를 할 때 정확히 boundary위에 올리는 것은 불기능하기 때문에 실제 공격상황에서는 이론적으로 증명된 값과 같이 $\delta$가 0에 가까운 값이면 안된다. Iterative Algorithm for $L_2$ Distance 사진을 예시로 $x_t$가 정확히 boundary에 위치하지 않고 그 위인 파란색 영역에 위치했다고 하고 $\delta$도 0에 가까운 매우 짧은 값이라고 하면 $\phi$함수로 판단하기에 빨간색으로 향하는 vector도 파란색 영역안에 속하게 되어서 잘못 판단하게 된다. 그래서 논문에서는 $\delta$값으로 $\delta = \sqrt{d}\zeta$를 사용한다. 여기서의 $\zeta$값은 앞서 다른 step size가 아니고 binary search를 수행할 때 사용했던 정밀도(threshold)를 말한다.  

### Variance Reduction   
또한 $\delta = \sqrt{d}\zeta$와 같이 델타를 크게 만들었다 하더라도 여전히 빨간색 향하는 vector이면서 파란색에 속해버리는 vector가 있을 수 있는데 이를 해결하기 위해 Variance Reduction기법을 이용한다. 

$$
\begin{aligned}
Baseline \bar{\phi} &:= \frac{1}{B} \sum_{b=1}^B \phi(x^{'} + \delta u_b) \\
\widehat{\triangledown S}(x^{'}, \delta) &:= \frac{1}{B-1} \sum_{b=1}^B (\phi(x^{'} + \delta u_b) - \bar{\phi})u_b
\end{aligned}$$

Baseline 변수인 $\bar{\phi}$는 $\phi$값의 평균을 구한 것이다. 이렇게 만든 baseline 변수를 gradient estimate과정에서 매번 빼줄 수 있도록 한다. 실제로 이렇게 예측값을 구했을 때 variance가 감소하는 효과를 보였다. 논문에서 증명된 내용을 보면 baseline을 빼줌으로써 variance가 다음과 같이 감소한다고 한다. 

$$Var(\widehat{\triangledown S}) = Var(\tilde{\triangledown S}) \left\lbrace 1 - Const \cdot B^{-2p} \right\rbrace$$


### Overall Algorithm
**Algorithm 1** Binary Search    
**Require** : Samples $x^{'}, x,$ with a binary function $\phi$, such that      
$\quad\phi(x^{'}) = 1, \phi(x) = 0$, threshold $\theta$, constraint $\ell_p$.    
**Ensure** : A sample $x^{''}$ near the boundary.        
　Set $\alpha_l = 0$ and $\alpha_u = 1$.  
　**while** $|\alpha_l - \alpha_u| > \theta$ **do**.   
　　Set $\alpha_m ← \frac{\alpha_l + \alpha_u}{2}$.    
　　**if** $\phi(\prod_{x, \alpha_m} (x^{'})) = 1$ **then**   
　　　Set $\alpha_u ← \alpha_m$. <br>
　　**else**         
　　　Set $\alpha_l ← \alpha_m$.        
　　**end if**     
　**end while**    
　Output $x^{''} = \prod_{x, \alpha_u}(x^{'})$.   
 
<br>
<br>
<br>

# Reference
## Web Links
https://www.youtube.com/watch?v=KbelFArAgNQ&list=PLRx0vPvlEmdADpce8aoBhNnDaaHQN1Typ&index=28    
interpolation : https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%EB%B3%B4%EA%B0%84%EB%B2%95     
monte carlo method : https://en.wikipedia.org/wiki/Monte_Carlo_method


## Papers
HopSkipJumpAttack : https://arxiv.org/pdf/1904.02144.pdf   
PGD Attack : https://arxiv.org/pdf/1706.06083.pdf  
Carlini-Wagner Attack : https://arxiv.org/pdf/1608.04644.pdf  
Low Frequency Adversarial Perturbation : http://proceedings.mlr.press/v115/guo20a/guo20a.pdf    
Opt Attack : https://arxiv.org/pdf/1909.10773.pdf    
Limited Attack  : https://arxiv.org/pdf/1804.08598.pdf  
