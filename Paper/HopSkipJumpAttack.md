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

#### Whilte Box Setting 
Whilte Box 공격에서는 모델의 정보(네트워크 구조, 가중치 등)이 모두 공격자에게 드러난 경우다. 
이런 경우엔 입력 값에 대한 gradient를 구할 수 있다. adversarial attack의 경우는 이미 학습이 되어있는 네트워크에서 입력값을 바꾸어 공격자가 의도한 결과를 내도록 하는 것이기 때문에 가중치 값은 그대로 둔 상태에서 입력값의 gradient를 구해고 gradient descent를 하여 loss함수를 최소화 하는 방식으로 공격하게 된다. 

#### PGD Attack 
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

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/f7163c11-135b-492a-b160-1eeabe57755c" height="40%" width="40%"></p>


위 사진과 같이 2차원 상의 데이터 분류 모델이 있다고 가정하고 주황색 X의 Loss를 증가시키는 것이 목적이라고 하자. 현재 주황색 X는 class 0으로 분류가 되어 있다. $L_{\infty}$ Attack을 한다고 하면 각각의 축 마다 최대 
$\epsilon$크기만큼 바뀔 수 있기 때문에 위 사진과 같은 범위 안에 데이터가 존재할 수 있다. 
이때 현재 주황색 X를 기준으로 cross entropy loss를 증가시키는 방향으로 각각의 축에 대해 $\alpha$만큼 이동한다고 하면 $x_1$축에 대해서는 오른쪽으로 가고 $x_2$축에 대해서는 아래쪽으로 갈 것이다. 이런 식으로 update를 진행하여 X가 class 1으로 분류되도록 만들게 된다. 

#### Carlini-Wagner Attack



# Reference
## Web Links
https://www.youtube.com/watch?v=KbelFArAgNQ&list=PLRx0vPvlEmdADpce8aoBhNnDaaHQN1Typ&index=28

## Papers
HopSkipJumpAttack : https://arxiv.org/pdf/1904.02144.pdf  
