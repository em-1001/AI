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
적대적공격에는 공격 유형을 Whilte Box 공격과 Black Box 공격으로 나눌 수 있는데 흔히 이를 Threat Model이라 한다. 이는 공격자가 어디까지 알고있는지를 기준으로 공격을 구분한 것인데, 먼저 Whilte Box 공격의 경우 말 그대로 공격자가 모델에 대해 완전히 알고 있는 경우를 말한다. 만약 공격자가 output layer까지만 접근할 수 있다고 하면 이를 Score-based threat model이라 한다. 또한 예측된 레이블 하나에 대해서만 접근이 가능한 경우는 Decision-based threat model이라 한다.  



# Reference
## Web Links
https://www.youtube.com/watch?v=KbelFArAgNQ&list=PLRx0vPvlEmdADpce8aoBhNnDaaHQN1Typ&index=28

## Papers
HopSkipJumpAttack : https://arxiv.org/pdf/1904.02144.pdf  
