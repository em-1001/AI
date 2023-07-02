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

# Reference
## Web Links
https://www.youtube.com/watch?v=KbelFArAgNQ&list=PLRx0vPvlEmdADpce8aoBhNnDaaHQN1Typ&index=28

## Papers
HopSkipJumpAttack : https://arxiv.org/pdf/1904.02144.pdf  
