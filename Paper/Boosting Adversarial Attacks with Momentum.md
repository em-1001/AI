# Background
## Softmax Function 
Softmax는 다중 클래스 분류 모델에서 일반적으로 마지막 레이어에 사용되는 함수이다. 
그래서 Logit layer $Z(x)$이후에 Softmax를 취하게 된다. 
이때 Softmax를 사용한 결과는 확률분포가 되기 때문에 클래스에 대한 모델의 확률 값을 모두 합하면 1이 된다. 
참고로 Logits에서 가장 높은 값을 갖는 class가 실제 확률 값에서도 가장 크다.  

## Cross-Entropy Loss Function
Cross-Entropy는 마지막 레이어에서 Softmax를 사용하는 분류 문제일 경우 일반적으로 사용하는 비용 함수이다. 

$$CrossEntropy(S, L) = -\sum_{j}L_i log(S_i)$$

$$S \ : \ Softmax \ result$$  

$$L \ : \ one-hot \ vector$$  

softmax 결과에 log를 취하고 one-hot vector를 사용하여 특정 class의 확률값이 높아질 수 있도록 학습한다. 

이러한 Cross-Entropy는 이미지에 대해 어떠한 출력값을 내도록 네트워크의 가중치를 학습시키는 것이고, 
적대적 공격에서는 네트워크의 가중치를 학습하는 대신 이미지를 업데이트한다. 
이런 개념에서 경사하강은 가중치뿐 아니라 입력 데이터(이미지)에 대해서도 수행할 수 있다. 
예를 들어 input x를 원본 class인 y가 아니라 다른 class로 분류하고 싶으면 x에 따른 cross-entropy loss를 구하고 경사를 구한 뒤 loss를 늘리는 방향으로 업데이트를 하면 된다.     

## FGSM(Fast Gradient Sign Method)
위와 같은 공격 방식을 잘 활용한 예가 FGSM이다.
FGSM은 고차원 공간에서 선형적인 행동(linear behavior)은 적대적 예제를 만들기에 충분하다는 아이디어를 활용한 것이고 대표적인 $L_{\infty}$ Attack이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/0a7ecc0a-9e5d-44bd-a885-615b95c194a3" height="70%" width="70%"></p>

FGSM을 White-Box Attack 상황이라 가정하면 입력에 대한 비용 함수의 Gradient를 계산해 한 번 업데이트(Single-step)을 수행하면 된다. loss에 대한 gradient를 구하고 각 방향(sign)으로 입실론 만큼 이동할 수 있도록 하는 것이다. 
즉 각 입력 뉴런(픽셀), 각 차원에 대해 모두 loss가 증가하는 방향으로 입실론만큼 업데이트를 하는 것이다. 
단 FGSM은 단 한번만 gradient를 계산해서 loss가 증가하는 방향으로 업데이트하는 것이기 때문에 의도한 만큼 loss를 높이지 못할 수도 있다. 이러한 점을 보완한 것이 PGD Attack이다. 


# Boosting Adversarial Attacks with Momentum


# Reference
## Web Link
https://www.youtube.com/watch?v=QCgujoTPbmU&list=LL&index=3&t=80s  
## Paper




