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
## MI-FGSM(Momentum Iterative Fast Gradient Sign Method)
본 논문에서는 Non-targeted 공격을 위한 목적 함수를 다음과 같이 정의한다. 

$$\underset{x^{\star}}{arg \max} J(x^{\star}, y),\ s.t.\ ||x^{\star} - x||_{\infty} \leq \epsilon$$

FGSM과 마찬가지로 $L_{\infty}$ norm안에서 입실론 이하의 크기만큼 변경될 수 있도록 하되 loss 함수의 값이 최대가 될 수 있도록 한다. 
이때 매 번 gradient를 새롭게 구해서 업데이트를 수행하는 것이 아니라 앞에서 부터 t개 까지의 기울기 정보를 모두 가진 상태에서 Momentum을 활용해 업데이트를 수행한다. 
이전까지의 기울기 정보를 활용하므로써 poor local maxima에 빠지지 않도록 한다. 전체 알고리즘은 다음과 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/7b0ad2b8-9391-4de7-907a-46ab73b7cae6" height="60%" width="60%"></p>

전체적인 흐름은 현재의 적대적 예제에 대해서 gradient를 구하고 이렇게 구해진 gradient를  $L_1$ distance에 대해 정규화를 수행하고 이전 단계에 사용된 gradient에 뮤 만큼의 factor를 곱해서 그 값을 반영할 수 있도록 한다(6). 여기서 $µ \cdot g_t$를 더해준 부분이 Momentum을 적용한 부분이라고 할 수 있다. 기본적으로 본 논문에서는 이 µ 값을 1로 설정해서 사용한다. 

다음으로 적대적 예제를 sign gradient를 통해 업데이트하고(7) 이 부분은 FGSM과 동일하다. 즉 FSGM을 여러번 수행하는 $L_{\infty}$ PGD Attack과 동일한 방법인데 Gradient를 구하는 과정에서 Momentum을 적용한 것이라 이해할 수 있다. 

또한 현재 알고리즘에서는 PGD에 쓰이는 Clip함수가 보이지 않는데 그 이유는 알파 자체를 $\alpha = \epsilon / T$로 설정했기 때문에 T 번 만큼 반복을 했을 때 각각의 픽셀에 대해서 최대 입실론 만큼만 바뀔 수 있기 때문이다.   

## MI-FGSM for Ensemble of Models 
본 논문에서는 이러한 MI-FGSM이 높은 transferability를 갖기 때문에 특히 black box attack에서 강점을 갖는다고 주장하는데 이러한 transferability를 높을 수 있는 방법이 바로 Ensemble 모델에 대한 MI-FGSM이다.  

이떄 모델의 Ensemble에 대해서 MI-FGSM을 수행하기 위해 Ensemble in logits 메서드를 사용하는데 이는 쉽게 말해 Logits 값의 가중치 합을 구하고 여기에 Softmax Cross-Entropy Loss를 구하고 이 Loss로 업데이트를 진행하는 것이다. 

Non-targeted Attack을 위한 목적 함수는 다음과 같다. 

$$\underset{x^{\star}}{arg \max} J(x^{\star}, y),\ s.t.\ ||x^{\star} - x||_{\infty} \leq \epsilon$$

$$l(x) = \sum_{k=1}^K w_k l_k(x)$$

$$J(x, y) = -1_y \cdot log(softmax(l(x)))$$

loss에 들어가는 logit값 $l(x)$는 Ensemble로 총 K개의 모델을 사용한다고 했을 때 각각의 모델에 대한 logit값의 가중치 합을 구하는 것이다. 
이러한 가중치 합에 softmax를 취한뒤 log를 적용해 cross-entropy를 구하는 것이다. 

전체적인 알고리즘은 다음과 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/253de828-0846-4674-8b86-63db9b0dcbc7" height="60%" width="60%"></p>

위에서 봤던 것과 거의 동일한 방식으로 MI-FGSM을 수행하는 것을 볼 수 있다. 단, K개의 모델을 모두 속일 수 있는 적대적 예제를 만드는 것이기 때문에 총 T번만큼 반복을 수행할 때 현재의 적대적 예제를 각각의 분류모델에 넣어서 logit값들을 모두 구한 뒤 이러한 logit값들의 가중치 합을 구하고 이 가중치 값에 대한 cross-entropy loss를 구해서 loss를 증가시키는 방향으로 untargeted attack을 수행한다.

위 알고리즘에서 보이는 (6), (7)은 앞서 살펴본 MI-FGSM의 (6), (7)과정과 동일하다. 결과적으로 총 K개의 모델을 모두 속일 수 있는 적대적 예제를 만듦으로써 보단 global한 perturbation을 찾을 수 있게 되는 것이고 transferability가 높아지는 것이다. 


## Experiment result
<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/a02b3582-015d-4310-9eaa-c0b229812013" height="80%" width="80%"></p>

실험 결과는 위와 같다. white-box 상황에서는 하나의 모델에 대해 100%에 가까운 공격 성공률을 보이고, black-box 상황에서는 하나의 모델에 대해 좋은 공격 성능을 보인다. 다만 black box 상황에서 ensemble adversarial training을 적용한 모델에 대해서는 낮은 공격 성공률을 보인다. ensemble adversarial training을 적용한 모델은 방어율이 높을 뿐만 아니라 특정한 모델에 대해서만 만든 perturbation은 해당 모델에 대해서 overfitting되는 경향이 있기 때문에 해당 모델이 아닌 다른 모델에 대해서는 낮은 공격 성공률이 나오게 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/bad0fd86-df03-49c6-963c-c825df9cafcd" height="80%" width="80%"></p>

추가적으로 MI-FGSM에서 decay factor인 µ를 1로 설정했을 때 경험적으로 우수한 공격력을 보인다고 한다. 
또한 MI-FGSM은 많은 수의 반복을 거쳐도 높은 공격 성공률을 보였다. 참고로 위 그래프에서 초록색 선은 white box attack이고, 다머지는 다른 모델에 대해 transfer based attack을 한 것이다. 

반복 수가 많아지면 대상 모델로 지정한 모델에 대해서만 overfitting된 적대적 예제가 만들어지기 쉬운데 MI-FGSM은 보다 global한 perturbation을 찾기에 유리하기 때문에 black box 상황에서 많은 수의 반복을 거친다 하더라도 상대적으로 덜 overfitting되고 높은 공격 성공률을 보일 수 있게 되는 것이다. 


## Advantages of MI-FGSM
- 기본적인 FGSM의 경우 단 한번의 step으로 공격을 수행하기 때문에 공격 대상 모델에 대해 under-fitting되는 특징이 있다. 
이 때문에 어느 정도의 transferability는 보이지만, 적대적 예제를 만든 모델 즉, white box 모델에 대해서는 충분히 강력하지 못하다. 

- 반면 I-FGSM(Iterative FGSM)은 과하게 over-fitting되며 poor local maxima에 빠질 수 있다. 
이 때문에 오히려 일반적인 FGSM보다 transfer-based-attack에서 좋은 성능을 내지 못하게 된다. 

- 본 논문에서 제안한 Moemtum을 활용한 MI-FGSM은 poor local maxima에 빠지지 않은 경향이 있다. 
결과적으로 좋은 transferability를 보이며 white-box 공격과 black-box 공격 모두 우수한 성능을 보인다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/8d978b61-fa31-4501-b534-dfff74b5e53d" height="50%" width="50%"></p>

실제 MI-FGSM을 이용해 만들어지는 perturbation들은 코사인 유사도(cosine similarity)가 높다. 
즉 특정한 local maxima에 각각 개별적으로 빠지는 것이 아니라 상대적으로 global한 maxima를 찾아낸다는 것이다. 


## Attacking an Ensemble of Models
<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/92871b65-06b7-4529-bf7f-d2de04f728ee" height="80%" width="80%"></p>

Ensemble 모델을 공격할 때의 결과는 위와 같다. Ensemble 항목은 모든 모델을 이용해 white-box 공격을 한 것이고, Hold-out은 초록색의 공격 대상 모델을 제외한 나머지 3개의 Ensemble을 이용해 적대적 예제를 만들고 black-box 공격을 한 것이다. 

결과를 확인해 보면 white-box 상황에서는 100%에 가까운 공격 성공률을 보이고, black-box 상황에서는 나머지 3개의 Ensemble을 이용해 공격했을 때 제외된 공격 대상 모델에 대해서도 좋은 공격 성공률을 보이는 것을 확인할 수 있다.  

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/f3f0f7f9-52a7-4b47-9919-c32474226566" height="50%" width="50%"></p>

공격 대상이 Ensemble일 때의 결과는 위와 같다. 마찬가지로 Ensemble 상황에서는 모든 모델 즉, 7개의 모델에 대해서 공통적으로 동작하는 적대적 예제를 만들었을 때는 white-box 상황이므로 100%에 가까운 성공률을 보인다. Hold-out 에서는 왼쪽 모델을 제외한 앞서 표에서 확인한 4가지 모델과 현재 표의 2개의 Ensemble모델을 합친 6개의 Ensemble모델에 대해 동작하는 적대적 예제를 만들고 해당 모델에 black-box 공격을 수행한 것이다. 결과를 보면 black-box 상황에서 ensemble adversarial training이 된 모델에 대해서도 좋은 공격 성공률을 보이는 것을 볼 수 있다. white-box도 마찬가지로 ensemble adversarial training을 사용한다 하더라도 입실론만 충분히 크다면 100%에 가깝게 공격이 가능하다.  


## Attack Method 
본 논문은 앞서 살펴본 공격 설정외에 Momentum iterative method가 다양한 공격 설정에 대해서도 적용할 수 있다고 한다. 

예를 들어 untargeted가 아닌 targeted 공격을 위한 기본적인 gradient계산 공식은 다음과 같이 작성할 수 있다. 

$$g_{t+1} = \mu \cdot g_t + \frac{J(x_t^{\star}, y^{\star})}{||\triangledown_x J(x_t^{\star}, y^{\star})||_1}$$

위 처럼 특정한 $y^{\star}$ class로 분류가 되는 방향으로 loss를 구성하여 update를 진행하면 되는 것이다. 

**Targeted MI-FGSM** with an $L_{infty}$ norm bound:

$$x_{t+1}^{\star} = x_t^{\star} - \alpha \cdot sign(g_{t+1})$$

또한 $L_{infty}$ norm bound에 대해서는 앞서 확인한 방법대로 FGSM을 반복해서 공격을 수행하면 된다. 

**Targeted MI-FGM** with an $L_2$ norm bound:

$$x_{t+1}^{\star} = x_t^{\star} - \alpha \cdot \frac{g_{t+1}}{||g_{t+1}_2||}$$

반면 FGSM이 아니라 $L_2$ distance상에서도 perturbation의 크기가 제한될 수 있도록 해서 공격을 수행할 수 있다. 위 식에서는 $L_2$ projection이 사용된 것을 확인할 수 있다. 


# Reference
## Web Link
https://www.youtube.com/watch?v=QCgujoTPbmU&list=LL&index=3&t=80s    

## Paper
https://arxiv.org/pdf/1710.06081.pdf  




