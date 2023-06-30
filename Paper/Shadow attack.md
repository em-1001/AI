## Shadow attack
이 논문은 뉴럴 네트워크를 공격하는 새로운 공격 유형으로 Shadow attack(그림자 공격)을 제안한다. 
Shadow attack의 특징은 다음과 같다. 

1. Imperceptibility : 정상적인 이미지 처럼 보임
2. Misclassification : 타겟 클래스로 오분류하도록 유도한다.
3. Strongly certified : 높은 인증 반경(certificate radius)을 가진다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/7cdd8a2a-4b36-4412-bfc0-c58a70e0c305" height="80%" width="80%"></p>

Strongly certified의 경우 Adversarial examples을 방어기 위한 기법으로 certified defense가 있는데 이러한 방어 기법을 뚫는 기법이라고 할 수 있다. 
즉 이 논문은 제목인 Breaking Certified Defenses Semantic Adversarial Examples with Spoofed Robustness Certificates에서도 알 수 있듯이 적대적 공격에 대한 방어 기법으로
Robustness Certificates와 같은 기법들이 나왔는데 이런 방어 기법을 다시 뚫는 기법을 Shadow attack으로 제안하는 것이다. 

## Background

### Adversarial training

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/523ba308-0597-46ee-be1b-0c5f7660145f" height="80%" width="80%"></p>

우선 적대적 예제라는 것은 인간의 눈에 띄지 않게 약간 변형된 데이터로, 뉴럴 네트워크의 부정확한 결과를 유도하게 된다. 
이러한 적대적 예제를 이용한 공격에 가장 효과적이라고 알려진 방어 기법은 Adversarial training이다. 
적대적 학습이라고도 불리며 Adversarial training은 뉴럴 네트워크를 공격으로부터 Robustness하게 만들기 위해 Adversarial example을
학습 데이터로 이용하는 방법이다. 

$$\underset{\theta}\min \underset{(x, y)\in X}E \left[\underset{\delta \in S}\max \ L(x + \delta; y; \theta)\right]$$


Adversarial training은 뉴럴 네트워크를 학습시키는 과정에서 batch단위로 이미지 데이터를 먼저 Adversarial example로 만든 다음에 그 example을 다시 원래의 class로 분류할 수 있도록 학습시키는 방법이다. 

위의 수식에서 $x, y$의 train data set이 있을 때, perturbation을 의미하는 $\delta$를 perturbation 제약 조건인 $S$범위 안에서 
$x$에 더해 만들어진 Adversarial examples가 loss값을 증가시키는 방향으로 만든 후 다시 원래의 class로 분류하도록 $\min$을 적용해 준다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/73cf486a-3666-4ee6-b3e8-04c0c2c60c2c" height="60%" width="60%"></p>

robust해 질 수는 있다. 다만 Adversarial example을 만드는 부분이 해결하기 쉽지 않은 문제라는 것이다. 
애초에 충분히 강한 Adversarial example을 만들어아 네트워크가 학습하면서 더욱 강해질 수 있는데, 애초에 그런 강력한 example을 찾는것이 어렵다는 것이다.
그러나 PGD를 이용하면 충분히 강력한 Local solution을 찾을 수 있고, 이를 이용해 Adversarial example을 만들어 학습시키면 충분히 경험적으로 강력한 모델을 만들 수가 있게된다.
추가적으로 PGD를 이용할 때 random start와 같을 것을 사용하면 더 좋은 example을 만들어낸다는 것을 실험적으로 보여주었다.  

하지만 완벽한 max는 찾기가 어려운데, 위 사진과 같이 현재 지점에서 경사하강으로 판단했을 때, loss를 max로 만들려면 파란색 쪽으로 이동하는 것이 맞겠지만 실은 초록색이 Global solution이다. 
따라서 이처럼 Local solution으로 만들어진 Adversarial example로 학습할 경우 충분히 강력한 모델이 만들어지지 않을 수 있다는 것이다.  
그러므로 여전히 더 강력한 공격이 나온다면 그 공격에 의해서 모델이 뚫릴 수가 있다고 주장한다. 

### Certified Adversarial Robustness
래서 Adversarial training의 이런 한계를 해결할 수 있는 수학적 모델로 Certified Adversarial Robustness가 제안되었다. 이는 입력 이미지가 주어졌을 때 특정한 크기의 $L_p - boundary$ 안에서 Adversarial example이 만들어질 수 없도록 수학적으로 보장하는(guaranteeing)방어 기법 유형이다. 

#### Randomized smoothing
이런 Certified Adversarial Robustness에서 가장 각광받는 방법이 Randomized smoothing이다. 
아이디어는 모델을 학습 시킬 때 기존 이미지가 있으면 노이즈를 조금 씩 섞어서 학습을 진행한다.  
이때 미리 정해좋은 특정 variance $\sigma^2$ 를 가지도록 노이즈를 만들고 이렇게 하면 정해진 variance크기에 따라 노이즈가 들어가게 된다. 

이후에 이렇게 학습된 모델로 평가를 진행할 때는 별도의 함수 $g(x)$를 이용하는데 이 $g$는, 
inference를 할 때 하나의 이미지가 있으면 그 이미지를 중심으로 Gaussian 노이즈를 만들어 섞은 다음 이를 이용해서 이전에 
학습시켰던 모델인 $f$에 넣은 결과를 내놓는다. 그렇게 해서 $f$가 가장 많은 빈도로 내놓는 class결과를 이용해 inference를 하겠다는 것이다. 
즉 노이즈를 이용해 모델을 학습시킨 다음에 사용할 때도 노이즈를 섞어서 노이즈가 어떻게 평가가 되는지를 기반으로 inference를 하겠다는 것이다. 

실제로 $g$를 이용할 때는 **Monte Carlo Algorithm**을 이용하고, 이렇게 했을 때의 결과는 $g$가 $L_2$ 노름 
$\sigma \cdot \phi^{-1}(p)$ radius안에서 provably robust하다고 할 수 있다. 
이는 수학적으로 논문에서 증명된 내용이고 이 공식에 따라 특정한 이미지가 특정 범위안에서는 이동을 해도 Adversarial example이 만들어지지 않느다는 것을 보장할 수 있게된다. 
참고로 $\phi^{-1}$는 어떠한 정규분포가 있을 때의 inverse CDF 값이다. 

$$r = \sigma \cdot \phi^{-1}(p)$$

예를 들어서 어떤 이미지가 80%의 확률로 고양이라 분류되었다고 하고, 학습 시킬때의 $\sigma$를 0.5로 두었다고 할 때, 
논문의 공식에 대입하면 $p = 0.8, \sigma = 0.5 \to \sigma \cdot \phi^{-1}(p) \approx 0.5 \cdot 0.842 = 0.421$ 가 되어 
이 0.421의 $L_2$ boundary안에서는 무조건 고양이로 분류됨을 보장할 수 있다. 

이번에 다루는 논문인 이 Randomized smoothing 기법을 뚫는 아이디어에 대해 다루게 된다. 

## Shadow Attack for Randomized Smoothing
본 논문의 핵심 아이디어는 모든 perturbation의 크기가 특정한 $L_2$ boundary로 국한될 수 있냐는 것이다. 애초에 인간의 눈에 잘 띄지 않으면서 이런 decision boundary로 부터 멀리 떨어져 있도록($R \ge r$) Adversarial example을 만들 수도 있기 때문이다. 
이런 관점에서 보면 Certified Defense는 수학적으로는 의미가 있지만 실제 방어 기법으로는 여전히 부족함을 지적하는 논문이라 할 수 있다. 

논문에서 제안하는 테크닉은 아래와 같다. 아래 식이 논문에서 제안하는 objective function이 된다. 

$$\underset{\delta}\max L(\theta, x + \delta) - \lambda_c C(\delta) - \lambda_{tv} TV(\delta) - \lambda_s Dissim(\delta)$$

#### Dissimilar
총 4개의 항으로 구성되어 있는데 우선 가장 마지막 항인 Dissimilar항 부터 살펴볼 것이다. 이 부분은 쉽게 얼마나 같지 않은지를 평가하는 term이라고 볼 수 있다. 실제 용도는 최대한 같아지도록 만들기 위해 이런 Loss function을 사용한다. 그래서 $\lambda_s Dissim(\delta)$가 의미하는 바는 하나의 픽셀이 있을 때 그 픽셀의 각 RGB채널이 얼마나 다른가를 의미하고 RGB채널의 값들이 비슷하도록 만든다. 논문의 주제가 그림자 공격인 이유도 GrayScale(Shadow)과 유사한 형태로 perturbation을 넣으려하기 때문인 것이다. 실제 RGB표를 보면 알 수 있듯이 R, G, B 각각의 값이 서로 같으면 white, gray, black계열의 색들이다. 

$$Dissim(\delta) = ||(\delta_R - \delta_G)^2, (\delta_R - \delta_B)^2, (\delta_G - \delta_B)^2||_2$$

위 식을 보면 각 채널에 대해서 비슷한 perturbation이 들어가도록 만드는 것을 볼 수 있다. 

```py
def get_sim(t: torch.Tensor) -> torch.Tensor:
	return ((t[0] - t[1]) ** 2 + (t[1] - t[2]) ** 2 + (t[0] - t[2]) ** 2).norm(p=2)
```

#### Smoothing
이제 다음으로는 Smoothing하는 term인 $\lambda_{tv} TV(\delta)$이다. 이 부분에서는 만들어지는 perturbation이 더욱 자연스럽게 보이도록 total variation을 진행한다. total variation은 Smoothing기법의 일환으로 노이즈를 제거하기 위한 목적으로 자주 사용된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/2212aa35-7cef-43fa-8728-a8887bb840c8"></p>

Adversarial example도 사실은 일종의 노이즈라 볼 수 있기 때문에 원래 이러한 total variation은 방어 기법으로도 사용하는 논문도 있다. 다만 이 논문에서는 특이하게 Adversarial example에서의 perturbation을 더욱 자연스럽게 보이게하기 위한 용도로 사용하였다. 

$$Anisotropic \ \ TV(\delta) = \sum_k \sum_{i \ j}\left(\sqrt{(D_h\delta)_ {i \ j}^2} + \sqrt{(D_v\delta)_{i \ j}^2}\right)$$

논문에서 사용하는 total variation은 Anisotropic TV로 이 값이 의미하는 바는 인접한 픽셀끼리의 차이이다. 

```py
def get_tv(t: torch.Tensor) -> torch.Tensor:
	x_wise = t[:, :, 1:] - t[:, :, :-1]
	y_wise = t[:, 1:, :] - t[:, :-1, :]
	return (x_wise * x_wise).sum() + (y_wise * y_wise).sum()
```

위 코드는 바로 옆 픽셀들 간의 차이를 한번에 계산하도록 한 것이다. 
이렇게 인접한 필셀들 간의 값을 유사하게 만들기 때문에 variation값이 감소하게 되는 것이다. 

#### Color Regularizer
다음은 Color Regularizer term인 $\lambda_c C(\delta)$이다. 이 부분은 perturbation의 크기 자체가 커지지 않도록 해주는 것이다. 이때 각 색상채널별로 평균값을 따로 구해서 그 평균값이 작아지도록 만든다. 
앞선 TV에서는 각 색상채널별로 variation을 줄이기 때문에 엄밀히 말하면 perturbation의 크기와는 상관이 없다. 그렇기에 perturbation의 크기가 작아지도록 하는 term이 필요한 것이다.  

$$C(\delta) = ||Avg(|\delta_R|), Avg(|\delta_G|), Avg(|\delta_B|)||_2^2$$

```py
def get_color(t: torch.Tensor) -> torch.Tensor:
	return t.abs().mean([1, 2]).norm() ** 2
```





# Reference
### Web links
https://www.youtube.com/watch?v=D1j3QiXPRag&list=LL&index=7&t=1680s  

### Papers
Randomized smoothing : https://arxiv.org/pdf/1902.02918.pdf  
Shadow attack : https://arxiv.org/pdf/2003.08937.pdf     

