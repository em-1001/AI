# Background
## Adversarial Training
Adversarial Training은 모델을 학습시키는 방법 중 하나로 모델을 adversarial example을 이용해 학습시킴으로서 모델이 더욱 robust해지도록 하는 것이다. 이 Adversarial Training을 사용하는 예시로 PGD learning이 있다.  
PGD Attack으로 만들어진 adversarial example에 대해 다시 원본 class로 분류하도록 학습시키면 실제 공격이 들어왔을 때 robust하게 동작한다는 것이다. 

Adversarial Training의 일반적인 Objective function은 다음과 같다. 

$$\underset{\theta}\min p(\theta), \ where \ p(\theta) = E_{(x, y) \sim D} [\underset{\delta \in S}\max L(\theta, x + \delta, y)]$$

$\underset{\delta \in S}\max L(\theta, x + \delta, y)$ 이 부분은 PGD를 통해 계산이 되는데 adversarial example $x + \delta$는 $\delta \in S$ boundary조건을 만족하면서 기존 모델 $\theta$에 대해서 최대한 Loss를 증가시키는 방향으로 이미지 업데이트가 된다. 이렇게 Loss를 최대화 하는 adversarial example에 대해서 다시 Loss를 최소화 시키는 방향으로 Adversarial Training을 하는 것이다. 즉 adversarial example을 만들고 이 adversarial example에 대해서 다시 원래의 class로 분류하도록 Loss를 낮추는 방식으로 학습을 진행하는 것이다.     
<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/2c173e6d-0405-4811-b01c-82e3ceb98512" height="70%" width="70%"></p>

이러한 방식으로 학습을 진행하면 결과적으로 위 빨간색 선과 같은 decision boundary를 얻게 되는 것이다. 


# Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
꼭 adversarial example을 통한 공격이 아니더라도 인공지능을 활용하다보면 데이터 손상과 Perturbation에 대해 다뤄야하는 경우가 많다. 이제부터 본격적으로 보다 일반적인 데이터 손상에 대해 다루는 방법에 대해 알아볼 것이고 첫 번째는 Benchmarking Neural Network Robustness to Common Corruptions and Perturbations이다. 
여기서 말하는 Corruptions과 Perturbations은 adversarial example에서의 의미와는 다르다. Perturbations은 공격자가 의도적으로 만든 Perturbations이 아니라 자연적으로 발생할 수 있는 Perturbations을 의미하고 Corruptions은 이미지의 조도, 날씨 등의 영향으로 인한 Corruptions을 의미한다.   

우선 Deep-learning model의 robustness는 크게 2가지 카테고리로 연구가 진행 중이다. 첫 번재는 Worst-case adversarial perturbation으로 이는 말 그대로 모델의 Loss를 증가시키는 adversarial perturbation을 어떻게 만들 수 있는지, 그리고 어떻게 robustness하게 만들 수 있는지에 대해 연구하는 분야이다. 그래서 이는 앞서 다룬 adversarial example에 대한 연구라 할 수 있고 기존 연구는 적대적 공격에 대한 robustness에 초점이 맞추어졌었다. 두 번째는 본 논문에서 초점을 두고 있는 것으로 Common corruptions and perturbation이다. 이는 일반적인 상황에서 발생할 수 있는 corruptions과 perturbation에 대한 robustness를 연구하는 분야이다.                


### Corruption Dataset
<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/aae673d7-1f8f-462e-9b51-2d49cef75bb1" height="70%" width="70%"></p>

예시들을 보면 간단한 노이즈부터 blur현상 Saturate 등이 있다. 이 Robustness Benchmarks 논문에서는 우선 어떻게 이런 손상된 데이터를 만들 수 있는지부터 언급한다. 
논문에서 제안하는 것은 ImageNet-C benchmark로 15개의 서로다른 corruption을 갖는 이미지를 추가로 제공해서 validation imageset을 구성하는 것이다. 

다만 이러한 연구분야에서 가장 중요한 점이 Evaluation시에 네트워크는 ImageNet-C에 대한 dataset을 직접적으로 학습데이터로 사용해서는 안된다는 것이다. 기존에 가지고 있는 학습 dataset에 대해서 적절하게 수정해서 학습에 사용하고 직접적으로 ImageNet-C와 같이 손상된 dataset으로 학습을 진행하지 않으면서 test 시에는 ImageNet-C에 대해 좋은 성능을 내는 것이 목표인 것이다.             

그래서 Evaluation metric으로는 Corrupted Error를 제안한다. 이는 기존 base 모델인 AlexNet과 비교해서 얼마나 우수한지를 평가한다. 

$$CE_c^f = \left(\sum_{s=1}^5 E_{s,c}^f \right) / \left(\sum_{s=1}^5 E_{s,c}^{AlexNet} \right)$$

$c$ : Corruption type  
$s$ : Severity (1 $leq$ s $leq$ 5)  
$E_{s,c}^f$ : Top-1 error of a network $f$   


imagenet dataset으로 학습된 AlexNet이 있다고 했을 때 AlexNet과 정확도가 같다면 1이 나올 것이고, 만약 method를 적용하여 만든 모델이 더 에러율이 낮다면 CE는 1보다 작은 값이 될 것이다. $c$는 어떤 종류의 손상을 가져올지를 정하고, $s$는 손상 정도를 얼마나 심하게 할지를 정한다. 그래서 모든 Corruption type과 Severity case에 대해 Top-1 error의 평균 값을 구하고 이를 AlexNet과 비교해 상대적으로 얼마나 우수한지를 평가한다.  

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/721766f6-6255-4e58-8ca9-c9d5a284dfce" height="70%" width="70%"></p>

Severity는 위처럼 5단계로 있어서 Severity가 낮은 값부터 높은 값까지 종합적으로 평가하기에 용이하다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/38ef3876-c11b-40f8-9757-7759686c425f" height="70%" width="70%"></p>

그리고 발생할 수 있는 데이터 손상 타입으로는 위와 같이 크게 4가지로 나눈다. 


### Perturbation Dataset
본 논문에서는 Perturbation에 대해서도 Robustness Benchmark를 제안한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e927b349-8694-4d39-885f-8d205fcd8365" height="35%" width="35%"></p>

위 사진은 time series에 따른 이미지 변화이다. 그래서 각각 서로 다른 Perturbation을 서서히 가하는 것으로 볼 수 있다. 
그래서 이러한 time series sequence는 30 frame까지 구성되고 각 frame은 이전 frame에 대해 Perturbation을 가한 것으로 볼 수 있다. 

여기서 제안하는 Evaluation metric은 Flip Probability이다. 이는 쉽게 말해 이전 프레임과 현재 프레임이 서로 다른 클래스로 분류되는 지를 평가하는 것이다. 그래서 분류 결과가 뒤집히는(flipping)일이 적을수록 우수하다고 할 수 있다. 

$$FP_p^f = \frac{1}{m(n-1)} \sum_{i=1}^m \sum_{j=2}^n 𝟙 (f(x_j^{(i)}) \neq f(x_{j-1}^{(i)})) = 𝕡_{x \sim S} (f(x_j^{(i)}) \neq f(x_{j-1}^{(i)}))$$

$m$ : The perturbation sequences  
$n$ : The number of frames  
$p$ : Perturbation type  
$S$ : $\left\lbrace x_1^{(i)}, x_2^{(i)}, ..., x_n^{(i)} \right\rbrace_{i=1}^m$

perturbation sequences의 총 갯수가 m이라고 하고 한 sequences에서 frame의 수가 n이라 했을 때 전체 sequences를 다 확인하면서 각 sequences의 모든 frame을 보는데 여기서 frame은 2부터 시작해야 이전 frame과 비교할 수 있다. 


# AugMix



# Reference 
## Web Link
https://www.youtube.com/watch?v=TPujPAtsH8A&list=LL  

## Paper 
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations : https://arxiv.org/pdf/1903.12261.pdf  
