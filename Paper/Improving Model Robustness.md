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




# Reference 
## Web Link
https://www.youtube.com/watch?v=TPujPAtsH8A&list=LL  

## Paper 
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations : https://arxiv.org/pdf/1903.12261.pdf  
