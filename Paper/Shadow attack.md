## Shadow attack
이 논문은 뉴럴 네트워크를 공격하는 새로운 공격 유형으로 Shadow attack(그림자 공격)을 제안한다. 
Shadow attack의 특징은 다음과 같다. 

1. Imperceptibility : 정상적인 이미지 처럼 보임
2. Misclassification : 타겟 클래스로 오분류하도록 유도한다.
3. Strongly certified : 높은 인증 반경(certificate radius)을 가진다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/7cdd8a2a-4b36-4412-bfc0-c58a70e0c305" height="80%" width="80%"></p>

Strongly certified의 경우 adversarial examples을 방어기 위한 기법으로 certified defense가 있는데 이러한 방어 기법을 뚫는 기법이라고 할 수 있다. 
즉 이 논문은 제목인 Breaking Certified Defenses Semantic Adversarial Examples with Spoofed Robustness Certificates에서도 알 수 있듯이 적대적 공격에 대한 방어 기법으로
Robustness Certificates와 같은 기법들이 나왔는데 이런 방어 기법을 다시 뚫는 기법을 Shadow attack으로 제안하는 것이다. 

## Background

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/523ba308-0597-46ee-be1b-0c5f7660145f" height="80%" width="80%"></p>

우선 적대적 예제라는 것은 인간의 눈에 띄지 않게 약간 변형된 데이터로, 뉴럴 네트워크의 부정확한 결과를 유도하게 된다. 
이러한 적대적 예제를 이용한 공격에 가장 효과적이라고 알려진 방어 기법은 Adversarial training이다. 
적대적 학습이라고도 불리며 Adversarial training은 뉴럴 네트워크를 공격으로부터 Robustness하게 만들기 위해 Adversarial example을
학습 데이터로 이용하는 방법이다. 

$$min\underset{y}

## Reference
https://arxiv.org/pdf/2003.08937.pdf
https://www.youtube.com/watch?v=D1j3QiXPRag&list=LL&index=7&t=1680s
