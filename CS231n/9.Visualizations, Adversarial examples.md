# Visualizations
CNN이 어떻게 좋은 성능을 내는지 확인하기 위해 Visualizations을 이용하여 들여다 본다. 
CNN이 좋은 성능을 내는 것은 알고 있는데 이게 어떻게 이런 성능을 내는지는 여전히 blackbox인 경우가 많다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e6836d8b-a1c4-480b-ac3f-569c28b4cb01" height="90%" width="90%"></p>

CNN이 무엇을 하는지 알아내는 가장 간단한 방법은 row activation을 살펴보는 것이다. 하나의 뉴런을 activation하게 하는 부분을 시각화 하는 것인데 위 사진을 보면 pool5 layer에서 임의의 뉴런을 취한 다음에 여러 장의 이미지를 학습시킨 것이다. 
그러면 이 임의의 뉴런을 가장 excite시키는 것이 어떤 것인지 알 수 있는데 사진의 각 행이 하나의 뉴런에 매치한다고 볼 수 있다. 
예를 들어 1행의 뉴런을 보면 사람 사진에 대해 반응한 것이고, 4행 뉴런을 보면 text에 반응한 것이다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/67567942-7936-49b4-9613-be6e785179c5" height="90%" width="90%"></p>

두 번째 방법은 Filter(Kernel)를 visualize한 방법이다. 위 사진은 gabor filter로 특정 외곽선의 방향과 같은 것을 검출하는 필터이다. 
예를 들어 conv1 layer의 filter들에 대해 시각화 하면 위 사진과 같이 나오고, 여기서 이미지에 직접 작용하는 filter는 
첫 번째 conv에 있는 filter라고 할 수 있다.   

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/6488d5cd-f7fb-4f21-a13f-7e23b5e8c453" height="90%" width="90%"></p>

첫 번째 layer가 아닌 그 다음 layer들의 weight들은 visualize를 할 수는 있지만 raw이미지에 대한 것이 아니라 전 단계의 activation
에 대한 visualize이기 때문에 해석하기가 쉽지는 않다. 의미가 크지 않다고 할 수 있다. layer 2부터 괄호 안에 있는 이미지 묶음이
하나의 filter에 대응하는 것이라 볼 수 있다. 


세 번째 방법은 representation 자체를 visualize하는 것이다. classification직전의 layer인 fc7 layer에 이미지에 대한 
4096차원의 코드가 들어있다고 볼 수 있고, 이 각각의 코드들을 모아서 visualize하는 방식이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/92eacb5c-3685-44a4-8db2-357243cc80cf" height="30%" width="30%"></p>

대표적으로는 t-SNE라는 방식으로 CNN의 시각으로 볼 때 유사한 것을 가까운 곳으로 위치시켜서 클러스터링 하는 방법이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e6bc782e-30e2-4c72-8625-349f493c3094" height="60%" width="60%"></p>

네 번째 방법으로는 Occlusion experiments라는 방법으로 은닉을 통해 실험을 하는 것이다. 
위 사진을 자세히 보면 회색으로 가려진 부분이 있는데 이 부분을 0으로 된 행렬로 은폐시켜서 
이 은폐시키는 사각형을 sliding 시키면서 위치에 따라 classification 확률이 어떻게 변하는지 관찰한다. 

2열의 결과 사진을 보면 예상할 수 있듯이 강아지의 경우 얼굴 사진을 가리면 확률이 크게 감소하게 되고, 
자동차의 경우엔 바퀴부분을 가리는 경우 분류 능력이 떨어지는 것을 볼 수 있다.    

https://youtu.be/AgkfIQ4IGaM 실제 이 영상을 보면 우리가 따로 요청을 하지 않았음에도 네트워크가 학습을 진행하면서 어떤 뉴런은 옷의 주름, 어떤 뉴런은 text 이런식으로 알아서 각각의 역할을 한다는 것을 알 수 있다. 

영상에서도 언급이 되었다시피 Activation을 Visualize하는데는 아래 2가지 접근 방법이 있다.    

1. Deconvolution-based approach
2. Optimization-based approach

## Deconvolution-based approach
우선 Deconvolution-based approach가 무엇인지 알기 위해서 이미지가 input으로 들어왔을 때 특정 layer의 어느 한 뉴런의 
gradient를 어떻게 계산할지에 대한 질문에 답을 해야 한다. 일단 임의의 뉴런이 존재하는 곳 까지만 forward pass를 해주고, 
activation을 구한 다음 해당 layer에 있는 뉴런들에서 우리가 보고자 하는 임의의 뉴런을 제외한 나머지 뉴런들의 gradient를 
모두 0으로 만들어주고 해당 임의의 뉴런에 대해서만 gradient를 1.0으로 주고 여기에서 부터 역전파를 진행하면 된다. 
이렇게 하면 이미지에 대한 gradient를 시각화하여 볼 수 있게 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/fc609e09-b5f2-4bc2-953c-255b8c508f9e" height="50%" width="50%"></p>

하지만 이때 왼쪽 사진과 같이 애매한 시각화 결과가 나올 수 있는데, 이때는 그냥 역전파가 아니라 Guided backpropagation을 이용하면 된다. 
Guided backpropagation을 사용하게 되면 positive한 요소만 역전파 시에 반영하여 이전 왼쪽 이미지 처럼 negative한 요소와 
상쇄되어 애매한 결과가 나오지 않고 더 선명하게 나오게 된다.  
Guided backpropagation은 다른 것은 바뀌는 것이 없고 relu만 modified relu로 사용한 것이다.  

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/91c14885-0f4a-4d34-9d6a-e9e6a0909537" height="70%" width="70%"></p>

a를 보면 forward pass를 거치고 결과로 나온 feature map에서 관찰하고자 하는 뉴런이 숫자 '2'가 적힌 뉴런이라고 하면 
해당 뉴런의 gradient만 1로하고 나머지는 0으로 바꿔버린다.  
b의 relu의 경우를 보면 input에 대해 0보다 작은 부분은 모두 0으로 처리하고 역전파의 경우 앞서 0으로 바뀐 4군데 빼고 
나머지는 값 그대로 전달되는 것을 확인할 수 있다. 

Guided backpropagation의 경우에는 이전 forward pass에서 relu에 의해 0으로 처리된 부분 외에 
역전파 시에도 0보다 작은 값들이 전부 0으로 바뀌게 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/cba99fb8-6677-423b-ab9f-67b1c03fb6aa" height="40%" width="40%"></p>

Deconvolution-based의 또다른 방법으로 deconvnet이라는 것도 제시가 되었었는데, deconvnet은 relu의 영향을 받지 않고 
그냥 역전파 시에 0보다 작은 값들만 0으로 처리하게 된다. 물론 양수 값들은 그대로 전달이 된다. 이 방법 역시 잘 동작을 하게 된다. 

## Optimization-based approach
### Find images that maximize some class score
Optimization to Image는 이미지에 대해 최적화를 하는 것이다. 
이미지가 Optimization의 대상이 되는 파라미터가 되는 것이다. 
즉 일반적으로는 weight들이 파라미터가 되었었는데, 이와는 다르게 weight 파라미터들을 고정시켜두고 
이미지를 파라미터로 이용하게 되고 update도 weight가 아닌 이미지를 update하게 된다. 

그래서 특정 class의 score를 최대화 하는 이미지를 찾으려 하는데, 그 때 사용되는 식이 아래와 같다 

$$arg \ \underset{I}\max S_c(I) - \lambda||I||_2^2$$

$S_c$는 Softmax 이전 class c에 대한 score이고, 
뒤에 두 번째 term 같은 경우 L2 regularization과 같은 term이 들어가게 되는 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/86c41804-ed8c-4dee-92ee-d11f8903f417" height="70%" width="70%"></p>

그래서 Optimization-based를 하는 방법은 일단 0으로 구성된 zero image를 네트워크에 넣어주고 forward pass를 해준다. 
그리고 중요한 점은 Score Vector에서 우리가 관심을 갖는 Class에 대해서만 1로 설정해주고 나머지는 다 0으로 해준다. 
그런 다음 역전파를 해주면 가중치에 대한 update가 아닌 이미지에 대한 update를 약간 수행하게 되는 것이고
그 다음 이렇게 방금 update된 이미지를 다시 input으로 넣어서 이 가정을 반복해 주면 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/f0290f04-9223-4e53-9f41-83035143b71f" height="60%" width="60%"></p>

그렇게 반복을 하게 되면 위와 같은 이미지를 얻게된다. 그래서 처음에 zero image였던 것이 각각의 class에 따라 
위의 이미지 처럼 되고 이를 통해 class가 어떤 특성을 보고 activation이 되는지를 확인할 수 있는 것이다. 

### Visualize the Data gradient
앞에서는 특정 class의 score를 maximize하는 이미지를 찾는 방법을 알아보았고, 이제는 data의 gradient를 visualize하는 방법을 알아볼 것이다. 

기본적으로 data의 gradient는 세 개의 채널을 가진다는 것을 염두해두고  강아지 사진이 있다고 할 때 이 이미지를 우선 CNN에 돌리고 강아지 gradient를 1로 설정해 준다. 그 상태에서 역전파를 하면 이미지 gradient에 도달했을 때 그 도달한 순간의 RGB 3개로 1차원의 heatmap과 같은 것을 생성하게 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/bfe7f485-1578-45e0-abe3-4d839a6cffc0" height="60%" width="60%"></p>

위 사진에서 object가 있는 부분은 하얀색으로 표현되어 하얀 부분은 영향을 주는 부분, 검은 부분은 영향을 주지 못하는 부분이라 할 수 있다. 

또한 이러한 방식으로 적용해서 grabcut이라는 segmentation알고리즘으로 
segmentation을 한 사례도 있다고 한다.  

이렇게 지금까지 살펴본 방법은 원하는 class의 score gradient를 1로 줘서 
해당 class를 시각화 한 것인데 꼭 이렇게  score gradient를 1로 주지 않고 
어떤 ConvNet layer이든간에 임의의 뉴런의 activation값을 1로 해주고 나머지 뉴련의 activation값을 0으로 만들어서 동일한 효과를 얻을 수 있다. 

$$arg \ \underset{I}\max S_c(I) - \lambda||I||_2^2$$

우리가 앞서 봤던 것은 score에 대해 2번째 term과 같은 regularization을 썼었는데, 이 방법 대신 동일한 방법으로 update 시키는데 regularization term 대신에 이미지 자체를 blur처리 해주는 것이다. blur를 해줌으로써 너무 잦은 frequency가 발생하는 것을 방지하기 때문에 오히려 효과가 좋을 수도 있다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e7a265de-fa0d-4a31-824b-32fcb3eba5e4" height="70%" width="70%"></p>

결과를 보면 blur처리를 해줬을 때가 조금 더 선명하게 결과가 나오는 것을 볼 수 있다. 
정리하면 앞선 방법은 L2 regularization을 사용했고, 여기에서는 단지 blur 처리를 해준 것이다. 

여기서 질문이 하나 나오는데 CNN의 코드가 주어졌을 때(FC layer라고 하면 4096 개의 코드) 원래의 이미지를 복원할 수 있냐는 것이다. 
복원하기 위해서는 복원할 이미지의 코드는 주어진 코드와 유사해야하고 자연스러워 보여야 할 것이다. 

이렇게 복원을 하기위해서 아래 식과 같이 차이를 minimize하는 즉 회귀를 통해 유사한 이미지를 찾는 과정을 거친다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/099964e8-6c0e-4d0a-94e6-9b4398477bf9" height="40%" width="40%"></p>

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/f78e7b6f-a09d-464c-9c5b-7eae3e9e078d" height="70%" width="70%"></p>

실제 복원한 예를 보면 FC7 layer의 4096 코드로 복원한 것이다. 그림을 보면 4096의 코드가 이미지 복원을 위한 어느정도의 정보를 가지고 있는지 알 수 있다. 이러한 복원은 FC7에서만 가능한 것이 아니라 conv layer의 어느 부분에서도 가능하다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/695f08d0-2115-4283-8572-d14ecc9a5d42" height="70%" width="70%"></p>

위 사진은 모든 layer에 대한 복원을 한 것인데 입력 이미지와 멀어지는 뒷단의 layer 일수록 흐릿하고 앞단으로 갈수록 선명한 것을 볼 수 있다. 

이 그림들을 통해 forward pass로 갈 수록 원본이미지에 대한 특징을 어떻게 잃어가는지를 파악할 수 있게 된다. 


# Neural Style
Neural Style도 마찬가지로 이미지를 Optimization하는 과정에서 만들 수 있는 모델이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/00b71b2d-08c7-4c93-b86c-24d06577b9e0" height="60%" width="60%"></p>

위 사진과 같이 원본 이미지를 특정 화가의 풍과 합성해서 그림을 만들어내는 것을 볼 수 있다. 

Neural Style이 동작하는 과정은 우선 content image(원본 이미지)를 CNN에 넣고 각각의 layer에서의 activation들을 저장한다. 그 다음에는 style target(합성할 스타일 이미지) image도 CNN에 넣는데 여기선 style과 관련된 통계를 activation이 아닌 pair-wise statistics(아래 예시에서는 pair-wise statistics 중 gram matrices를 사용)값을 사용한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/00bbbe85-6594-46b2-9aad-4e17ac55d8d9" height="80%" width="80%"></p>

그래서 다음과 같이 최적화를 진행하는데 앞서 말한 activation들과 gram matrices들의 각각의 loss를 구하게 되는데, 적절한 초기 값으로 시작하는 random image를 conv에 넣어준 뒤 activation인 content 쪽과 gram matrices인 style 쪽과의 loss를 최소화해 나가면서 서로 경쟁하는 식으로 Neural Style을 만들어 나가게 된다. 

한 가지 참고할 점은 앞 강의에서 Optimization관련 개념을 배울 때 모멘텀, 아담과 같은 first order Optimization을 배웠고 secode order로 L-BFGS를 배웠는데 바로 이 Neural Style이 L-BFGS로 최적화 될 수 있는 좋은 예라고 한다. 왜냐면 이게 거대한 data set을 가지는 것도 아니며 큰 연산을 필요로 하지 않기 때문이다. 

# Adversarial examples
이렇게 이미지에 대한 최적화를 진행하므로서 어떤 class의 score라도 maximize가 가능함을 보았는데 이를 이용해서 CNN을 속일 수가 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/67c5293b-80ba-443d-b4f8-eab295873b9c" height="80%" width="80%"></p>

예시를 보면 버스 이미지를 타조 class의 gradient만 1로 설정하고 나머지는 다 0으로 설정한 뒤 역전파를 하면 실제 눈으로는 버스 사진처럼 보이는데, classify 할 때는 타조로 하게 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/9432f0c8-6715-4700-8b93-95400fb96ad4" height="80%" width="80%"></p>

이와 같이 random한 이미지 뿐만 아니라 위 사진과 같이 random한 noise로도 똑같이 적용이 가능하다. 이 noise 역시 우리 눈에는 그냥 noise로 보이는데 실제로는 각각의 target된 class로 분류되는 것을 볼 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/96124458-0e5a-44d4-b8b7-a4151781875b" height="80%" width="80%"></p>

더 나아가 이런 특이한 패턴들도 인식하게 만들 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/d22ec357-e7eb-4db0-a57d-7986f9620e47" height="80%" width="80%"></p>

sigmoid binary classify로 적대적 example을 만드는 예시를 보면 우리가 목표하는 것은 확률을 50%이상으로 만들어 1로 classify하도록 하는 거라 하면 기존의 x와 w를 내적하면 0.0474로 약 0.5%의 확률이 나와 0으로 분류될 수 밖에 없는데, adversarial x를 w가 양수인 경우에는 조금 더 크게 만들고 음수인 경우에는 조금 더 작게 만들어서 0.5씩 모두 차이를 준 결과 88%라는 결과가 나오는 것을 볼 수 있다. 즉 원래 x는 class 0으로 분류가 되어야 맞는 것인데 1로 분류하도록 adversarial x를 만든 것이다. 

이 경우는 10개의 차원만 가지는 예시였는데 실제 이미지의 경우 224X224의 큰 차원을 가지게 때문에 0.5가 아니라 훨씬 작은 변화만 주더라도 쉽게 Adversarial examples을 만들 수가 있는 것이다. 이러한 예시는 기본적으로 linear classify에 문제가 있음을 기초로한 것이다. 



# Reference
https://www.youtube.com/watch?v=j_BeROoelLo&list=PL1Kb3QTCLIVtyOuMgyVgT-OeW0PYXl3j5&index=8  
