# Attention Model
Attention설명에 앞서 이전에 공부한 Image Captioning이 어떻게 동작하는지 다시 생각해보면 input image를 CNN에 넣어줘서 마지막 단에 있는 FC로 부터 single feature vector를 추출하고 이를 통해 RNN의 첫 Hidden state vector를 초기화 해준다. 그 다음 RNN에 first word가 들어가고 기본의 Hidden state와 first word가 들어가서 결과를 추출한다. 이러한 과정을 반복하게 된는데 이러한 모델의 한계점은 RNN이 전체 이미지를 한 번만 보게 된다는 것이다.   

## Soft Attention for Captioning

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/6ef254eb-6ccc-4b1b-9220-691b36907e73" height="70%" width="70%"></p>

그래서 새로운 방법은 RNN이 이미지를 한 번만 보는 것이 아니라 매 time step마다 이미지의 다른 부분을 살펴보게 하는 것이다.  
이미지를 CNN에 넣어주고 feature를 추출하는데 이때 FC로부터 feature를 추출하는 것이 아니라 이보다 더 앞에 있는 CNN으로 부터 추출하는 것이다. FC로 부터 feature를 추출하면 single vector이지만 CNN에서 추출하면 feature grid를 얻게된다. 

예를 들어 AlexNet이라고 하면 conv5 layer의 차원은 7x7x512가 되는데 여기서 7x7이 spatial grid가 되고, 7x7 grid 각각은 512차원의 feature vector를 가지게 되는 것이다. 그래서 이 feature grid에서 하나의 grid는 input image의 특정 부분의 feature를 가지고 있다고 할 수 있다. 그래서 이 feature grid가 첫 hidden state를 초기화 해주고, 이를 통해 앞서 captioning과는 달리 a1을 바로 추출한다. 여기서 a1은 location에 대한 확률 분포가 된다. 이는 어떤 단어가 나올것인가에 대한 확률 분포가 아니라 위치에 대한 확률 분포가 된다. 

이러한 위치 확률 분포를 다시 이전의 feature와 연산해서 Weighted feature를 구하게 된다. 이 Weighted feature가 feature vector가 되는 것이고 feature vector는 input image를 summarize하는 single summarization vector가 되는 것이다. 그리고 다음 단계에서 비로소 First word가 들어가는데 이 first word와 Weighted feature 그리고 이전 단계의 hidden state 이렇게 3가지가 input으로 들어가서 h1을 생성한다. 그리고 h1은 두 가지의 output을 갖는데 d1은 단어에 대한 확률 분포이고, a2는 또 다시 위치에 대한 확률 분포를 갖는다. 그리고 이러한 연산을 반복한다.       

여기서 매 step마다 feature와 연산을 통해 summarization vector를 생성한다고 했는데 이 summarization vector를 생성하는데는 Soft, Hard 두 가지 방법이 있다.  

### Soft vs Hard Attention 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4e5753e6-ae52-4e27-be55-7305237dc061" height="80%" width="80%"></p>

이미지에서 CNN을 통해 feature grid를 위와 같이 뽑아내고 이 feature grid는 물론 특정 layer로 부터 추출된 것이다. 이 grid는 RNN으로 들어가서 사진 아래와 같이 location에 대한 확률 분포를 얻게 된다. 그래서 이 feature grid와 위치 확률 분포 두 가지가 결합하여 생성되는 것이 context vector z 인 것이다. 이는 D 차원을 갖는 하나의 summarization vector가 되는데 soft attention에서는 모든 location을 다 고려한다. 그래서 z는 각 위치에 대한 확률을 각각 곱해서 더한 꼴로 구해진다. 이러한 방법은 위 사진의 예로는 4가지 영역을 모두 고려하기 때문에 gradient를 구할 때 문제될 것이 없다. 그래서 일반적인 경사하강으로 학습시키고 end to end로 역전파하는데 매우 적합하다.   

Hard attention은 모든 grid를 고려하지 않고 가장 높은 확률을 갖는 grid의 element를 선택하게 된다. 예를들어 Pa가 가장 높은 확률을 갖는다면 해당 영역을 선택하는 것이다. 그리고 argmax에 의해서 해당 위치의 feature vector 요소를 추출하게 된다. 그런데 이렇게 argmax를 쓰면 dz/dp가 거의 대부분의 경우에 0이 된다. 그 이유는 Pa가 가장 높은 확률을 갖는다고 가정할 때 확률 분포에 조금 변화가 생기더라도 여전히 Pa가 argmax가 될 것이기 때문에 dz에는 변화가 생기지 않기 때문이다. 이렇게 되면 경사하강을 사용할 수 없고 역전파로 학습을 시킬 수 없는 단점이 있다. 그래서 이 경우는 별도의 강화학습으로 학습시켜야 한다.            

그래서 정리하면 soft attention은 역전파가 매우 용이한 반면 Hard attention은 역전파가 안되기 때문에 end to end pipline을 갖기가 힘들고 별도의 강화학습이 필요하다. 각자가 장단점이 있는데 soft attention은 모든 location을 고려하기 때문에 hard보다 연산이 훨씬 많고 hard는 end to end가 불가능 한 반면 연산이 적고 정확도가 다소 높다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/f38d83c1-3477-446a-8ba7-c6b7557b1ed5" height="70%" width="70%"></p>

그래서 결과를 보면 soft는 결과가 흐리게 나오는데 이러한 현상이 나오는 이유는 모든 위치를 고려하고 모든 위치에 대한 확률을 평균했기 때문이고 hard는 한 군데만 명확하게 표현이 되는 것을 볼 수 있다. 이는 최대 확률 한 군데만 고려하기 때문이다. 그래서 아래 단어들을 보면 두 경우 모두 좋은 결과를 똑같이 내는 것을 확인할 수 있다. 


## Soft Attention for Translation 
Soft Attention을 번역에 이용한 예도 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e1993a50-6be8-4a54-b02d-afdb6307f2b4" height="70%" width="70%"></p>

many to many로 input sequence에 대한 Attention을 주고 이를 통해 input 단어들의 확률 분포를 추출했다. 위 예시로는 my라는 단어는 mi라는 단어에 가장 높은 확률 분포를 보여서 my로 번역한 것이다. 

비디오 역시도 비디오의 프레임 별로 Attention을 주어서 captioning할 수 있다. 
더 나아가 Attention은 Q & A에서도 사용할 수 있다. 이미지를 보고 어떤 계절인거 같은지와 같은 질문을 했을 때 ground truth는 fall이고 모델도 fall이라 예측했는데 이때 주목한 것이 낙엽이라고 한다. 


## Attending to arbitrary regions 
지금까지는 정해진 grid에만 Attention을 주었었다. 그래서 꼭 정해진 grid가 아니라 임의의 영역에도 Attention을 주는 방법이 나오게 된다. 
처음으로 나온 논문은 text를 읽은 다음에 손글씨로 그 text를 써주는 모델이다. 이러한 예시는 output에 임의의 지점에 Attention을 준 예시가 될 수 있다.  

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3a55dcb1-9a2f-47b1-963f-ef4b426fa001" height="70%" width="70%"></p>

오른쪽 사진을 보면 위에가 사람이 쓴 것이고 아래가 기계가 쓴 것인데 거의 구분이 안되는 것을 볼 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1abb00a5-a082-49fc-a0e2-7eba1ce64c80" height="80%" width="80%"></p>

RNN을 공부할 때 DRAW라는 모델이 있었는데 위 사진처럼 이 모델은 input에 있어서도 임의의 지점에 Attention을 주면서 이미지를 classify 하는 것을 볼 수 있고, output의 경우에도 임의의 지점에 Attention을 주어 이미지를 생성해내는 것을 볼 수 있다. 


## Spatial Transformer Networks
이렇게 임의의 지점에 Attention을 주는 모델로 쉽게 설명된 모델이 나왔는데 그것이 Spatial Transformer Networks이다.
Spatial Transformer Networks은 다음과 같이 동작한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/73d9abfe-4b2d-4e21-b31d-09b792eedc78" height="80%" width="80%"></p>

input 이미지에서 object를 빨간 박스처럼 localize한다. 그리고 이렇게 localize한 영역에 대한 좌표를 기반으로 해당 부분을 crop하고 rescaling을 해준다. 여기까지의 과정을 함수라고 한다면 이를 미분가능한 함수로 만들 수 있겠냐는 질문이 나오는데 output의 픽셀 좌표를 input의 픽셀좌표에 mapping해주는 함수를 만들자는 아이디어가 나왔다. 그래서 input 각각의 Xs, Ys 좌표마다 output의 어떤 Xt, Yt와 mapping이 되는지 찾는 과정을 반복했고 결과적으로 오른쪽 아래 처럼 sampling grid를 얻게 된다. 그리고 bilinear interpolation을 이용해서 최종 결과를 얻게된다. 여기서 네트워크는 아이디어 수식의 세타값을 예측하므로써 input에 Attention을 주게 되는 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/94041bd0-f9e7-4062-867c-29bb24311c4b" height="80%" width="80%"></p>

이를 pipline으로 표현하면 위와 같다. full image가 들어오고 spatial transformer를 거쳐서 output은 input으로 부터의 ROI가 된다. 
중요한 점은 spatial transformer의 전체 과정이 연속적이고 미분가능하다는 것이다. 그래서 이 전체의 과정을 end to end로 진행할 수 있다는 장점이 있다. 

spatial transformer안에 있는 작은 localization network에있는 세타(affine transform coordinate)를 예측하는 부분은 localization을 한 다음에 평행을 유지하면서 transform된 좌표를 얻는 과정을 말하고, 다음 단계에서는 grid generator에서 방금 구한 세타를 이용해 sampling grid를 추출하게 된다. 그리고 마지막 단계에서는 sampler가 bilinear interpolation을 이용해서 결과를 산출하는 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/7ce24e44-8121-4eb6-a0fc-1887738ae430" height="70%" width="70%"></p>

그래서 그 결과는 위와 같이 차원이 꼬여있는 입력에 대해서도 입체적으로 분석을 하면서 결과를 내는 것을 볼 수 있고, 회전된 이미지에 대해서도 복원을 해주는 것을 볼 수 있다. 



# Reference   
https://www.youtube.com/watch?v=Bmx2S1dSAV0&list=PL1Kb3QTCLIVtyOuMgyVgT-OeW0PYXl3j5&index=12   



