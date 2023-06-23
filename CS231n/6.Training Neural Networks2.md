# Optimization
Optimizer는 loss를 줄이기 위해 weight와 learning rate와 같은 neural network 속성을 변경하는데 사용하는 Algorithm이다. 
Optimization algorithms의 전략은 loss를 줄이고 정확한 결과를 제공하는 역할을한다. 

신경망을 학습시키는 과정을 다시 살펴보면 아래와 같다. 
```py
while True:
  data_batch = dataset.sample_data_batch()
  loss = network.forward(data_batch)
  dx = network.backward()
  x += - learning_rate * dx
```
우리가 관심을 가질 부분은 코드의 마지막 줄이다. 위 코드에서는 단순 경사 하강이지만 이러한 update 방법에는 여러가지가 있다. 

![Saddle Point - Imgur](https://github.com/em-1001/Information-Security/assets/80628552/5bbe1f39-c69c-456b-b9f8-7aa215d7f889)

### Stochastic Gradient Descent (SGD)
SGD같은 경우 매우 느려서 실제 사용하기가 쉽지 않다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/dd61ca4b-dcf7-47cf-8bcf-eb22205e372c" height="70%" width="70%"></p>

SGD가 느린 이유는 이렇다. 위 사진의 경우 경사가 수직으로는 가파르고 수평으로는 완만한걸 알 수 있다. 따라서 경사가 급한 수직 방향으로는 빠르게 움직이고 수평으로는 느리게 움직여 
위 사진과 같이 지그제그형태로 update가 되고, 이 때문에 느리게 학습하는 것이다. 

### Momentum
```py
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```
Momentum은 위 코드와 같이 update가 된다. v는 속도가 되고 속도를 먼저 업데이트 하고 x를 속도로 업데이트 하는 방식이다. 
v를 구하는 식에 있는 뮤($u$)는 마찰계수로 점차 속도가 느려지게 만든다. 

Momentum은 영상을 보면 처음에 v를 빌드업해주는 과정이 있어서 overshooting이 발생하는 것을 볼 수 있다. 

### Nesterov Momentum
```py
v = mu * v - learning_rate * dx
x += v
```

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/39d6669c-7fa9-4b59-b892-f7dd5500eab0" height="70%" width="70%"></p>

기존 Momentum 업데이트의 v를 구하는 부분을 보면 뮤와 v를 곱한 `mu * v`를 momentum step이라 하고, `learning_rate * dx`를 gradient step이라고 한다. 
이에 따라 사진의 왼쪽은 기본 Momentum 업데이트를 하는 방식인데 Nesterov Momentum의 경우 gradient step을 원점에서 진행하지 않고 이미 momentum step이 진행되었을 것이라 예상되는 지점에서 한다. 

영상의 nag를 보면 방향을 예측하며 가기 때문에 일반 momentum보다 빠르게 update하는 것을 볼 수 있다. 

### AdaGrad 
```py
cache += dx ** 2
x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)
```

AdaGrad는 cache라는 개념을 도입한다. 일반적인 경사하강에 cache를 나눈 값으로 계산하는 것을 볼 수 있다. 
cache는 계속해서 값이 커지며 우리가 가진 파라미터 벡터와 동일한 사이즈를 갖는 벡터라 생각할 수 있다. 
즉 파라미터 별로 다른 learning_rate를 제공하게 되는 것이다. 참고로 1e-7는 0으로 나누는 것을 방지하는 작은 값이다. 
AdaGrad가 update를 하는 방식은 SGD에서의 사진을 봤을 때 수직 축은 경사가 크므로 cache가 커져서 update 속도를 낮추게 되고 
수평의 경우 경사가 낮아 cache가 작고 update 속도는 빠르게 된다. 

하지만 AdaGrad도 문제가 있는데, 시간이 지남에 따라 cache가 계속 증가하므로 learning_rate가 0 이 되어 학습이 중단되는 경우가 발생할 수 있다.  


### RMSProp
AdaGrad의 학습 종료 현상을 개선하기 위해 만들어 졌다. 

```py
cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)
```

RMSProp은 decay_rate라는 하이퍼 파라미터를 도입하여 cache의 값이 서서히 줄어들도록 만들어 준다. 
이렇게 하므로서 AdaGrad의 장점인 경도에 따른 조정을 유지하면서 학습 종료 현상을 해결하게 된 것이다. 

### Adam
```py
m = beta1*m + (1 - beta1)*dx # update first moment
v = beta2*v + (1 - beta2)*(dx ** 2) # update second moment
x += - learning_rate * m / (np.sqrt(v) + 1e-7)
```
Adam은 RMSProp과 momentum을 결합한 형태이다. 코드의 1째 줄은 momentum과 유사하고 2~3번째 줄은 RMSProp과 유사하다. 
beta1, beta2는 하이퍼 파라미터로 보통은 0.9 0.99 등으로 설정한다. 

Adam의 최종적인 코드 형태는 아래와 같다. 

```py
m, v = # ... initialize caches to zeros
for t in xrange(1, big_number)L
  dx = # ... evaluate gradient
  m = beta1*m + (1 - beta1)*dx # update first moment
  v = beta2*v + (1 - beta2)*(dx ** 2) # update second moment
  mb = m / (1 - beta1**t) ## correct bias 
  vb = v / (1 - beta2**t) ## correct bias
  x +=  - learning_rate * mb / (np.sqrt(vb) + 1e-7)
```

추가된 부분은 bias correction으로 최초의 m과 v가 0으로 초기화 되었을 때, 즉 t가 작은 수일 때 m, v를 scaling up 해주는 역할을 한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/77e778ea-8c75-4ba6-b008-cd6d677c99c6" height="70%" width="70%"></p>

결론적으로 지금까지의 update 방식들은 모두 learning_rate을 하이퍼 파라미터로 갖게 되는데 어떤 learning_rate가 최선이냐는 질문에는 답이 없다. 
learning_rate은 결국 시간이 지남에 따라 decay 시키는 것이 가장 최적이 된다. 
step decay의 경우 가장 간단한 방법으로 epoch을 돌때마다 일정한 간격으로 learning_rate를 감소시키는 방법이다. 
epoch이란 모든 train data 한 바퀴 돌아 학습시키는 것을 말한다. 
이외에도 exponential decay, 1/t decay 등이 있다. 주로 쓰이는 것은 exponential decay라고 한다. 
그리고 update 방법으로는 Adam이 가장 많이 쓰인다. 

## Second order optimization methods
지금 까지 알아본 update 방식은 경사를 이용한 first order optimization methods 였고 Second order optimization methods의 경우에는
헤시안($H$)을 이용해 경사 뿐만 아니라 곡면에 대한 정보를 알아내어 학습할 필요없이 최저점으로 이동시키는 방법이다. 





