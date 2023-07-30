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
