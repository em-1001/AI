# paper
Unknown Sniffer for Object Detection: Don’t Turn a Blind Eye to Unknown Objects : https://arxiv.org/pdf/2303.13769.pdf  

Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection : https://arxiv.org/pdf/2011.12885.pdf  

# link 
Object Detection] ATSS: Adaptive Training Sample Selection : https://velog.io/@markany/%EB%94%A5-%EB%9F%AC%EB%8B%9D-Object-Detection-Adaptive-Training-Sample-Selection

Grounding DINO : SOTA Zero-Shot Object Detection : https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/


$$\hat{y} = \sigma_{i=0}^{n}P(y_{i})y_{i} $$

$$DFL(S_{i}, S_{i+1}) = -((y_{i+1}-y)\log(S_{i}) + (y-y_{i})\log(S_{i+1}))$$
target 을 y라 할때 y와 가장 가까운 값 $y_{i} \leq y \ge y_{i+1}$ 인 $y_{i}, y_{i+1}$에서 $P(y)$가 peak 값을 가지도록 위와 같은 complete form의 cross entropy 를 이용해 학습 한다. $S_{i}=\frac{y_{i+1}-y}{y_{i+1}-y_{i}}, S_{i+1} = \frac{y-y_{i}}{y_{i+1}-y_{i}}$ 이다. ($S_{i},S_{i+1}$ 이 왜 저렇게 되는지 궁금하면 linear interpolation을 생각해 보자)

예를 들어 실례를 들어 보자면 unnormalized logits
$y_{i}=[9.38, 9.29, 4.48, 2.55, 1.30, 0.42, 0.03, -0.28, -0.51, 0.83, -1.27, -1.56, -1.78, -1.94, -.193, -1.38]$ 이라고 할때 target $y=0$ 이면 cross entropy 에 의해 $p(y_{0}), p(y_{1})$이 근처에서 peak 값을 갖도록 학습이 되는 방식이다.


