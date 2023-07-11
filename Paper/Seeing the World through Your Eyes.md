# Introduction
본 논문은 사람의 눈에 맺힌 상을 통해 3D로 그 사람이 보고 있는 것을 복원하는 기술을 제안한다. 
이전에 연구된 imaging formulation과 2D 이미지를 3D로 변환해주는 모델 NeRF(Neural radiance Fields)로 부터 영감을 얻었다고 한다. 
multi-view information 캡처를 위해 카메라가 움직일 필요가 있는 표준적인 NeRF capture 설정과는 달리 본 논문에서는 카메라를 고정시키고 머리의 움직임에 따라 달라지는 eye image의 multi-view를 이용한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/5795dc6d-9dfb-4d28-bba9-7ec3ba6ab29c" height="50%" width="50%"></p>


논문에서 제안하는 eye images를 3D NeRF로 재구성하는 것은 크게 2가지 어려움이 있었다. 
첫 번째는 source separation 이다. 눈의 복잡한 홍채(**iris**) 질감에서 나오는 반사를 분리해야 했다. 
이러한 복잡한 패턴은 선명한 일반적인 이미지와는 달리 픽셀 대응을 방해하고 3D 복원에 모호성을 주었다.
두 번째 문제는 각막(**cornea**) pose estimation이다. 눈은 매우 작기 때문에 정확히 localize하기가 힘들다. 
multi-view reconstruction는 위치의 정확성과 3D orientations에 의존하기 때문에 cornea pose estimation 문제는 구현을 어렵게 만든다. 

이러한 문제를 해결하기 위해서 논문에서는 NeRF를 eye images를 training할 수 있도록 2가지 중요한 요소를 통해 repurpose 했다. 

1. **texture decomposition** : leverages a simple radial prior to facilitate separating the iris texture from the overall radiance field.
2.  **eye pose refinement** : which enhances the accuracy of pose estimation despite the
challenges presented by the small size of eyes.

- **Radial prior for irises** : 눈 이미지에서 홍채(iris) texture를 decomposition하기 위해서 Radial prior를 제안한다. deep learning 에서 prior는 noise가 없는 이미지의 분포를 의미한다고 한다. 이러한 Radial prior는 재구성된 radiance field의 quality를 크게 향상시켰다. 논문에서 인공지능은 카메라로 부터 나온 ray에 의해 눈에서 반사된 radiance field를 학습하게 되는데, 이렇게 training한 걸 토대로 reconstruction한 것을 보여줄 때 iris를 reconstruction 결과로부터 제거하기 위해서 iris texture를 탐지하도록 하는 2D texture map을 학습시킴과 동시에 texture decomposition을 하는 것이다.         
- **Cornea pose refinement** :  cornea pose refinement procedure는 noisy한 눈의 pose estimation을 경감시켜준다. 이는 사람의 눈에서 feature extraction하는 것을 가능하게 해준다.   

# Related Work



# Reference
## Web Links
https://www.matthewtancik.com/nerf   
https://sonsnotation.blogspot.com/2020/12/14-image-restoration.html     
https://twitter.com/AiBreakfast/status/1669816017890029569     
https://nuggy875.tistory.com/168   

## Papers 
NeRF : https://arxiv.org/pdf/2003.08934.pdf  
Seeing the World through Your Eyes : https://arxiv.org/pdf/2306.09348.pdf  
