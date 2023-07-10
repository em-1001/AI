# Introduction
본 논문은 사람의 눈에 맺힌 상을 통해 3D로 그 사람이 보고 있는 것을 복원하는 기술을 제안한다. 
이전에 연구된 imaging formulation과 2D 이미지를 3D로 변환해주는 모델 NeRF로 부터 영감을 얻었다고 한다. 
multi-view information 캡처를 위해 카메라가 움직일 필요가 있는 표준적인 NeRF capture 설정과는 달리 본 논문에서는 카메라를 고정시키고 머리의 움직임에 따라 달라지는 eye image의 multi-view를 이용한다. 

논문에서 제안하는 eye images를 3D NeRF로 재구성하는 것은 크게 2가지 어려움이 있었다. 
첫 번째는 source separation 이다. 눈의 복잡한 홍채(**iris**) 질감에서 나오는 반사를 분리해야 했다. 
이러한 복잡한 패턴은 선명한 일반적인 이미지와는 달리 픽셀 대응을 방해하고 3D 복원에 모호성을 주었다.
두 번째 문제는 각막(**cornea**) pose 추정이다. 눈은 매우 작기 때문에 정확히 localize하기가 힘들다. 
multi-view reconstruction는 위치의 정확성과 3D orientations에 의존하기 때문에 cornea pose estimation 문제는 구현을 어렵게 만든다. 



# Reference
## Web Links
https://www.matthewtancik.com/nerf  


## Papers 
NeRF : https://arxiv.org/pdf/2003.08934.pdf  
Seeing the World through Eyes : https://arxiv.org/pdf/2306.09348.pdf  
