# Background
## YOLOv1...

# Real-Time Flying Object Detection with YOLOv8
본 논문은 현재 state-of-the-art인 YOLOv8을 이용한 비행체 탐지모델을 제안한다. 일반적으로 Real-time object detection은 object의 공간적 사이즈(spatial sizes), 종횡비(aspect ratios), 모델의 추론 속도(inference
speed), 그리고 noise 등의 변수로 어려움이 있었다. 비행체는 위치(location), 크기(scale), 회전(rotation), 궤도(trajectory)가 매우 빠르게 변하기 때문에 앞선 문제들은 비행체를 탐지하는데 더욱 부각된다. 그렇기에 비행체의 이러한 변수에 대해 thorough하고 빠른 추론속도를 갖는 모델이 중요했다. 

![image](https://github.com/em-1001/AI/assets/80628552/7c8e5c53-3e12-46fa-813f-6698c1b06538)

본 논문에서는 dataset중 80%를 train, 20%을 validation으로 나누었다. 각 dataset의 이미지는 class number가 label되어있고, bounding box 가장자리의 좌표를 표시해놨다. 하나의 이미지에는 평균적으로 1.6개의 object가 있고, median image
ratio는 416x416이다. 이미지는 auto orientation으로 전처리 되었으며, augmentations은 적용하지 않았다. 

##  Model Choice and Evaluation
논문에서는 우선 YOLOv8의 small, medium, and large 버전에 대해 최적의 inference speed와 mAP50-95를 갖는 모델 버전을 선택했고, 이후에  hyper parameters를 최적화 했다. 
VOLOv8의 각 버전에 대한 정보는 아래와 같다.   
nano (yolov8n), small (yolov8s), medium (yolov8m), large (yolov8l), and extra large (yolov8x)

|Model|Input size(pixels)|mAP50-95|params(M)|FLOPS(B)|
|-|-|-|-|-|
|YOLOv8n|640|37.3|3.2|8.7|
|YOLOv8s|640|44.9|11.2|28.6|
|YOLOv8m|640|50.2|25.9|78.9|
|YOLOv8l|640|52.9|43.7|165.2|
|YOLOv8x|640|53.9|68.2|257.8|

small, medium, large 모델들의 parameters는 각각 (11151080, 25879480, & 43660680)이고, layers는 (225,295, & 365)이다. 본 논문에서 학습시킨 결과 small과 medium 사이에서는 mAP50-95에서 큰 성능향상이 있었으나 medium, large 사이에서는 그렇지 않았다고 한다. 또한 validation set에 대해서 small, medium, and large의 inference speed는 각각 4.1, 5.7, and 9.3 milliseconds 였다고 한다. 원래 목표였던 average inference speed는 30 to 60 frames for 1080p였고, medium size
model을 multiple 1080p HD videos에서 테스트해본 결과 average total speed (pre-proccess speed(0.5ms) + inference speed(17.25ms) + post-process speed(2ms)) of 19.75 ms(50 frames per second)로 목표에 적합하여 모델을 medium size로 결정하고 hyper-parameters 튜닝을 진행했다고 한다. 





# Reference 
## Web Link 
One-stage object detection : https://machinethink.net/blog/object-detection/  
YOLOv5 : https://blog.roboflow.com/yolov5-improvements-and-evaluation/   
YOLOv8 : https://blog.roboflow.com/whats-new-in-yolov8/       
mAP : https://blog.roboflow.com/mean-average-precision/  
　　　https://a292run.tistory.com/entry/mean-Average-PrecisionmAP-%EA%B3%84%EC%82%B0%ED%95%98%EA%B8%B0-1   
　　　https://donologue.tistory.com/405    
SiLU : https://tae-jun.tistory.com/10     
Weight Decay, BN : https://blog.janestreet.com/l2-regularization-and-batch-norm/  
focal Loss : https://gaussian37.github.io/dl-concept-focal_loss/  


## Paper 
Real-Time Flying Object Detection with YOLOv8 : https://arxiv.org/pdf/2305.09972.pdf   
YOLO : https://arxiv.org/pdf/1506.02640.pdf    
YOLOv2 : https://arxiv.org/pdf/1612.08242.pdf    
YOLOv3 : https://arxiv.org/pdf/1804.02767.pdf  
YOLOv4 : https://arxiv.org/pdf/2004.10934.pdf   
YOLOv6 : https://arxiv.org/pdf/2209.02976.pdf  
YOLOv7 : https://arxiv.org/pdf/2207.02696.pdf  
