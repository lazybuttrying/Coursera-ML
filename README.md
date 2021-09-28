# Deep-Learning-Specialization


https://www.coursera.org/learn/neural-networks-deep-learning?#faq

Instructor : Andrew Ng


## Neural Networks and Deep Learning
- Neural Networks와 Deep Learning 기초 개념 

## Improving Deep Neural Networks
## Hyperparamter Tuning, Regularization and Optimization
- DNN의 프로세스에 대한 이해와 성능을 높이는 방법에 대해 알아보기
1. 모델 훈련, bias와 varaiance를 분석하는 방법
2. 초기화, L2, Dropout Regularization
3. hyperparameter 튜닝, Batch Regularization, Gradient 분석
4. 다양한 최적화 알고리즘 
  - mini batch gradient descent
  - momentum
  - RMSprop
  - Adam
5. Tensorflow에서 구현하기


## Structuring Machine Learning Projects
- 머신러닝 프로젝트를 구축하는 방법과 의사결정 방법에 대해
1. 오류 진단, 이런 오류를 줄이기 위한 전략의 우선순위를 결정
2. 일치하지 않는 train-test set 처리, 인간 수준과 비교하여 ML 목표 설정
3. End-to-end Learning, Transfer Learning, Multi-task Learning


## Convolutional Neural Networks
- 컴퓨터 비전의 발전 역사
- 자율 주행, 얼굴 인식, 방사선 이미지 읽기 등의 분야에 대해서 알아보기
- 최근의 변형된 Convolution Network 이해하기 (Residual Network)
- 시각적 감지 및 인식 작업에 Conv NN 적용해보기
- Nerual Style Transfer를 사용하여 예술작품을 생성
- 이미지, 비디오, 2D, 3D에 적용해보기


## Sequence Models
- 음성 인식, 음악 합성, 챗봇, 번역, 자연어 처리에 대해

- RNN 구축 및 훈련 (+ GRU, LSTM)
- 문자 하나 수준의 자연어 모델링에 RNN 적용하기
- Word Embedding
- HuggingFace tokenizers, Transformer model로 NLP 문제 해결하기 

# 
# 
# Reference

##
## Neural Networks and Deep Learning
### Week 2:
- Implementing a Neural Network from Scratch in Python – An Introduction (Denny Britz, 2015)
  - http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
- Why normalize images by subtracting dataset's image mean, instead of the current image mean in deep learning? (Stack Exchange)
  - https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current
  
### Week 3:
- Demystifying Deep Convolutional Neural Networks (Adam Harley)
  - https://www.cs.ryerson.ca/~aharley/neural-networks/
- CS231n: Convolutional Neural Networks for Visual Recognition (Stanford University)
  - https://cs231n.github.io/neural-networks-case-study/
  
### Week 4:
- Autoreload of modules in IPython (Stack Overflow)
  -  https://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


##
## Improving Deep Neural Networks
## Hyperparamter Tuning, Regularization and Optimization
### Week 3:
- Introduction to gradients and automatic differentiation (TensorFlow Documentation)
  - https://www.tensorflow.org/guide/autodiff
- tf.GradientTape (TensorFlow Documentation)
  - https://www.tensorflow.org/api_docs/python/tf/GradientTape


## 
## Convolutional Neural Networks
### Week 1:
- The Sequential model (TensorFlow Documentation)
  - https://www.tensorflow.org/guide/keras/sequential_model
- The Functional API (TensorFlow Documentation)
  - https://www.tensorflow.org/guide/keras/functional

### Week 2:
- Deep Residual Learning for Image Recognition (He, Zhang, Ren & Sun, 2015)
  - https://arxiv.org/abs/1512.03385
- deep-learning-models/resnet50.py/ (GitHub: fchollet)
  - https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, ​& Adam, 2017)
  - https://arxiv.org/abs/1704.04861
- MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler, Howard, Zhu, Zhmoginov &Chen, 2018)
  - https://arxiv.org/abs/1801.04381
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019)
  - https://arxiv.org/abs/1905.11946
  
### Week 3:
- You Only Look Once: Unified, Real-Time Object Detection (Redmon, Divvala, Girshick & Farhadi, 2015)
  - https://arxiv.org/abs/1506.02640
- YOLO9000: Better, Faster, Stronger (Redmon & Farhadi, 2016)
  - https://arxiv.org/abs/1612.08242
- YAD2K (GitHub: allanzelener)
  - https://github.com/allanzelener/YAD2K
- YOLO: Real-Time Object Detection
  - https://pjreddie.com/darknet/yolo/
- Fully Convolutional Architectures for Multi-Class Segmentation in Chest Radiographs (Novikov, Lenis, Major, Hladůvka, Wimmer & Bühler, 2017)
  - https://arxiv.org/abs/1701.08816
- Automatic Brain Tumor Detection and Segmentation Using U-Net Based Fully Convolutional Networks (Dong, Yang, Liu, Mo & Guo, 2017)
  - https://arxiv.org/abs/1705.03820
- U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger, Fischer & Brox, 2015)
  - https://arxiv.org/abs/1505.04597
   
### Week 4:
- FaceNet: A Unified Embedding for Face Recognition and Clustering (Schroff, Kalenichenko & Philbin, 2015)
  - https://arxiv.org/pdf/1503.03832.pdf
- DeepFace: Closing the Gap to Human-Level Performance in Face Verification (Taigman, Yang, Ranzato & Wolf)
  - https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf
- facenet (GitHub: davidsandberg)
  - https://github.com/davidsandberg/facenet
- How to Develop a Face Recognition System Using FaceNet in Keras (Jason Brownlee, 2019)
  - https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
- keras-facenet/notebook/tf_to_keras.ipynb (GitHub: nyoki-mtl)
  - https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb
- A Neural Algorithm of Artistic Style (Gatys, Ecker & Bethge, 2015)
  - https://arxiv.org/abs/1508.06576
- Convolutional neural networks for artistic style transfer
  - https://harishnarayanan.org/writing/artistic-style-transfer/
- TensorFlow Implementation of "A Neural Algorithm of Artistic Style"
  - http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
- Very Deep Convolutional Networks For Large-Scale Image Recognition (Simonyan & Zisserman, 2015)
  - https://arxiv.org/pdf/1409.1556.pdf
- Pretrained models (MatConvNet)
  - https://www.vlfeat.org/matconvnet/pretrained/



## 
## Sequence Models
### Week 1
- Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy (GitHub: karpathy)
  - https://gist.github.com/karpathy/d4dee566867f8291f086
- The Unreasonable Effectiveness of Recurrent Neural Networks (Andrej Karpathy blog, 2015)
  - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- deepjazz (GitHub: jisungk)
  - https://github.com/jisungk/deepjazz
- Learning Jazz Grammars (Gillick, Tang & Keller, 2010)
  - http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf
- A Grammatical Approach to Automatic Improvisation (Keller & Morrison, 2007)
  - http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf
- Surprising Harmonies (Pachet, 1999)
  - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf

### Week 2
- Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings (Bolukbasi, Chang, Zou, Saligrama​ & Kalai, 2016)
  - https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf
- GloVe: Global Vectors for Word Representation (Pennington, Socher & Manning, 2014)
  - https://nlp.stanford.edu/projects/glove/
- Woebot
  - https://woebothealth.com/

### Week 4
- Natural Language Processing Specialization (by DeepLearning.AI)
  - https://www.coursera.org/specializations/natural-language-processing?
- Attention Is All You Need (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser​ & Polosukhin, 2017)
  - https://arxiv.org/abs/1706.03762

