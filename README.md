# Text-sentiment-analysis

## 개요
이 저장소는 **PyTorch 및 Hugging Face `transformers` 라이브러리**를 사용하여 **BERT 기반 감정 분석**을 수행하는 프로젝트이다. 
**IMDb 영화 리뷰 데이터셋**을 활용하여 긍정/부정을 분류하는 이진 감정 분석을 수행한다.

## 주요 기능
- **BERT (Bidirectional Encoder Representations from Transformers)**를 활용한 자연어 처리(NLP) 수행
- **토큰화, 어텐션 마스크, 세그먼트 임베딩**을 적용하여 BERT 입력 구성
- **Hugging Face `datasets` 라이브러리**를 이용하여 IMDb 데이터셋 로드
- **모델 학습, 평가 및 시각화 지원**
- **AdamW 옵티마이저 및 학습률 스케줄링을 활용한 파인튜닝 적용**

## 필수 라이브러리 설치
다음 명령어를 실행하여 필요한 라이브러리를 설치한다:
```bash
pip install torch transformers datasets matplotlib
```

## 사용 방법
### 1. IMDb 데이터셋 다운로드 및 BERT 파인튜닝
```bash
python main_all.py --epochs 3 --batch_size 16 --lr 2e-5
```

### 2. 모델 평가 실행
```bash
python main_all.py --evaluate
```

### 3. 모델 예측 시각화
```bash
python main_all.py --visualize
```

## 실험 결과
- IMDb 데이터셋을 활용한 파인튜닝 후 **높은 정확도(90% 이상) 달성**
- **matplotlib을 이용한 예측 결과 시각화 지원**

## Dataset 시각화 
Dataset 클래스로부터 4개 객체에 대해 아래와 같이 리턴받음을 직접 데이터 샘플을 출력하여 확인해볼 수 있다. 
```bash
{'text': 'There is no relation at all between Fortier and Profiler but the fact that both are police series about violent crimes. Profiler looks crispy, Fortier looks classic. Profiler plots are quite simple. Fortier\'s plot are far more complicated... Fortier looks more like Prime Suspect, if we have to spot similarities... The main character is weak and weirdo, but have "clairvoyance". People like to compare, to judge, to evaluate. How about just enjoying? Funny thing too, people writing Fortier looks American but, on the other hand, arguing they prefer American series (!!!). Maybe it\'s the language, or the spirit, but I think this series is more English than American. By the way, the actors are really good and funny. The acting is not superficial at all...', 'input_ids': tensor([  101,  2045,  2003,  2053,  7189,  2012,  2035,  2090,  3481,  3771,
         1998,  6337,  2099,  2021,  1996,  2755,  2008,  2119,  2024,  2610,
         2186,  2055,  6355,  6997,  1012,  6337,  2099,  3504, 15594,  2100,
         1010,  3481,  3771,  3504,  4438,  1012,  6337,  2099, 14811,  2024,
         3243,  3722,  1012,  3481,  3771,  1005,  1055,  5436,  2024,  2521,
         2062,  8552,  1012,  1012,  1012,  3481,  3771,  3504,  2062,  2066,
         3539,  8343,  1010,  2065,  2057,  2031,  2000,  3962, 12319,  1012,
         1012,  1012,  1996,  2364,  2839,  2003,  5410,  1998,  6881,  2080,
         1010,  2021,  2031,  1000, 17936,  6767,  7054,  3401,  1000,  1012,
         2111,  2066,  2000, 12826,  1010,  2000,  3648,  1010,  2000, 16157,
         1012,  2129,  2055,  2074,  9107,  1029,  6057,  2518,  2205,  1010,
         2111,  3015,  3481,  3771,  3504,  2137,  2021,  1010,  2006,  1996,
         2060,  2192,  1010,  9177,  2027,  9544,  2137,   102]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1]), 'labels': tensor(1)}
```

## Fine tuning 감정 분류 훈련 과정 및 평가 결과 (긍정 / 부정)
fine tuning #1
- traing data 개수 : 100
- test data 개수 : 10 
```bash
Epoch 1/3
Train loss: 0.7219
Training Accuracy: 60.00%
Test Accuracy: 40.00%

Epoch 2/3
Train loss: 0.6619
Training Accuracy: 74.00%
Test Accuracy: 50.00%

Epoch 3/3
Train loss: 0.6298
Training Accuracy: 68.00%
Test Accuracy: 50.00%
```


fine tuning #2
- traing data 개수 : 150
- test data 개수 : 10 
```bash
Epoch 1/3
Train loss: 0.7107
Training Accuracy: 79.33%
Test Accuracy: 50.00%

Epoch 2/3
Train loss: 0.5434
Training Accuracy: 96.00%
Test Accuracy: 60.00%

Epoch 3/3
Train loss: 0.2311
Training Accuracy: 97.33%
Test Accuracy: 70.00%
```

fine tuning #3
- traing data 개수 : 200
- test data 개수 : 10 
```bash
Epoch 1/3
Train loss: 0.6997
Training Accuracy: 48.00%
Test Accuracy: 60.00%

Epoch 2/3
Train loss: 0.6773
Training Accuracy: 81.50%
Test Accuracy: 70.00%

Epoch 3/3
Train loss: 0.4505
Training Accuracy: 92.50%
Test Accuracy: 90.00%
```

fine tuning #4
- traing data 개수 : 250
- test data 개수 : 10 
```bash
Epoch 1/3
Train loss: 0.6816
Training Accuracy: 82.80%
Test Accuracy: 70.00%

Epoch 2/3
Train loss: 0.3879
Training Accuracy: 83.60%
Test Accuracy: 80.00%

Epoch 3/3
Train loss: 0.2379
Training Accuracy: 97.60%
Test Accuracy: 90.00%
```

fine tuning #5
- traing data 개수 : 300
- test data 개수 : 10 
```bash
Epoch 1/3
Train loss: 0.7034
Training Accuracy: 50.00%
Test Accuracy: 40.00%

Epoch 2/3
Train loss: 0.4824
Training Accuracy: 93.00%
Test Accuracy: 100.00%

Epoch 3/3
Train loss: 0.1829
Training Accuracy: 96.00%
Test Accuracy: 90.00%
```

- traing data 개수가 늘어날 수록 훈련 및 평가 정확도가 개선됨을 확인
- test data 샘플이 너무 적어 정확도 신뢰 낮음
- 평가 샘플을 늘려서 traing data 샘플 개수와 비슷한 정도의 개수를 가지고서 평가 필요함

### 평가 결과 시각화 샘플
![image](https://github.com/user-attachments/assets/d6181521-7184-4459-af04-df27aa7eee18)

![image](https://github.com/user-attachments/assets/68b780e2-54b0-43c4-adb7-2d0bf28b8447)




## 라이선스
이 프로젝트는 **MIT 라이선스** 하에 공개된다.

## 참고 문헌
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [IMDb Dataset](https://huggingface.co/datasets/imdb)

