### AIFFEL Campus Online Code Peer Review Templete
- 코더 : 박태하
- 리뷰어 : 김지원

#### 요구사항

1. 기존 데이터셋을 추가 정제하고, generation 성능을 끌어올리기 위한 기법들을 실험해 모델 perfomance를 향상시켜보았는가?	
- [ SOME ] 기존 데이터셋의 문제점을 분석하고 전처리 전략을 수립해 추가 정제를 진행했다. Beam search, Top-k(p) sampling 등 최선의 디코딩 전략을 수립해 향상된 모델 추론 결과를 제시했다. BLEU, ROUGE 등 생성된 텍스트를 평가하기 위한 메트릭을 적용한 정량적인 평가 결과와 주관적인 평가를 비교분석하였다.

2. 새로운 데이터를 수집해 전처리를 수행하여 모델을 재학습시켜보았는가?	
- [ NO ] 모두의 말뭉치, AI hub 등에 공개된 데이터를 사용해 추가 데이터셋을 구축하기 위한 기준과 근거를 수립했다. ChatGPT API나 다양한 한국어 benchmark 데이터셋을 활용해 Human Feedback 을 대체할 수 있는 아이디어를 구현했다. 위를 바탕으로 SFT, RM, PPO 세 단계에 필요한 각 데이터셋을 적절히 구축하여, 모델 추론 결과와 수립한 가설을 비교해보았다.

3. 학습 전략 또는 foundation model을 변경해 모델을 재학습시켜보았는가?	
- [ NO ] 더 적절한 Instruction Tuning 기법을 적용해 SFT를 해보거나, Reward Model의 ranking algorithm을 개선해보았다. KoGPT-2가 아닌 다른 모델을 initial model로 사용하여 모델 학습을 성공시켰다. 허깅페이스의 accelerate, bitsandbytes 라이브러리 등을 사용하여 더 큰 스케일의 모델로 ChatGPT를 re-building해 모델 성능을 향상시켰다.


### PRT(Peer Review Template)

- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 다양한 디코딩 전략을 수립하는것보다 하나로 정리되었습니다.
    - top_k=50,top_p=0.95 이런식으로 hyperparameters 를 setting 하였다. BLEU, ROUGE 등 정량적인 평가 결과를 보지는 못하였다. 
    - 코드 예로는 다음과 같다: 
```python
def generation(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
        torch.cuda.current_device())
    outputs = actor.generate(input_ids,
                             max_length=250,
                             do_sample=True,
                             top_k=50, # <- 이렇게 hyperparameter 를 설정하였다
                             top_p=0.95,
                             num_return_sequences=1)
    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
    print()
    print(output)
    return output
```    
    - 다른, 보지 못한 데이터에 대해 모델 재학습은 안 하였다 
    - 또한 학습전략도 KoGPT-2 기반이 아니었다.

- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 어려운 부분에 대한 doc string 혹은 주석은 없는것 같습니다.

- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나”
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 결과에 대한 코멘트가 있습니다.

- [ ]  **4. 회고를 잘 작성했나요?**
    - 회고가 없었습니다

- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 파이선 코드 스타일을 준수하여 썼습니다
