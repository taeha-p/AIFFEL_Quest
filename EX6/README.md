# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 박태하
- 리뷰어 : 김서연

## 평가문항	상세기준
1. Abstractive 모델 구성을 위한 텍스트 전처리 단계가 체계적으로 진행되었다.
2. 텍스트 요약모델이 성공적으로 학습되었음을 확인하였다.
3. Extractive 요약을 시도해 보고 Abstractive 요약 결과과 함께 비교해 보았다.

# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - <img width="918" alt="스크린샷 2023-10-10 오후 12 21 25" src="https://github.com/taeha-p/AIFFEL_Quest/assets/112914475/7af1d4f6-eeba-4905-abc7-44faa05fb61c">
    - <img width="899" alt="스크린샷 2023-10-10 오후 12 22 02" src="https://github.com/taeha-p/AIFFEL_Quest/assets/112914475/ed63cfc0-4567-4eec-a9ea-492bcefbb0e0">
    - 문제 해결이 잘 이루어졌다.
    
- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - <img width="910" alt="스크린샷 2023-10-10 오후 12 23 41" src="https://github.com/taeha-p/AIFFEL_Quest/assets/112914475/49307eee-67df-45ee-9a9f-fde986eb1314">
    - 어텐션 층의 입출력에 대한 부분이 잘 설명되어있다.
        
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 기존 노드 내용과 다른 새로운 시도를 해보면 좋을 것 같다(모델 설계 바꿔보기 등)
    
- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결<img width="273" alt="스크린샷 2023-10-10 오후 12 25 37" src="https://github.com/taeha-p/AIFFEL_Quest/assets/112914475/d9cdf3f9-3998-4ff1-bb1c-212fc066fa76">
    - 새롭게 배웠거나 잘 안 돼서 아쉬웠던 부분, 전체 프로젝트의 워크플로우 등에 대한 내용이 들어가면 좋을 것 같다.
        
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - ```python
      # 인코더의 LSTM 1
        encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True ,dropout = 0.4, recurrent_dropout = 0.4)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
      ```
      `state_h1`, `state_c1` 등 사용되지 않는 변수명 굳이 할당하지 않는 것이 좋다. 


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
