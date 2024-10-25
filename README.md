[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Tm6AYAOm)
# Scientific Knowledge Question Answering (RAG 시스템 개발 프로젝트)
## Team

| ![전은지](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이수형](https://avatars.githubusercontent.com/u/156163982?v=4) | ![서정민](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이지윤](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이승미](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [전은지](https://github.com/allisonej)             |            [이수형](https://github.com/dltngud1541)             |            [서정민](https://github.com/jmseo1216)             |            [이지윤](https://github.com/jiyuninha)             |            [이승미](https://github.com/seungmi1110)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                         |                            담당 역할                             |


- **전은지**: 팀장, 아이디어 수립 및 진행
- **이승미**: 데이터 전처리, RAG
- **서정민**: RAG 파이프라인 설계
- **이수형**: IR 담당
- **이지윤**: 데이터 분석 및 테스트

## 프로젝트 개요
이번 프로젝트는 **RAG (Retrieval-Augmented Generation)** 시스템을 사용하여 과학 상식 기반 질문에 대한 자동 응답 시스템을 개발한 프로젝트입니다. RAG는 검색엔진을 통해 문서를 추출하고, **LLM (Large Language Model)**이 이를 기반으로 답변을 생성하는 방식입니다. 이 시스템을 활용하여, 신뢰할 수 있는 출처 기반의 답변을 제공하는 것을 목표로 했습니다.

- **대회 기간**: 2024.10.02 10:00 ~ 2024.10.24 19:00
- **참여 팀**: UPSTAGE AI LAB, IR 1팀 (지식학회)

## 데이터
- **문서 데이터**: 학습 데이터는 제공되지 않았으며, 과학 상식 정보를 담은 4200여 개의 문서가 제공되었습니다.
- **데이터 형식**: 각 문서는 DOC_ID와 출처(SRC)가 포함된 JSONL 포맷으로 구성되었으며, 실제로 참조할 지식 정보는 CONTENT 필드에 저장되었습니다.

## 평가 방법
- 과학 상식 질문이 아닌 경우, 검색 결과가 없으면 1점을 부여하는 특별 평가 로직을 추가하여 평가를 진행하였습니다.
- 220개의 평가 메시지 중, 20개는 멀티턴 대화 및 일반 대화 메시지를 포함하여 다양성을 반영하였습니다.

## 아이디어
![7](https://github.com/user-attachments/assets/31d5e16d-6257-44cc-864d-136e5e66f2d5)

## 프로세스

### 1. GROUND TRUTH 제작
- 제출 횟수가 10회로 제한되었기에 그 이상으로 점수를 평가하기 위해 **GT (GROUND TRUTH)**를 작성.
- 작성된 정답지를 통해 도출된 TOPK에 대한 점수를 평가.
- 위의 결과를 통해 제출 횟수를 넘겨도 지속적으로 평가 가능.

### 2. SENTENCE TRANSFORMERS
- 여러 sentence-transformer 모델을 사용하여 sparse, dense, hybrid retrieve 실험을 진행.
- Hybrid retrieve에서 성능이 가장 좋았으며, 최종 결과에 반영.
![10](https://github.com/user-attachments/assets/2d825322-af3f-4b20-afd5-3d627c3a86f1)

### 3. HYBRID RETRIEVE
- 역색인과 VECTOR 유사도 혼합을 이용한 Hybrid Retrieve 검색 사용.
- 여러 파라미터 값을 수정하며 실험 진행.
![11](https://github.com/user-attachments/assets/6d24012e-5197-42dc-99ff-406a7595e1dd)

### 4. STOPTAGS
- ELASTICSEARCH NORI 형태소 분석기의 **STOPTAGS**를 사용하여 문법적 요소나 조사 등을 제거하거나 필터링.
- 베이스라인에서 "XSA" 추가
- `"stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX", "XSA"]` 설정.

### 5. SYNONYMS_FILTER
- 검색 시 여러 단어나 표현을 동의어로 처리하여 검색 정확도와 확장성을 높임.
- 영한/한영 번역 문제 해결을 위해 동의어 필터 적용.

### 6. FINE TUNING
- 한국어 임베딩 모델을 학습하여 문서의 CONTENT에 대한 임베딩 능력을 강화.
- 각 문서에 대해 가상의 질문 3개씩 생성하여 학습 데이터를 구축한 후 파인튜닝.

### 7. COLBERT
- BERT 기반의 의미 검색 모델을 사용하여 검색 성능 향상 시도.
- FINE_TUNING과 비교하여 성능이 향상되었으나 ElasticSearch에 비해 우위성을 보이지 않음.

### 8. RERANKING
- Huggingface 'Dongjin-kr/ko-reranker' 모델을 사용하여 최종 TopK 5개의 문서 ID를 반환.
- final 기준 약 0.08 성능 향상.
![17](https://github.com/user-attachments/assets/e6b99605-b5bd-4158-a890-b12d486245f3)

### 9. NEO4J RERANKING
- 문서 간 유사성을 계산하여 그래프를 구현한 뒤, 1차 또는 2차 검색을 통해 같은 범주의 문서 검색 능력 향상 시도.
- 시간 소요 및 자원 소비 증가 문제로 인해 적용하지 않음.

### 10. CHUNKING
- 문서마다 내용을 자르기 위한 문맥 기반 Chunking 기법 사용.
- final 기준 약 0.03 성능 향상.
![19](https://github.com/user-attachments/assets/331a94bd-ca3f-4f10-8337-14c2108ed60c)

### 11. 핵심내용 추출
- **문서 요약**: GPT-3.5 Turbo 모델을 사용하여 문서 내용을 간결하게 요약.
- **키워드 추출**: RAKE 알고리즘을 사용해 문서에서 상위 5개의 핵심 키워드를 추출.

### 12. CATEGORY 분류
- **LDA 모델**: 데이터 전처리 후 개선되었지만 다른 방식에 비해 성능이 떨어짐.
- **GPT**: 천문이나 지구과학 관련 내용에서 제대로 분류가 되지 않는 경우가 존재.
- **SRC**: ARC_Challenge가 포함된 데이터셋의 경우 다양한 주제가 섞여있어 GPT로 추가 분류 진행.
- 이 외에 클러스터링, BERT 모델을 사용해서 분류해 보았지만 제대로 분류되지 않음.

### 13. PROMPT ENGINEERING
- **Function Calling**: LLM에서 필요한 경우 검색 API를 호출하도록 설정. 과학 지식 전문가를 지식 전문가로 변경.
- **Query Rewriting**: STANDALONE_QUERY에서 더 정확한 검색을 위한 FINAL QUERY로 변경.
- **Persona QA**: 검색을 제외한 생성형 답변은 시간 소요와 토큰 사용을 줄이기 위해 주석 처리함.

## 결과

### 중간 최고점 방법
1. Hybrid_retrieve(`"query": "boost":0.0025, "knn": "boost":1`)
2. Standalone_query -> long_question 변경
3. Setting 값 추가 및 변경 (filter: 1. stoptags 2. synonyms_filter)
4. Mapping 값 similarity 변경: BM25 -> lm_jelinek_mercer
5. Prompt 변경

### 최종 최고점 방법
1. Hybrid_Retrieve(`"query": "boost":0.5, "knn": "boost":1`) 을 통해 100개 후보군 찾기
   - 특정 도메인의 키워드를 잘 잡기 위해 sparse_retrieve의 boost값을 0.0030에서 0.5로 비중을 크게 설정.
   - 대신, dense_retrieve의 영향력이 작을 수 있기에 hybrid_retrieve의 size를 100개로 크게 잡아 큰 범위의 후보군 생성.
   - Reranking을 통해 최종적으로 문맥을 더 고려한 정렬.
2. standalone_query -> long_question 변경
3. 프롬프트 수정 (과학 지식 -> 지식)
4. 'Dongjin-kr/ko-reranker' - Reranking 모델 이용 - 최종 topk 5개 docid 출력

## 회고

### 아쉬운 점
- 중간 점수는 오르나 최종 점수는 하락하는 ‘점수가 평가 기준이 되지 않는 상황’
- "대회에서 주어진 eval 데이터는 실제 사용자가 입력할 질의에 비해 지나치게 정제된 데이터로, 현실적인 사용 환경을 충분히 반영하지 못한 점이 다소 아쉬웠습니다.
- 강의 영상만으로는 성능 향상을 꾀하기 어려웠다.
- 라이브러리나 다른 방법으로 해결하지 못하고 synonym filter로 비현실적 처리.
- ColBERT 설치와 같이 강의에서 배웠지만 서버에 설치하기 매우 어려운 경우.
- 답변 생성을 위해 소모되는 토큰(GPT)의 양이 생각보다 컸음. GPT-3.5가 아닌 4.0을 사용하기에는 부담스러움.
- 대회 이후에 새롭게 알게 된 방법으로 다시 시도해보고 결과 확인할 기회가 없음.

### 잘한 점
- 처음에 아이디어를 빠르게 수립하여 시도해볼 방법들을 공유하고 각자 빠르게 테스트를 진행함.
- 매일 각자의 주 작업이 무엇인지 알 수 있음.
- 최대한 많은 Idea를 제시하고 시도를 해보았다.
- 토론을 통해 새로운 아이디어를 창출함.
- 검색 뿐만 아니라 커뮤니티에 문제 상황을 공유하고 해결함.
- 정확한 비교를 위해 변수 통제를 철저하게 하였다.

### 발전 가능성
- 중간 점수에서 LLM rerank가 안 좋은 것으로 보여 삭제했으나 실제로 좋은 결과를 가짐.
- inverter를 인버터로 인식되지 않고 Dmitricus를 한국어로 검색되지 않는 문제 해결.
- 음파와 파를 synonym 처리했으나 문맥적 처리가 필요.
- 문서가 포함한 내용을 포함한 정보를 content뿐만 아닌 또 다른 색인 자료로 활용.
- 짧은 문서에서는 적용하기 어려울 수 있지만, 긴 문서에서는 semantic Chunking, sliding window가 매우 효과적일 것이라 예상됨.
- 특정 항목들이 꾸준히 풀리지 않는 문제들이 있으므로 그 부분을 중점으로 해결 (ex. 215,91,84,230,82,306,80,47,303,273)
- 역색인 boost값 등 값의 최적값 찾기
  
## Result

### Leader Board

#### 중간 리더보드
<img width="680" alt="중간" src="https://github.com/user-attachments/assets/6a09b81e-57a4-4305-9341-3d24859c1eba">

#### 최종 리더보드
<img width="680" alt="final" src="https://github.com/user-attachments/assets/1f558cc5-4f78-4b4a-9b01-04da71f7732e">

### Presentation
[IR_1팀_발표자료.pdf](https://github.com/user-attachments/files/17515696/IR_1._.pdf)


## etc

### Reference

- https://github.com/8pril/upstage-ai-final-ir
- https://github.com/UpstageAILab/upstage-ai-final-ir1
- https://learn.microsoft.com/ko-kr/azure/databricks/generative-ai/tutorials/ai-cookbook/fundamentals-data-pipeline-steps
- https://developers.google.com/search/docs/fundamentals/how-search-works?hl=ko

