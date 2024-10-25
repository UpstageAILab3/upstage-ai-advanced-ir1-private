import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from datetime import date, datetime, timezone, timedelta
import time
import re
from openai import OpenAI
import traceback
import sys

# from calc_answer import calc_map
base_path = "/data/ephemeral/home/result_csv"
# set output folder
output_folder = "sample/"

# Variation
boosts = [0.0025]#, 0.0035]
question_llm_models = ["gpt-3.5-turbo-1106"]
# Sentence Transformer 모델 초기화
st_models = [#"snunlp/KR-SBERT-V40K-klueNLI-augSTS", 
            #  "hunkim/sentence-transformer-klue"#, 
            #  "jhgan/ko-sbert-sts", 
            #  "ddobokki/klue-roberta-base-nli-sts", 
             "jhgan/ko-sroberta-multitask"]
            #  "jhgan/ko-sroberta-sts"]
retrieve = ["hybrid"] #  "sparse", "dense", 
similarities = [
    # "BM25",            # BM25 similarity (default)
    # "DFR",             # DFR similarity
    # "DFI",             # DFI similarity
    # "IB",              # IB similarity
    # "LMDirichlet",     # LM Dirichlet similarity
    "LMJelinekMercer"  # LM Jelinek Mercer similarity
]
# 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
stoptags_list = [["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"],
                 ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX", "XSA"]]
# "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
# "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX", "XSA"]
#             #  "IC", "MAG", "MAJ", "MM", "SP", "SSC", "SSO", "XPN", "XSA", "XSN", "XSV", "UNA", "NA", "VSV" ]

es_username = "elastic"
es_password = ""

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="/data/ephemeral/home/elasticsearch-8.15.2/config/certs/http_ca.crt")

def create_folder_if_not_exists(a):
    # 전체 경로 생성
    folder_path = os.path.join(base_path, a)
    
    # 폴더가 존재하는지 확인
    if not os.path.exists(folder_path):
        # 폴더가 없으면 생성
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
create_folder_if_not_exists(output_folder)

# Constant

# in output 사용 토큰 제한 = 4096 (4000초반대)
# upstge ai 제공 api
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

conversation_ids = [2, 32, 57, 64, 67, 83, 90, 94, 103, 218, 220, 222, 227, 229, 245, 247, 261, 276, 283, 301]

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
# model = SentenceTransformer(models[0])

# 적절한 st_model만 필터링
# 성공적으로 초기화된 모델을 저장할 리스트
valid_st_models = []

for model_name in st_models:
    try:
        # 모델 초기화
        model = SentenceTransformer(model_name)
        # 초기화가 성공하면 valid_st_models 리스트에 추가
        valid_st_models.append(model_name)
    except Exception as e:
        print(f"모델 {model_name} 초기화 중 오류 발생: {e}")

# 초기화가 성공한 모델들 출력
print("오류가 나지 않은 모델들:", valid_st_models)

# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch] 
        # 개행 문자 지운다면 여기서 \n
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        # print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size, retrieved_docids = None):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    
    # if retrieved_docids: 
    #     print("retrieved docids: ", retrieved_docids)
    #     knn = {
    #         "field": "embeddings",
    #         "query_vector": query_embedding.tolist(),
    #         "k": size,
    #         "num_candidates": 100,
    #         "filter": {
    #             "terms": {"docid": retrieved_docids}
    #         }
    #     }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)

# 역색인 + Vector 유사도 혼합
def hybrid_retrieve(query_str, size, boost):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]
    
    body = {
        "query": {
            "match": {
                "content": {
                    "query": query_str,
                    # "boost": 0.0005
                    "boost": boost
                }
            }
        },
        "knn": {
            "field": "embeddings",
            "query_vector": query_embedding.tolist(),
            "k": 10,
            "num_candidates": 100,
            "boost": 1
        },
        "size": size
    }
    
    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", query=body["query"], knn=body["knn"], size=body["size"])



# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages, retrieve_mode=None, boost=0.0025):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create( # client - OpenAI()
            model=llm_model,
            messages=msg,
            tools=tools,
            # tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=60
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # 검색 결과 추출
        if retrieve_mode == "sparse":
            search_result = sparse_retrieve(standalone_query, 3)
        elif retrieve_mode == "dense":
            search_result = dense_retrieve(standalone_query, 3)
        elif retrieve_mode == "hybrid":
            search_result = hybrid_retrieve(standalone_query, 3, boost)

        response["standalone_query"] = standalone_query
        retrieved_context = []

        # Original - 기존
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        # # 메시지 생성 생략
        # content = json.dumps(retrieved_context)
        # messages.append({"role": "assistant", "content": content})
        # msg = [{"role": "system", "content": persona_qa}] + messages
        # try:
        #     qaresult = client.chat.completions.create(
        #             model=llm_model,
        #             messages=msg,
        #             temperature=0,
        #             seed=1,
        #             timeout=30
        #         )
        # except Exception as e:
        #     traceback.print_exc()
        #     return response
        # response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename, retrieve_mode=None, boost=0.0025):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        conversation_ids_pred = []
        for line in f:
            j = json.loads(line)
            # print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"], retrieve_mode, boost)
            # print(f'Answer: {response["answer"]}\n')
            if response["answer"] != "" and response["standalone_query"] == "":
                conversation_ids_pred.append(j["eval_id"])

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1
            
        cc = [ids for ids in conversation_ids_pred if ids in conversation_ids] # conversation_ids == conversation_ids_pred
        ic = [ids for ids in conversation_ids_pred if ids not in conversation_ids] # conversation_ids_pred - conversation_ids
        mc = [ids for ids in conversation_ids if ids not in conversation_ids_pred] # conversation_ids - conversation_ids_pred
        # 대화문 (검색 api호출하ㅣ않는 대화)이 잘 필터되었는지 확인
        # print("score : ", calc_map(of))
        print("Correct conversation ids : ", len(cc), cc)
        print("Incorrect conversation ids : ", len(ic), ic)
        print("Missing conversation ids : ", len(mc), mc)
        print("Accuracy : ", len(cc) / len(conversation_ids))

            

# Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
# 추가할 기능 :             
# 1. 영어는 한국어로 바꾸어 검색
# 2. 유사/유의어는 같은 단어로 구분
def setting_mapping(content_similarity, stoptags):
    if content_similarity == "BM25":
        custom_similarity = {
            "type": "BM25"
        }
    elif content_similarity == "DFR":
        custom_similarity = {
            "type": "DFR",
            "basic_model": "g",  # 기본 모델: g, if, etc.
            "after_effect": "l",  # l 또는 b 같은 매개변수
            "normalization": "h2"  # h1, h2, z 등의 정규화
        }
    elif content_similarity == "DFI":
        custom_similarity = {
            "type": "DFI",
            "independence_measure": "standardized"  # 다른 옵션: saturated, chisquared
        }
    elif content_similarity == "IB":
        custom_similarity = {
            "type": "IB",
            "distribution": "ll",  # 또는 spl
            "lambda": "df",  # 또는 ttf
            "normalization": "h2"  # 다른 옵션: no, h1, h2
        }
    elif content_similarity == "LMDirichlet":
        custom_similarity = {
            "type": "LMDirichlet",
            "mu": 2000  # 기본값 2000, 필요시 조정
        }
    elif content_similarity == "LMJelinekMercer":
        custom_similarity = {
            "type": "LMJelinekMercer",
            "lambda": 0.7  # 0과 1 사이의 값, 혼합 계수
        }
    else:
        # 기본값으로 BM25 유사도 사용
        custom_similarity = {
            "type": "BM25"
        }    
    print("Similarity Type : ", content_similarity)

    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter", "synonyms_filter"]
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                    "stoptags": stoptags
                        # ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]


                    # "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX", "XSA"]
                    #             #  "IC", "MAG", "MAJ", "MM", "SP", "SSC", "SSO", "XPN", "XSA", "XSN", "XSV", "UNA", "NA", "VSV" ]

                },
                "synonyms_filter": {
                    "type": "synonym",
                    "lenient": True,
                    "synonyms": [
                        "복숭아, 복숭아나무",
                        "밀물, 만조",
                        "썰물, 간조",
                        "관계, 서로 연결, 긴밀",
                        "순기능, 도움",
                        "리보오솜, 리보솜",
                        "네트웍, 네트워크",
                        "역할, 기능, 기여, 담당",
                        "전력, 전류",
                        "이루어져, 구성, 형성",
                        "땅, 토양",
                        "가장 많이, 대부분",
                        "단점, 부정적인",
                        "원자력 발전, 에너지",
                        "형태, 유형",
                        "원자, 가장 작은 입자",
                        "내부 구조, 구성"
                        "자연, 생태계",
                        "뭉쳐, 모여",
                        "날씨, 기후",
                        "측정, 추정",
                        "비만도, 체중 상태",
                        "관계, 영향",
                        "뭐야, 의미",
                        "해안, 바다",
                        "전구, 램프",
                        "번식, 생식",
                        "작은 기체 하나, 기체의 분자",
                        "다음 세대, 후손",
                        "무게, 질량",
                        "잠, 수면",
                        "약, 약물",
                        "자석, 자성체",
                        "동물, 생물체",
                        "누가, 비해",
                        "결혼 전, 혼전",
                        "숨, 호흡",
                        "인터페론, interferon", ## 여기서부터 새로 추가
                        "트로이군, trojan",
                        "슈퍼옥시드 디스무타아제, Superoxide Dismutase",
                        "드미트리 이바노프스키, Dmitri Ivanovsky",
                        "카탈로그, catalog",
                        "파이썬, python",
                        "디엔에이, DNA",
                        "이란 콘트라, 이란-콘트라",
                        "병합 정렬, merge sort",
                        "샘플, sample",
                        "연산자, operator",
                        "칼리시바이러스,칼리시 바이러스",
                        "브리지 인버터, bridge inverter",
                        "기억 상실증, 기억 상실, 기억력 저하",
                        "음파, 파",
                        "통학 버스, 학교 버스",
                        "극대화, 가장 큰",
                        "태양광을 에너지로 변환하는 능력, 광합성",
                        "자라다, 자랄",
                        "스모그, 매연",
                        "동일한, 하나의",
                        "나이가 들수록, 노인들의",
                        "학습된, 사회화",
                        "탄소의 내부 구조, 탄소 원자의 구조",
                        "생존의 확률, 생존률"
                    ]
                }
            }
        },
        "index": {
            "similarity": {
                "custom_similarity": custom_similarity  # 조건에 따라 선택된 유사도 방식
            }
        }
    }
    
    # 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
    mappings = {
        "properties": {
            "content": {"type": "text", 
                        "analyzer": "nori",
                        "similarity" : "custom_similarity"},
            "embeddings": {
                "type": "dense_vector",
                "dims": 768,
                # "dims": 1024,
                "index": True,
                "similarity": "l2_norm"
                # "similarity": "cosine"
                # "similarity": "dot product"
                # "similarity": "max_inner_product"
            }
        }
    }
    
    # settings, mappings 설정된 내용으로 'test' 인덱스 생성
    create_es_index("test", settings, mappings)



# 작동부
# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("/data/ephemeral/home/data/summarized_documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
    
for model_name in valid_st_models:
    try:
        model = SentenceTransformer(model_name)
        embeddings = get_embeddings_in_batches(docs)
        print("Sentence Transformer : ", model_name)
                    
        # 생성한 임베딩을 색인할 필드로 추가
        for doc, embedding in zip(docs, embeddings):
            doc["embeddings"] = embedding.tolist()
            index_docs.append(doc)

        
        
        for similarity in similarities:
            for stoptags in stoptags_list:
                setting_mapping(similarity, stoptags)
                
                # 'test' 인덱스에 대량 문서 추가
                ret = bulk_add("test", index_docs)

                # # 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
                # print(ret)

                # test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

                # # 역색인을 사용하는 검색 예제
                # search_result_retrieve = sparse_retrieve(test_query, 3)

                # # 결과 출력 테스트
                # for rst in search_result_retrieve['hits']['hits']:
                #     print('score:', rst['_score'], 'source:', rst['_source']["content"])

                # # Vector 유사도 사용한 검색 예제
                # search_result_retrieve = dense_retrieve(test_query, 3)

                # # 결과 출력 테스트
                # for rst in search_result_retrieve['hits']['hits']:
                #     print('score:', rst['_score'], 'source:', rst['_source']["content"])


                # 아래부터는 실제 RAG를 구현하는 코드입니다.
                
                # 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
                
                for llm_model in question_llm_models:
                    # llm_model = "gpt-4o-mini-2024-07-18" # 너무 질문 여부 파악 못함
                    print("LLM model : ", llm_model)

                    # RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
                    persona_qa = """
                    ## Role: 과학 상식 전문가

                    ## Instructions
                    - 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
                    - 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
                    - 한국어로 답변을 생성한다.
                    """
                    # # RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
                    for i in [2]:
                        if i == 0:
                    
                            # # 1. Original
                            print("function Calling : Original")
                            persona_function_calling = """
                            ## Role: 과학 상식 전문가

                            ## Instruction
                            - 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
                            - 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
                            """
                        elif i == 1:
                            print("function Calling : bit Changes")
                            # # 2. bit changes
                            persona_function_calling = """
                            ## Role: 지식 전문가

                            ## Instruction
                            - 사용자가 지식에 관해 질문하면 반드시 search api를 호출합니다.
                            - 나머지 대화에는 api를 호출하지않는다.
                            """
                        elif i == 2:          
                            print("function Calling : team 8")  
                            # # 3. team 8
                            persona_function_calling = """
                            ## Role: 과학 상식 전문가

                            ## Instruction
                            - 사용자가 지식에 관해 질문하는 경우에는 반드시 search 함수를 호출한다.
                            - 나머지 메시지에는 함수 호출 없이 적절한 대답을 생성한다.
                            """
                        elif i == 3:     
                            print("function Calling : gpt prompt")               
                            # 4. gpt prompt
                            persona_function_calling = """
                            ## Role: 지식 전문가

                            ## Instruction
                            - 사용자가 지식에 관해 질문하면 반드시 search api를 호출하여 documentary 기반의 정보를 제공합니다.
                            - 모든 정보는 검색된 documentary 자료를 바탕으로 답변해야 하며, 모델 자체의 지식은 사용하지 않습니다.
                            - 지식과 무관한 일반적인 대화에는 api를 호출하지 않습니다.
                            """

                        # Function calling에 사용할 함수 정의
                        tools = [
                            {
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "description": "search relevant documents",
                                    "parameters": {
                                        "properties": {
                                            "standalone_query": {
                                                "type": "string",
                                                "description": "User's question in Korean. Full message if the user message is single-turn." 
                                                    # "description": "Final query suitable for use in search from the user messages history."
                                            }
                                        },
                                        "required": ["standalone_query"],
                                        "type": "object"
                                    }
                                }
                            },
                        ]
                        print("EVAL")
                        for retrieve_mode in retrieve:
                            for boost in boosts:
                                print("retrieve : ", retrieve_mode)
                                start = time.time()
                                
                                # 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
                                eval_rag("/data/ephemeral/home/data/eval.jsonl", f"{base_path}/{output_folder}/{start}.csv", retrieve_mode, boost)

                                fin = time.time()
                                tat = timedelta(seconds=(fin - start))
                                
                                hours, remainder = divmod(tat.total_seconds(), 3600)
                                minutes, seconds = divmod(remainder, 60)
                                tat_hms = f"{int(hours)}h-{int(minutes)}m-{int(seconds)}s"
                                print(tat_hms)

                                current_time = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d-%Hh-%Mm")

                                os.rename(f"{base_path}/{output_folder}/{start}.csv", f"{base_path}/{output_folder}{current_time}_{tat_hms}_{model_name.split('/')[-1]}_{llm_model}_{similarity}_{retrieve_mode}_{boost}.csv")
    except Exception as e:
        print(e)