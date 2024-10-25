import pandas as pd
import json
import os

# 파일 경로 or 폴더 경로 둘 다 ok 
path = "/data/ephemeral/home/result_csv/sample"

get_result_of_analyze = False
view_and_0_filter = False
chk_question_or_not = False
get_conetnt = True

conversation_ids = [2, 32, 57, 64, 67, 83, 90, 94, 103, 218, 220, 222, 227, 229, 245, 247, 261, 276, 283, 301]

document_json_path = "/data/ephemeral/home/data/documents.jsonl"

gt = { #eval_id : topk
        78: ["c63b9e3a-716f-423a-9c9b-0bcaa1b9f35d"],
        213: ["79c93deb-fe60-4c81-8d51-cb7400a0a156"],
        107: ["25de4ffd-cee4-4f27-907e-fd6b802c6ede"],
        81: ["bd91bda8-351e-4683-bb1a-8254f93e2376"],
        280: ["38686456-b993-4cbb-af0d-1c53df2f3e12"],
        10: ["99a07643-8479-4d34-9de8-68627854f458"],
        100: ["d9ce8432-f72e-4253-9735-98318a6f9f7f"],
        279: ["0f0dd1ae-a36c-4c97-9785-4698400c67b1", "de1ab247-9d48-48f7-8499-31606f53c108"],
        42: ["4b49f3a2-32c9-4b2e-89c4-4719f98e7a74"],
        308: ["72c780ec-57bb-4fe1-976a-d7ee3d3dbb52"],
        205: ["ae28101b-a42e-45b7-b24b-4ea0f1fb2d50"],
        289: ["1f442344-084b-44f8-838b-332be289083c", "421aac6b-49ce-4697-a68f-850152f323d7"],
        268: ["7c14c33c-f15c-41f6-ab5c-4d37c2eb3513"],
        18: ["63846d07-8443-4bf8-8cd9-bc6cc7826555"],
        9: ["bfbba89d-fdaa-400d-8848-ca1ff8d51cd7"],
        101: ["144f5e5e-8069-425f-80b3-6388195ba4ee"],
        236: ["2077ea5b-53ac-4242-bbfc-20005ad63db8"],
        59: ["82b095fd-2fb6-48ae-8476-2f457f8b6650"],
        25: ["35c5dcc7-4720-4318-901e-770105ae63fd"],
        5: ["84f3f0e3-7ff2-4090-9961-aa7bbe8ca412", "59d5d7bb-6700-40ad-8884-ff43b1a9a1a0", "abf99ff1-d6bf-4020-b752-da7cb8611915"],
        104: ["73089763-06d2-4395-b235-aa3e6a399531"],
        276: [],
        14: ["2077ea5b-53ac-4242-bbfc-20005ad63db8"],
        270: ["a729b4f2-c734-4c60-9205-1518ba762593", "191c4b9f-6feb-49dd-90ad-9f2eebb6113e"],
        238: ["f48600d6-e492-43eb-b564-1860aa81da5f"],
        269: ["05b5a4f4-b115-4b76-9fe1-b80c4498289b"],
        43: ["cefe7caf-6cd1-422a-b41e-e82b543556e9", "8a78364e-63bf-4915-b718-fdc461bc62c9"],
        65: ["0bda5010-9ac6-447b-b484-60e380f4921d", "f5f54058-8c3c-4f6b-9549-db99b17685ed", "d9492876-df4d-4570-a58d-5a0438315fc9"],
        97: ["1655c90b-29c7-47ef-a092-01f2550db3aa", "85d28a10-9380-4afe-afef-b34449ef86bf"],
        206: ["70d104b2-8d74-4799-a09d-5a4c8dd577c0"],
        21: ["7150c749-dff2-4bd5-90ff-ff1e1cda468b"],
        221: ["8ae1234f-2a28-4069-a017-e99de5d67cc6"],
        71: ["5043c033-841c-46dd-94a3-1b5bef034c62"],
        254: ["af966ff7-109a-4c28-a644-393f5333ce69"],
        226: ["e21aceaf-be57-426c-b999-7ee8a309db36"],
        241: ["468d098e-2322-4950-ac11-9756f3112944"],
        261: [],
        45: ["41ca41ac-66e3-4a6b-a604-87bf8b3a8d4d"],
        19: ["d8ad7175-469b-45b5-8eb9-69504cd04f0f"],
        210: ["4764014a-4240-4c65-aa92-20eb1369a2f7"],
        231: ["5392d86a-bc7a-46c3-8272-94d982a65eed"],
        233: ["029064ed-d9f6-42cd-9b86-88cc9f611414"],
        263: ["1a277fb7-4cd7-409b-9f28-d83cef78ca10", "c8fd4323-9af9-4a0d-ab53-e563c71f9795"],
        201: ["469e37c0-a241-4675-8b1a-aa31d11a438c"],
        293: ["36788458-5fb5-4bcd-be02-3a47e5c8c19d"],
        208: ["b22a35e9-244e-44d6-b7bf-97f3ff664866"],
        282: ["2fb58e26-5ea0-4b50-b80d-4b03640042b4"],
        62: ["e7fef7f2-2549-499b-8b36-e4628119d352"],
        55: ["21383ddc-b6bb-4cf7-8815-139a3c4d9fae"],
        257: ["c8bd9b15-8ce0-4307-9f49-0f205217178f"],
        58: ["25bf6c36-116f-42c9-9d1c-c179e6292a34"],
        283: [],
        32: [],
        94: [],
        15: ["26cb5bba-0b80-41d4-9e42-aada06c879ca"],
        4: ["aa674ad5-ae70-4223-8685-e717a27dc1b3"],
        300: ["41ce1303-0091-4414-b26d-18f66101a99f"],
        243: ["5ff8f00a-a4e6-43fd-8616-3104a4c4d637"],
        34: ["55726582-8401-4a6f-889b-e9bd3953be7c", "cc6c9dc5-4d30-4653-bee7-9f3ba90fbf48", "6335780f-292d-49da-a79c-3eee1d51a903"],
        246: ["cc56ca24-fde0-458d-95f7-d3d31b79acb5", "ec539caa-4b62-4b5f-8428-489809f80611"],
        212: ["b303e4ec-87fb-40f9-8704-9037bad5af8a"],
        214: ["2213e6c8-ebb4-4cb5-b4c4-4a1773c63bcc"],
        259: ["dbfa9bbf-e2da-4d01-8aea-d6f25d43ffcf"],
        267: ["c3829f80-57bf-4db6-9e06-dd81b8bb6148"],
        90: [],
        66: ["ec87a926-171d-4f62-9acc-1b870c010a16"],
        20: ["ad5d883f-4352-4d25-ba7b-cbb605c73662"],
        24: ["bf4977ff-8fa5-4e82-b957-b6955c5bcbf0", "77a60236-e5b8-4c86-a422-1f3fa9726492"],
        281: ["de5d9a1f-67f6-45ea-a4a4-0540f5b7583e"],
        264: ["31e7fe84-bd3a-4b49-8adc-cc2e1f7a5b42"],
        79: ["c59f4bbd-71f6-4073-807f-09e8f0d3efcc"],
        304: ["ff7e8f8e-8d03-419f-bd98-93e3651fd01a"],
        292: ["8eac310f-a32f-462b-8ca8-da7c8fcd41e7", "655f64b4-74e8-4f5e-9b49-13a976ad3ee4"],
        215: ["ae30b754-a275-43dc-a2c9-95ab33a7c557"],
        225: ["6fec9716-f49f-40e9-913e-db3f4df46379"],
        108: ["d23082b4-ab12-4bcb-8658-d54ba791f263"],
        98: ["5db9db05-7e10-41d4-8f1f-81259fdc8ea7"],
        38: ["d525c38e-a296-4dec-8a60-d1fb98a355e3"],
        287: ["79c61856-f978-48c5-a7a8-0863bc661106"],
        295: ["6a8852ee-b847-4bc8-9b22-a9b47b56181e"],
        255: ["62cb0474-809d-4554-82b8-861ba23d8cb4"],
        76: ["6c9bca81-11e0-41a4-b894-3bcd66dc2bf6"],
        248: ["b39fe7fd-1fa8-46ef-ba31-f4464727d56e"],
        85: ["15f317a8-1d4d-41c4-93ec-bc9be43c3984"],
        290: ["93d1aa65-a113-4dd2-8959-e8204bfea616"],
        73: ["0d316cb2-c466-4b2b-b28b-0a726d513ef2"],
        309: ["395d6c0b-a199-451e-81ed-b49bfd853927"],
        239: ["2ead4363-f521-4377-b71c-f380fd9f5094", "deb0df64-4649-4785-a87a-2ad90a819c25"],
        284: ["552fc915-5ee7-4d6f-a45f-86e8e6a5d02c", "6b11c698-eedd-4731-ae6f-c91a327c4725"],
        220: [],
        96: ["810e57ae-aab1-470d-b078-435beb1b5ce8"],
        217: ["0a29b75f-af58-4b6a-a124-0d71ffbcbc89"],
        6: ["1a707124-fe53-45aa-baaa-e3e8d6697742"],
        237: ["59d9a47e-3f46-4caa-b4e9-9d3056b3e453"],
        245: [],
        35: ["02759425-149b-467e-af41-11f924577549", "4392ecba-ea79-496b-8931-e1a79caad179", "c6f86dc0-0ecc-4232-87ad-d12dea2e4c5b"],
        242: ["209121c3-69df-414a-a1b9-c74a1b14384f"],
        296: ["2570699b-b8c3-409e-b797-a1ee57c1fd86"],
        307: ["5a8089d2-36d8-4734-a16c-85db8226bfd3"],
        229: [],
        30: ["f532c956-a335-4688-9078-b1a13a0cfe58", "cc9425b7-beaa-4bd9-9ed4-96cd82bda481"],
        16: ["21737d6c-8ec9-40fd-80ed-50548228f0e9"],
        247: [],
        285: ["b38604f4-94bd-4482-97df-7a768331558a"],
        77: ["5b8fa65f-9a69-4990-8618-50aa1911e3ec"],
        250: ["1f70c6f1-f1a7-44be-90d6-d397bace344b", "a06ea0ff-67d9-4967-bbe2-0d9022551740"],
        68: ["59ce17a4-82be-4e0c-adbd-3ef4fd4e8a33"],
        86: ["d2762c95-6397-44b5-b20e-81d4b333dd69"],
        232: ["af4a89a5-4fe3-4655-88d7-fd2fcd118441"],
        109: ["00774f87-0573-4dd8-b4dc-c43ce406a51f"],
        1: ["4a8549c8-8e81-4804-9df9-852952dd5747"],
        28: ["03390467-c6dc-4242-a039-c7e9a8bb242f"],
        203: ["2e0e35c2-db89-4cfd-b782-47b1acd8404d","f81c0a09-4082-4b6a-8546-12371ff89fba"],
        91: ["ec326ad8-286b-4f58-9a03-31b94e969f33"],
        105: ["b7a0428e-c402-43cb-8e89-3d5f9a7f644b"],
        67: [],
        13: ["30ee5c53-1559-42f5-94c3-2ffb0b9682aa"],
        57: [],
        301: [],
        266: ["bf023a9d-47f3-4d77-8bf1-5a854b5403a7"],
        2: [],
        253: ["1c5a826a-2475-478c-b79b-eb48a59e0f5d"],
        49: ["74b5d96e-ea97-4476-850b-f78057d7c457"],
        294: ["c53b9d18-1192-4e28-843f-5ae62685a4be"],
        17: ["4d21e0e7-b334-4fe3-9eb2-bf1cbd4ef232"],
        207: ["2e0e35c2-db89-4cfd-b782-47b1acd8404d"],
        227: [],
        204: ["14eab0c3-51d8-48e0-9703-47f34f64bbd2"],
        89: ["29fade0c-58e6-4ffd-8d24-5f2fb0744790","59952996-37f3-412e-a74c-a13f0cd1aafa","6f8d2975-3a40-441e-9dd2-c555d7d2fd85"],
        224: ["7c75080c-5d8f-41e2-ad26-c90ad3f671b0"], #"f75d3717-24c8-41e6-a78d-ea6f92444209"
        256: ["f1a89f8a-7510-4400-8fbe-20bcb0a2adc3","10668a9a-0c49-4078-89ae-cade2913f3e5"],
        291: ["84a2a25e-6247-4e3b-9596-27b1c43812e2"],
        8: ["95cf4ddd-4a18-4e75-bc7d-0cc10964e52e","d5569147-478a-4b93-b5f1-19dff5e4c092"],
        106: ["34965328-bc80-41a0-ba1a-daece23d4be7","fc408e3d-9c04-44c4-89e4-139cacce27e3","b2e0e809-c9e9-4465-9248-07a9b49b034f"],
        84: ["a81ef669-2b8d-41d0-a8ff-2ab5da089488","75e95fab-8764-4703-a1e0-79953943744f","daefb46c-00c4-4e12-8996-5a4134e28fc9"],
        70: ["7ec6e16f-9a7b-435e-a80d-2a1201f9f644","6d88e982-53a9-44f7-896d-d0f949a07237"],
        230: ["6dad80e2-0a4a-44ee-b6e5-748eb3033320"],
        92: ["3f2e5ed1-2522-4421-896f-86a6dbd996f5"],
        72: ["c8fd4323-9af9-4a0d-ab53-e563c71f9795"],
        211: ["a91d0f81-373c-414b-9682-8ded87f2984d"],
        258: ["627f33cc-dee7-46d1-b392-eb1a5fea3026"],
        51: ["36c3b7a5-e415-453a-a43d-f620d652b29a"],
        200: ["d1242985-4e58-41ad-bb80-86a34a6399f4"],
        274: ["5b269648-5d9f-4043-bcf5-962a8e5913e4"],
        41: ["2d1cca9c-238e-439e-ac7a-37295202ceff", "a4e26513-2c87-4acd-8675-11ccb2dcfa67"],
        87: ["c649c195-d0c2-4e1a-988b-0111661810b8","304e9cf1-a292-40d2-8b11-a936f8208ae9"],
        235: ["68ec9c78-13f8-49b9-96d9-b3040a2f4771"],
        262: ["6bc8a76c-de25-49f0-ab05-aee03b565c89"],
        288: ["9954d917-170e-477f-84c9-6d1d42927eed"],
        60: ["917b7cf4-29c9-4a12-bd8d-2421e2fd98e0"],
        222: [],
        46: ["bd5ffdb4-0bc5-41a0-a779-27b0f63fc61a"],
        234: ["7413982d-c207-4bcb-9489-13ef8c761fa4"],
        23: ["a3c79177-5bfc-4d31-a542-fa6d9cd38c22"],
        82: ["6a8b4d1c-1b9e-4f6d-8a01-8f48934ceb7d","5057d6af-55d5-4a91-9e86-832bdd14f701","48a8e51f-0ff4-479f-8b88-382f70133886"],
        61: ["e139e58c-9c3b-4b96-b393-1929ef04f553"],
        56: ["fdf2729f-f2f6-4aef-bb57-152a4215e5df"],
        83: [],
        99: ["1a4899b4-8df8-48ff-bc25-80bb75633c1e"],
        75: ["530c19d3-a054-4dea-a621-152f71873465"],
        299: ["91a2fe6f-a626-499e-8111-919e7620ca90"],
        306: ["416e57d1-d61f-4ab5-a75f-95d53f8a0fce"],
        80: ["0d7e7394-4e11-4825-8243-750088162234"],
        64: [],
        271: ["c528c66d-07cc-4fc1-976d-631b76dddc58","0598d1c1-f304-47c2-927c-6076838a69e8"],
        39: ["fc0cbdb8-0add-4931-8591-a21ecbfa5b35"],
        219: ["d41d40d7-8914-48dc-a3ee-c95afcad4f95"],
        273: ["9ae85189-ebe3-40d4-9c17-1523246e7833"],
        12: ["9b7bc48a-6e0c-450c-8364-a7da2ac4d4d4"],
        33: ["eab2a705-18b6-4bbe-84f4-acacb4cd8a1f"],
        240: ["01684b2f-9360-4d70-93b1-05adb631aa62"],
        252: ["9f51187f-8895-43a5-820f-790a9e36d49a"],
        69: ["fdb1fef4-b0ff-42c8-b326-5279ba661a90"],
        47: ["cb4a3d86-988e-4425-b295-bbe34317c5cb"],
        0: ["e85754e3-12e4-44ec-a2c4-78a5b18b8aa7"],
        52: ["70fd8635-e601-421c-9025-64fcbb2d13be"],
        103: [],
        209: ["3149730b-15c0-4e4b-a8d4-0448a477238e"],
        251: ["8b22b0a3-63f8-4b38-9d98-0af90499e212"],
        275: ["dd00abd6-626b-4741-83d1-5b36fa36a635"],
        48: ["3f5468bc-4d08-445d-8f27-13034e402d01"],
        36: ["36450671-2474-4e24-867c-46cab8f53a77"],
        218: [],
        298: ["531fef8d-f493-42bc-980f-4a86f91bc5ea"],
        40: ["464ace62-ddf2-423d-a5d7-2f17e6785c8e"],
        260: ["e098388d-e054-4be2-8ebe-ce680360009c","b8912c36-293d-41be-8c13-4db3b61d2a1e"],
        11: ["ec301d79-dd16-4e88-b389-eb5f12bafef8"],
        249: ["e9a3297d-f12b-4f25-94a2-78c5c2393ba7"],
        22: ["a2f73ce5-2117-4e61-b8fb-17a1c21dad37"],
        93: ["74736be6-4150-40db-93c1-a28cf671edb3"],
        31: ["8918f0f9-795d-4f15-8982-2cede58c202e","4e7af464-9ff0-499f-913c-79971d6b1c77","81762624-6b09-44e3-b5fb-973ed8f5b7de"],
        50: ["147f526c-82c9-4001-8e68-77c909b73491"],
        277: ["ba29d074-70ca-4cbc-a786-aad0870eecf1"],
        27: ["0300c15e-5e76-4b43-aead-f80319f536dd"],
        272: ["2276e958-d625-450a-8bee-b06b762e0b16"],
        54: ["39821ad7-cabe-4ccc-a6bc-4c0a9592fa4a"],
        302: ["b0f3aa6c-8e92-4044-8e53-571e38d9451c"],
        3: ["a5097fd3-23ee-40a3-81d8-2a909cc202ee","dfbf8b98-4fb7-4d72-a382-615f66b4c9ca","254558bf-c608-47aa-952e-3f94805911f3"],
        202: ["b782c2de-0398-423b-8bc0-fc296ed74e4a"], # 이거 정답 아닐 확률 높음
        74: ["538f5608-6b05-498e-8b85-88d14723801c"],
        44: ["8ce9b365-0ab9-4230-af45-6cdfd9a4c25b"],
        63: ["157054e4-e5a1-4cd7-8bca-08ba63248790"],
        37: ["497d109c-5076-4287-a612-cc9f885150d9"], # 이거 정답 아닐 확률 높음
        29: ["f50cc54c-c51f-45d2-958e-bba326e33417"], # 매우 애매
        102: ["6e3b0253-50a3-4ab5-b387-2af50eccf264"],
        228: ["74a4c893-af24-42d3-b48f-ae6738295200"],
        297: ["bcb32b8a-4ba8-47c6-b565-04569f496a26"],
        278: ["10ad718b-f08c-405c-bffe-b819f64ad012"],
        53: ["34d9ff5e-ccde-4e54-bd7e-e400c925bf62"],
        26: ["3ee43354-16a6-4416-87f5-e380da6dbbe1","7a7412db-e9f4-4797-909b-dc8109011540"],
        265: ["56e5d264-6f82-4e01-8a96-5f369204dc44"],
        216: ["185008d2-6091-4beb-94e4-d847d70fdea6"],
        95: ["647f45f2-bafd-4b6e-a100-2ae559cc8925"],
        305: ["613bcebe-d2b1-4611-9cd5-ac904cd3361f"],
        7: ["281273f5-0adf-4bf7-9b3e-26c77b894ec7","e4d075ab-c60b-4b22-bb31-ea0ca56e1faf"],
        88: ["27422081-5e3e-4716-b7c2-883e7961640d"],
        244: ["fed07bdf-90e8-4236-bc5e-43eb5fe17061"],
        223: ["e2161953-e80b-41b3-a83b-5bc7d76a801c"],
        303: ["ad9423ec-d8cb-4972-b395-871db5cb9f45"],
        286: ["466f5811-9b37-4589-9571-bafadd92c349"],
        266: ["bf023a9d-47f3-4d77-8bf1-5a854b5403a7"],
    }
# mAP claculation
def calc_map(pred):    
    sum_average_precision = 0    
    for j in pred:        
        if gt[j["eval_id"]]:            
            hit_count = 0            
            sum_precision = 0
            # print(j["topk"][:3])            
            for i,docid in enumerate(j["topk"][:3]):                
                if docid in gt[j["eval_id"]]:                    
                    hit_count += 1                    
                    sum_precision += hit_count/(i+1)            
            average_precision = sum_precision / hit_count if hit_count > 0 else 0        
        else:            
            average_precision = 0 if j["topk"] else 1        
        sum_average_precision += average_precision    
    return sum_average_precision/len(pred)

def load(file_path):
    # 폴더 경로
    folder_path = os.path.dirname(file_path)

    # 파일 이름 (확장자 제외)
    file_name_without_ext, file_ext = os.path.splitext(os.path.basename(file_path))
    
    # 파일 읽기
    df = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                df.append(json.loads(line))  # 각 줄을 JSON으로 변환
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON on line {line}")
                continue  # JSON 변환에 실패한 라인은 건너뜀
    return df, folder_path, file_name_without_ext



def load_documents(document_json_path):
    documents = {}
    with open(document_json_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            documents[doc['docid']] = doc['content']
    return documents

def get_result(df, folder_path, file_name_without_ext, get_content_of_first):
    # 다음 내용을 저장 : 'eval_id', 'standalone_query','topk', 'gt_topk', 'common', 'added', 'missing'
    
    # CHECK : id mismatch 
    for data in df:
        eval_id = data.get("eval_id")
        if eval_id not in gt:
            # 비교 로직 추가: GT와 topk 리스트 비교
            print(f"Eval ID: {eval_id} - GT: {gt[eval_id]} vs TopK: {data.get('topk')}")

    # mAP 계산
    map_score = calc_map(df)
            
    # gt 데이터와 JSON 파일 데이터 비교
    results = []
    
    for item in df:
        eval_id = item.get('eval_id')
        standalone_query = item.get('standalone_query')
        topk = item.get('topk', [])
        
        # ground truth에서 해당 eval_id에 대한 값 가져오기. 
        gt_topk = gt.get(eval_id, None)
        
        if gt_topk is None:
            print(f"eval_id {eval_id}는 gt에 존재하지 않습니다.")
            if get_content_of_first:
                results.append((eval_id, standalone_query, topk, gt_topk, "", "", ""))
            else:
                results.append((eval_id, standalone_query, topk, gt_topk))
        else:
            # topk에 다음 경우 분유하여 result에 추가 : pred in ans, pred not in ans, ans not in pred, pred == ans
            # 집합으로 변환하여 비교
            topk_set = set(topk)
            gt_topk_set = set(gt_topk)
            
            # 공통된 값 (교집합)
            common_values = topk_set & gt_topk_set
            
            # topk에 있지만 gt_topk에 없는 값 (차집합)
            added_values = topk_set - gt_topk_set
            
            # gt_topk에 있지만 topk에 없는 값 (차집합)
            missing_values = gt_topk_set - topk_set
            
            # 결과 추가
            if get_content_of_first:
                documents = load_documents(document_json_path)
                topk_content = ""
                gt_topk_content = ""
                if topk and topk[0] in documents:
                    topk_content = documents[topk[0]]
                if gt_topk and gt_topk[0] in documents:
                    gt_topk_content = documents[gt_topk[0]]
                    
                # 결과 추가
                results.append((eval_id, standalone_query, topk, gt_topk, 
                                list(common_values), 
                                list(added_values), 
                                list(missing_values), 
                                topk_content,
                                gt_topk_content))
            else:
                results.append((eval_id, standalone_query, topk, gt_topk, 
                                list(common_values),
                                list(added_values),
                                list(missing_values)))
    
    # DataFrame으로 변환 후 CSV로 저장
    included_file_path = f'{folder_path}/{file_name_without_ext}_{map_score:.5f}_results.csv'
    included_df = pd.DataFrame(results, columns=['eval_id', 'standalone_query', 'topk', 'gt_topk', 'common', 'added', 'missing', 'top1_content', 'gt_top1_content'])
    included_df.to_csv(included_file_path, index=False)
    print(f"결과가 {included_file_path}에 저장되었습니다.")
    
def viewer_0filter(data, folder_path, file_name_without_ext):
    # 다음 내용을 저장 : 'eval_id', 'standalone_query','topk', 'gt_topk', 'common', 'added', 'missing'
    # 결과가 이상한 것들만 추출

    # eval_id, topk (3개), answer, references (score만 3개) 추출
    result = []
    for item in data:
        topk_values = item.get('topk', [])
        references_scores = [ref.get('score', 0) for ref in item.get('references', [])]  # references에서 score만 추출
        
        result.append({
            "eval_id": item['eval_id'],
            "standalone_query": item['standalone_query'],
            "topk_1": topk_values[0] if len(topk_values) > 0 else None,
            "topk_2": topk_values[1] if len(topk_values) > 1 else None,
            "topk_3": topk_values[2] if len(topk_values) > 2 else None,
            "answer": item.get('answer', ""),
            "reference_score_1": references_scores[0] if len(references_scores) > 0 else None,
            "reference_score_2": references_scores[1] if len(references_scores) > 1 else None,
            "reference_score_3": references_scores[2] if len(references_scores) > 2 else None
        })

    # DataFrame으로 변환
    df_result = pd.DataFrame(result)

    # topk3가 모두 NaN인 행만 필터링
    df_filtered = df_result[df_result[['topk_1', 'topk_2', 'topk_3']].isna().all(axis=1)]

    # json형태를 변환하여 CSV 파일로 저장
    output_file_path_1 = f'{folder_path}/{file_name_without_ext}_transformed_with_topk_scores.csv'
    df_result.to_csv(output_file_path_1, index=False)

    # 필터링된 결과를 새로운 CSV 파일로 저장
    output_file_path_2 = f'{folder_path}/{file_name_without_ext}_empty_topk_results.csv'
    df_filtered.to_csv(output_file_path_2, index=False)

    print(f"json형태 to csv형태 변환 결과가 {output_file_path_1}에 저장되었습니다.")
    print(f"topk3가 모두 빈 칸인 행이 {output_file_path_2}에 저장되었습니다.")
    
def communication_check(data, folder_path, file_name_without_ext):
    conversation_ids_pred = []

    # conversation_ids_pred 추출
    for item in data:
        if item["answer"] != "" and item["standalone_query"] == "":
            conversation_ids_pred.append(item["eval_id"])

    # 정답 conversation_ids와 비교
    cc = [ids for ids in conversation_ids_pred if ids in conversation_ids]  # Correct conversation ids
    ic = [ids for ids in conversation_ids_pred if ids not in conversation_ids]  # Incorrect conversation ids
    mc = [ids for ids in conversation_ids if ids not in conversation_ids_pred]  # Missing conversation ids

    # 결과 출력
    correct_len = len(cc)
    incorrect_len = len(ic)
    missing_len = len(mc)
    total_len = len(conversation_ids)
    accuracy = correct_len / total_len if total_len > 0 else 0

    output = (
        f"Correct conversation ids : {correct_len} {cc}\n"
        f"Incorrect conversation ids : {incorrect_len} {ic}\n"
        f"Missing conversation ids : {missing_len} {mc}\n"
        f"Accuracy : {accuracy:.2f}\n"
    )

    # 결과를 출력
    print(output)

    # 결과를 txt로 저장
    output_file_path = f"{folder_path}/{file_name_without_ext}_conversation_check.txt"
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"결과가 {output_file_path}에 저장되었습니다.")
    
if os.path.isfile(path):
    print(f"{path}는 파일입니다.")
    
    data, folder, file = load(path)
    print(file, ' : ', calc_map(data))
    if get_result_of_analyze:
        get_result(data, folder, file, get_conetnt)
    if view_and_0_filter:
        viewer_0filter(data, folder, file)
    if chk_question_or_not:
        communication_check(data, folder, file)
    
elif os.path.isdir(path):
    print(f"{path}는 폴더입니다.")
    
    # 폴더 내의 모든 파일 중에서 .csv 파일만 필터링
    csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".csv")]

    # 파일에 작업
    for csv_file in csv_files:
        data, folder, file = load(csv_file)
        print(file, ' : ', calc_map(data))
        if get_result_of_analyze:
            get_result(data, folder, file, get_conetnt)
        if view_and_0_filter:
            viewer_0filter(data, folder, file)
        if chk_question_or_not:
            communication_check(data, folder, file)
        
else:
    print(f"{path}는 존재하지 않습니다.")