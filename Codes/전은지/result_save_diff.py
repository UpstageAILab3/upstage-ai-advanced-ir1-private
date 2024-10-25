import pandas as pd

# result 파일 경로
file_a_path = '/data/ephemeral/home/result_csv/2024-10-12-05h-57m_0h-4m-3s_ko-sbert-sts_gpt-3.5-turbo-1106_transformed_with_topk_scores.csv'  # 파일 A의 경로
file_b_path = '/data/ephemeral/home/result_csv/2024-10-12-06h-13m_0h-4m-39s_klue-roberta-base-nli-sts_gpt-3.5-turbo-1106_transformed_with_topk_scores.csv'  # 파일 B의 경로

# 파일 읽기
df_a = pd.read_csv(file_a_path)
df_b = pd.read_csv(file_b_path)

# 'id' 컬럼을 기준으로 두 데이터프레임을 병합 (inner join)
df_merged = pd.merge(df_a, df_b, on='eval_id', suffixes=('_a', '_b'))

# 두 파일의 같은 id에 대해 값이 다른 항목 필터링
df_diff = df_merged[
    (df_merged['topk_1_a'] != df_merged['topk_1_b']) |
    (df_merged['topk_2_a'] != df_merged['topk_2_b']) |
    (df_merged['topk_3_a'] != df_merged['topk_3_b'])
]

# 다른 항목들을 새로운 CSV로 저장
output_file_path = '/data/ephemeral/home/result_csv/2122.csv'
df_diff.to_csv(output_file_path, index=False)

print(f"다른 항목들이 {output_file_path}에 저장되었습니다.")
