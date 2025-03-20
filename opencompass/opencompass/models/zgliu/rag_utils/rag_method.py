from sentence_transformers import SentenceTransformer
import faiss
import jieba
from typing import List, Dict
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm, trange
import pandas as pd

nltk.download('punkt')
nltk.download('punkt_tab')

def mixed_segment(text):
    # 使用jieba进行中文分词
    words = jieba.cut(text)
    # 将分词结果转换为列表
    words = list(words)
    
    # 创建一个存储分词结果的列表
    segmented_words = []
    
    for word in words:
        # 如果是英文或数字，使用nltk的word_tokenize进行分词
        if word == ' ':
            pass
        elif word.isalpha():
            segmented_words.extend(word_tokenize(word))
        else:
            segmented_words.append(word)
    
    return segmented_words


def restore_sentence(segmented_words):
    # 还原为原句子，按中文和英文的特点处理
    restored_sentence = ""
    
    for word in segmented_words:
        # 中文与其他字符之间不加空格
        if any('\u4e00' <= char <= '\u9fff' for char in word):
            restored_sentence += word
        else:
            # 英文之间使用空格分隔
            restored_sentence += ' ' + word
    
    return restored_sentence.strip()


def load_embedding_model(model_path):
    embedding_model = SentenceTransformer(model_path,
                                        #   trust_remote_code=True,
                                        )
    return embedding_model


def bm25_search(embedding_model, tokenized_chunk_text, query_list, k:int=8):
    # print("Building BM25...")
    bm25 = BM25Okapi(tokenized_chunk_text)
    # 2. BM25 查询部分
    bm25_scores_combined = {}
    for query_part in query_list:
        bm25_scores = bm25.get_scores(query_part)
        # 将 BM25 的结果合并到字典中，若有重复索引则累加得分
        for idx, score in enumerate(bm25_scores):
            if idx in bm25_scores_combined:
                bm25_scores_combined[idx] += score
            else:
                bm25_scores_combined[idx] = score
                
    bm25_scores_combined = list(bm25_scores_combined.items())
    bm25_scores_combined.sort(key=lambda x: x[1], reverse=True)
    bm25_result = [x[0] for x in bm25_scores_combined[:k]]
    return bm25_result

def faiss_search(embedding_model, tokenized_chunk_text, query_list, k:int=8):
    # print("Building Faiss Index...")
    faiss_index = faiss.IndexFlatL2(768)
    faiss_batchsize = 128
    chunk_num = len(tokenized_chunk_text)
    num_batches = (chunk_num + faiss_batchsize - 1) // faiss_batchsize  # 向上取整
    # 遍历批次，进行编码并添加到索引中
    # for i in tqdm(range(num_batches)):
    for i in range(num_batches):
        start = i * faiss_batchsize
        end = min((i + 1) * faiss_batchsize, chunk_num)
        embeddings = embedding_model.encode([restore_sentence(x) for x in tokenized_chunk_text[start:end]])
        faiss_index.add(embeddings)

    query_embedding = embedding_model.encode(query_list)
    # print(f"{query_embedding=}, {query_embedding.shape=}")
    
    k = 8
    distance_threshold = -1

    # 1. FAISS 搜索部分
    faiss_distances, faiss_indices = faiss_index.search(query_embedding, k)

    # 展平距离和索引列表
    faiss_distances = faiss_distances.flatten().tolist()
    faiss_indices = faiss_indices.flatten().tolist()

    # 初始化选择的配对字典（用于存储符合条件的FAISS结果）
    faiss_selected = {}

    # 遍历FAISS搜索结果，筛选符合条件的结果并保留最小距离
    for dist, idx in zip(faiss_distances, faiss_indices):
        if distance_threshold > 0 and dist > distance_threshold:
            continue
        faiss_selected[idx] = min(dist, faiss_selected.get(idx, float('inf')))

    faiss_result = list(faiss_selected.items())
    faiss_result.sort(key=lambda x: x[1])
    faiss_result = [x[0] for x in faiss_result[:k]]
    return faiss_result

def bm25_faiss_search(embedding_model, tokenized_chunk_text, query_list, k:int=8):
    bm25_result = bm25_search(embedding_model, tokenized_chunk_text, query_list, k)
    faiss_result = faiss_search(embedding_model, tokenized_chunk_text, query_list, k)
    return list(set(bm25_result + faiss_result))

def main():
    input_text_list = []
    
    data_path = 'RAG/langchain/story.txt'
    data = open(data_path, 'r').read()
    input_text_list += [
        f"阅读下列故事，回答问题。{data}\n问题：{'是什么变成了马车？'}",
        f"阅读下列故事，回答问题。{data}\n问题：{'院子里的马车是什么做的？'}",
        f"阅读下列故事，回答问题。{data}\n问题：{'艾拉的交通工具是什么做的？'}",
    ]

    data_path = 'RAG/langchain/story_long.txt'
    data = open(data_path, 'r').read()
    input_text_list += [
        f"阅读下列故事，回答问题。{data}\n问题：{'是什么变成了马车？'}",
        f"阅读下列故事，回答问题。{data}\n问题：{'院子里的马车是什么做的？'}",
        f"阅读下列故事，回答问题。{data}\n问题：{'艾拉的交通工具是什么做的？'}",
    ]
    
    method = {
        "BM25": [bm25_search],
        "FAISS": [faiss_search],
        # "BM25+FAISS": [bm25_search, faiss_search],
    }
    answer_keyword = '南瓜'

    global_size = 32
    local_size = 32
    text_chunk_size = 64
    text_chunk_overlap = 16

    k = 16

    embedding_model_path = '~/models/jina-embeddings-v2-base-zh'
    embedding_model_path = os.path.expanduser(embedding_model_path)
    embedding_model = load_embedding_model(embedding_model_path)

    table_data = []
    
    for input_text_idx, input_text in enumerate(input_text_list):
        for method_name, method_func_list in method.items():
            tokenized_text = mixed_segment(input_text)
            tokenized_chunk_text = [tokenized_text[i:i + text_chunk_size] for i in range(0, len(tokenized_text), text_chunk_size - text_chunk_overlap)]

            global_part = tokenized_text[:global_size]
            local_part = tokenized_text[-local_size:]

            query_list = [global_part, local_part]

            result = []
            for method_func in method_func_list:
                result += method_func(embedding_model, tokenized_chunk_text, query_list, k)

            # 最终结果转换为列表
            final_combined_list = list(set(result))

            # print(f"{final_combined_list=}")

            selected_chunk = [restore_sentence(tokenized_chunk_text[idx]) for idx in final_combined_list]

            # print(f"{restore_sentence(global_part)=}")
            # print("#"*100)
            # print(f"{restore_sentence(local_part)=}")
            # print("#"*100)
            
            # for i, c in enumerate(selected_chunk):
            #     print(i, c)
            search_result = -1
            for i, chunk in enumerate(selected_chunk):
                if answer_keyword in chunk:
                    search_result = i
                    break
            
            # search_result = (search_result != -1)
            
            table_data.append({"input_text_idx": input_text_idx, "method_name": method_name, "search_result": search_result})
    
    df = pd.DataFrame(table_data)
    df_pivot = df.pivot(index='input_text_idx', columns='method_name', values='search_result')
    output_file_path = "./RAG_search_results.xlsx"
    df_pivot.to_excel(output_file_path, index=True)
            
    

# if __name__ == "__main__":
    
#     main()