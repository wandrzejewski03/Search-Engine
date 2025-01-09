import os
import pandas as pd
import pyterrier as pt
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
import string
from clean_text import clean_text
import numpy as np


if not pt.started():
    pt.init()

# settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.options.display.max_colwidth = 1000

# paths
corpus_path = r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\subset_answers.json'
val_queries_path = r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\val\subset_data.jsonl'
test_queries_path = r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\test\subset_data.jsonl'
test_qrels_path = r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\test\qrels.json'
val_qrels_path = r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\val\qrels.json'

# loading data
corpus_df = pd.read_json(corpus_path, orient='index').reset_index()
corpus_df.columns = ['docno', 'text']

val_queries = pd.read_json(val_queries_path, lines=True)[['id', 'title', 'user_id']]
val_queries.columns = ['qid', 'query', 'user_id']
val_queries['query'] = val_queries['query'].apply(clean_text)

test_queries = pd.read_json(test_queries_path, lines=True)[['id', 'title', 'user_id']]
test_queries.columns = ['qid', 'query', 'user_id']
test_queries['query'] = test_queries['query'].apply(clean_text)

test_qrels = pd.read_json(test_qrels_path, orient='index').reset_index()
test_qrels.columns = ['qid', 'docno']
test_qrels['label'] = 1

val_qrels = pd.read_json(val_qrels_path, orient='index').reset_index()
val_qrels.columns = ['qid', 'docno']
val_qrels['label'] = 1

val_queries = val_queries[val_queries['qid'].isin(val_qrels['qid'])]



def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

val_queries['query'] = val_queries['query'].apply(preprocess_text)


biencoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


embedding_dir = './embeddings'
query_embeddings_path = os.path.join(embedding_dir, 'query_embeddings.npy')
doc_embeddings_path = os.path.join(embedding_dir, 'doc_embeddings.npy')


if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)

if not os.path.exists(query_embeddings_path) or not os.path.exists(doc_embeddings_path):
    print("Embeddings not found. Computing embeddings...")
    # Precompute Embeddings (for queries and corpus docs)
    query_embs = biencoder_model.encode(val_queries['query'].tolist(), batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    doc_embs = biencoder_model.encode(corpus_df['text'].tolist(), batch_size=64, convert_to_tensor=True, show_progress_bar=True)



    np.save(query_embeddings_path, query_embs.cpu().numpy())
    np.save(doc_embeddings_path, doc_embs.cpu().numpy())
    print("Embeddings saved.")
else:
    print("Loading precomputed embeddings...")

    query_embs = np.load(query_embeddings_path)
    doc_embs = np.load(doc_embeddings_path)





def _biencoder_apply(df):
  query_embs = biencoder_model.encode(df['query'].values)
  doc_embs = biencoder_model.encode(df['text'].values)
  scores = cos_sim(query_embs, doc_embs)
  return scores[0]


bi_encT = pt.apply.doc_score(_biencoder_apply, batch_size=64)


pt_index_path = r'C:\Users\wikto\PycharmProjects\pythonProject\SearchEngine\corp_index'

if not os.path.exists(pt_index_path + '/data.properties'):
    indexer = pt.index.IterDictIndexer(pt_index_path, overwrite=True, stemmer=None, stopwords=True)
    index_ref = indexer.index(corpus_df.to_dict(orient="records"),
                              fields=({'text': 2096}),
                              meta=({'docno': 20, 'text': 20096}))
else:
    index_ref = pt.IndexRef.of(pt_index_path + '/data.properties')

index2 = pt.IndexFactory.of(index_ref)


MODEL = pt.BatchRetrieve(index2, wmodel="BM25") % 50


bi_pipeline = MODEL >> pt.text.get_text(index2, 'text') >> bi_encT
normalized_br = MODEL >> pt.pipelines.PerQueryMaxMinScoreTransformer()
normalized_bi_pipeline = bi_pipeline >> pt.pipelines.PerQueryMaxMinScoreTransformer()


bi_sum_25_pipeline = .25 * normalized_bi_pipeline + (1 - .25) * normalized_br
bi_sum_50_pipeline = .5 * normalized_bi_pipeline + (1 - .5) * normalized_br
bi_sum_75_pipeline = .75 * normalized_bi_pipeline + (1 - .75) * normalized_br


result = pt.Experiment(
    [
        MODEL,
        bi_pipeline,
        bi_sum_25_pipeline,
        bi_sum_50_pipeline,
        bi_sum_75_pipeline
    ],
    val_queries,
    val_qrels,
    names=[
        'BM25',
        'BiEnc',
        '.25*BiEnc + .75*BM25',
        '.5*BiEnc + .5*BM25',
        '.75*BiEnc + .25*BM25'
    ],
    eval_metrics=["map", "ndcg", "P.10"]
)


result.to_csv('bi_encoder_results.txt', sep='\t', index=False)


plt.figure(figsize=(10, 6))
for metric in ['map', 'ndcg', 'P.10']:
    plt.plot(['BM25', 'BiEnc', '.25*BiEnc + .75*BM25', '.5*BiEnc + .5*BM25', '.75*BiEnc + .25*BM25'],
             result[metric], label=metric)
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Evaluation Metrics for Different Model Combinations')
plt.legend()
plt.savefig('bi_encoder_results_reduced.png')
plt.show()
