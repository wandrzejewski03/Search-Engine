import pandas as pd
import os
import pyterrier as pt
from clean_text import clean_text
import matplotlib.pyplot as plt
import seaborn as sns


if not pt.started():
    pt.init()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.options.display.max_colwidth = 1000



# CORPUS
corpus_df = pd.read_json(
    r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\subset_answers.json',
    orient='index')
corpus_df = corpus_df.reset_index()
corpus_df.columns = ['docno', 'text']



# VAL QUERIES
val_queries = pd.read_json(
    r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\val\subset_data.jsonl',
    lines=True)
val_queries = val_queries[['id', 'title', 'user_id']]
val_queries.columns = ['qid', 'query', 'user_id']
val_queries['query'] = val_queries['query'].apply(clean_text)


# TEST QUERIES
test_queries = pd.read_json(
    r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\test\subset_data.jsonl',
    lines=True)
test_queries = test_queries[['id', 'title', 'user_id']]
test_queries.columns = ['qid', 'query', 'user_id']
test_queries['query'] = test_queries['query'].apply(clean_text)


# TRAIN QUERIES
train_queries = pd.read_json(
    r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\train\subset_data.jsonl',
    lines=True)
train_queries = train_queries[['id', 'title', 'user_id']]
train_queries.columns = ['qid', 'query', 'user_id']
train_queries['query'] = train_queries['query'].apply(clean_text)


# TEST QRELS
test_qrels = pd.read_json(
    r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\test\qrels.json',
    orient='index')
test_qrels = test_qrels.reset_index()
test_qrels.columns = ['qid', 'docno']
test_qrels['label'] = 1


# VAL QRELS
val_qrels = pd.read_json(
    r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\val\qrels.json',
    orient='index')
val_qrels = val_qrels.reset_index()
val_qrels.columns = ['qid', 'docno']
val_qrels['label'] = 1

# TRAIN QRELS
train_qrels = pd.read_json(
    r'C:\Users\wikto\Desktop\Wiktor\ERASMUS\Studies\Information Retrieval & Recommender Systems\data_project\PIR_data\answer_retrieval\train\qrels.json',
    orient='index')
train_qrels = train_qrels.reset_index()
train_qrels.columns = ['qid', 'docno']
train_qrels['label'] = 1



# INDEXING
pt_index_path = r'C:\Users\wikto\PycharmProjects\pythonProject\SearchEngine\corp_index'
if not os.path.exists(pt_index_path + "/data.properties"):
    indexer = pt.index.IterDictIndexer(pt_index_path, overwrite=True, stemmer=None, stopwords=True)
    index_ref = indexer.index(corpus_df.to_dict(orient="records"),
                              fields=({'text': 2096}),
                              meta=({'docno': 20, 'text': 20096})
                              )

else:
    index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")
index2 = pt.IndexFactory.of(index_ref)


MODEL = pt.BatchRetrieve(index2, wmodel="BM25")

# ONLY BM25
result_val = pt.Experiment(
    [
        MODEL,
    ],
    val_queries,
    val_qrels,
    names=[
        'BM25',
      ],
    eval_metrics=["map", 'ndcg', 'P.10']
)


result_test = pt.Experiment(
    [
        MODEL,
    ],
    test_queries,
    test_qrels,
    names=[
        'BM25',
      ],
    eval_metrics=["map", 'ndcg', 'P.10']
)





TF_IDF_MODEL = pt.BatchRetrieve(index2, wmodel="TF_IDF")

# ONLY TF_IDF

result_val_TF = pt.Experiment(
    [
        TF_IDF_MODEL,
    ],
    val_queries,
    val_qrels,
    names=[
        'TF-IDF',
      ],
    eval_metrics=["map", 'ndcg', 'P.10']
)


result_test_TF = pt.Experiment(
    [
        TF_IDF_MODEL,
    ],
    test_queries,
    test_qrels,
    names=[
        'TF-IDF',
      ],
    eval_metrics=["map", 'ndcg', 'P.10']
)


print(f"Results for data set val (BM-25): {result_val}")
print(f"Results for data set test (BM-25): {result_test}")

print(f"Results for data set val (TF-IDF): {result_val_TF}")
print(f"Results for data set test (BM-25): {result_test_TF}")



results = {
    'Model': ['BM25', 'BM25', 'TF-IDF', 'TF-IDF'],
    'Dataset': ['Validation', 'Test', 'Validation', 'Test'],
    'MAP': [result_val['map'].values[0], result_test['map'].values[0],
            result_val_TF['map'].values[0], result_test_TF['map'].values[0]],
    'NDCG': [result_val['ndcg'].values[0], result_test['ndcg'].values[0],
             result_val_TF['ndcg'].values[0], result_test_TF['ndcg'].values[0]],
    'P.10': [result_val['P.10'].values[0], result_test['P.10'].values[0],
             result_val_TF['P.10'].values[0], result_test_TF['P.10'].values[0]]
}



df_results = pd.DataFrame(results)
df_results.to_csv('BM-25_vs_TF-IDF_results.csv', sep='\t', index=False)


sns.set(style="whitegrid")


fig, axes = plt.subplots(1, 3, figsize=(18, 6))


sns.barplot(x='Dataset', y='MAP', hue='Model', data=df_results, ax=axes[0])
axes[0].set_title("MAP Comparison (Validation vs Test)")
axes[0].set_ylabel("Mean Average Precision (MAP)")


sns.barplot(x='Dataset', y='NDCG', hue='Model', data=df_results, ax=axes[1])
axes[1].set_title("NDCG Comparison (Validation vs Test)")
axes[1].set_ylabel("Normalized Discounted Cumulative Gain (NDCG)")


sns.barplot(x='Dataset', y='P.10', hue='Model', data=df_results, ax=axes[2])
axes[2].set_title("P@10 Comparison (Validation vs Test)")
axes[2].set_ylabel("Precision at 10 (P@10)")


plt.tight_layout()
plt.savefig("PLOT BM-25 TF-IDF")
plt.show()



