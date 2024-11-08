from data.preprocess_data import DataPreprocessor
from model.kmeans_clustering import KMeansClustering, cosine_similarity
from model.eval import Evaluator
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
import numpy as np
import mlflow
import contextlib

@contextlib.contextmanager
def disable_mlflow_logging():
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri("")
    try:
        yield
    finally:
        mlflow.set_tracking_uri(original_uri)

def run_clustering(start_idx: int, end_idx: int, debug_mode: bool = False):
    preprocessor = DataPreprocessor()
    preprocessed_df = preprocessor.transform()

    # Loop
    puzzle = preprocessed_df.startingGroups.to_list()
    results = []
    for words in puzzle[start_idx:end_idx]:
        
        clusterer = KMeansClustering(n_clusters=4, max_iter=10, init='kmeans++', embedding_dim=300)
        try:
            word_embeddings = clusterer.encode(words=words)
            X = word_embeddings
            # clusterer = clusterer.fit(X)
            
            # kmeans = KMeans(n_clusters=4, init='random', random_state=42, max_iter=100, tol=1e-16).fit(X)
            # print(words)
            clf = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=42, tol=0.1)
            clf_results = clf.fit_predict(X)

            # print('home cooked:', clusterer.labels)
            # print('kmeans:', kmeans.labels_)
            # print('constrained:', clf.labels_)
            
            true_labels = preprocessed_df.group_columns_array[2]
            # evaluator = Evaluator(words, kmeans.labels_)
            # evaluator = Evaluator(words, clusterer.labels)
            evaluator = Evaluator(words, clf.labels_)
            correct = evaluator.evaluate(true_labels)
            # print(f"Correct: {correct}")
            results.append(correct)

        except Exception as e:
            results.append(e)
            continue

        print(len(results))

    accuracy = sum(item for item in results if isinstance(item, bool))/len(results)

    if not debug_mode:
        mlflow.log_text("model", "constrained kmeans")  # "scikit kmeans" | "Custom Implementation"
        mlflow.autolog()
        mlflow.log_param("embedding dims", clusterer.embedding_dim)
        mlflow.log_param("start_idx", start_idx)
        mlflow.log_param("end_idx", end_idx)
        mlflow.log_text("results",str(results))
        mlflow.log_metric("correct", correct)
        mlflow.log_metric("accuracy", accuracy)
    
    print("results: ",results)
    print("accuracy: ",accuracy)
    
    return results

def main(start_idx: int = 0, 
         end_idx: int = -1, 
         debug: bool = False):
    mlflow.set_tracking_uri("file:///home/zorin/Documents/nyt_connections_solver/experiments/mlruns")
    mlflow.set_experiment("NYT Connections Clustering")
    
    if debug:
        # Run without MLflow logging
        with disable_mlflow_logging():
            results = run_clustering(start_idx=7, end_idx=100, debug_mode=True)
    else:
        # Run with MLflow logging
        with mlflow.start_run():
            results = run_clustering(start_idx=0, end_idx=50, debug_mode=False)

if __name__ == '__main__':
    # Set to True when debugging, False for normal runs with logging
    DEBUG_MODE = False
    main(debug=DEBUG_MODE)