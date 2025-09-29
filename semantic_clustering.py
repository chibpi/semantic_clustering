import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
import requests
import json
import time
warnings.filterwarnings('ignore')

class SemanticClusterGenerator:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the semantic cluster generator with a sentence transformer model
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.clusters = None
        self.cluster_model = None
        self.clustering_method = 'kmeans'  # Default clustering method
        
    def load_data(self, file_path):
        """
        Load data from CSV file and remove duplicate entries
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data with duplicates removed
        """
        try:
            # Use error_bad_lines=False to skip problematic lines
            df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
            print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
            
            # Remove duplicate entries
            initial_count = len(df)
            df = df.drop_duplicates()
            final_count = len(df)
            
            if initial_count != final_count:
                print(f"Removed {initial_count - final_count} duplicate entries")
                print(f"Final dataset has {final_count} unique rows")
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_text(self, text_series):
        """
        Preprocess text data (basic cleaning)
        
        Args:
            text_series (pandas.Series): Series containing text data
            
        Returns:
            pandas.Series: Preprocessed text
        """
        # Basic text cleaning
        processed_text = text_series.str.strip().str.lower()
        # Remove empty strings
        processed_text = processed_text[processed_text.str.len() > 0]
        return processed_text
    
    def generate_embeddings(self, texts):
        """
        Generate semantic embeddings for the texts
        
        Args:
            texts (list or pandas.Series): List of text strings
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        print("Generating semantic embeddings...")
        embeddings = self.model.encode(texts.tolist(), show_progress_bar=True)
        self.embeddings = embeddings
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def set_clustering_method(self, method='kmeans'):
        """
        Set the clustering method to use
        
        Args:
            method (str): Clustering method ('kmeans' or 'affinity')
        """
        if method.lower() in ['kmeans', 'affinity']:
            self.clustering_method = method.lower()
            print(f"Clustering method set to: {self.clustering_method}")
        else:
            print("Invalid clustering method. Use 'kmeans' or 'affinity'")
    
    def find_optimal_clusters(self, max_clusters=10):
        """
        Find optimal number of clusters using silhouette score (for K-means only)
        
        Args:
            max_clusters (int): Maximum number of clusters to try
            
        Returns:
            int: Optimal number of clusters
        """
        if self.embeddings is None:
            print("Please generate embeddings first")
            return None
        
        if self.clustering_method != 'kmeans':
            print("Optimal cluster finding only supported for K-means")
            return None
        
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        print("Finding optimal number of clusters for K-means...")
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            silhouette_avg = silhouette_score(self.embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")
        
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_clusters}")
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Different Numbers of Clusters (K-means)')
        plt.grid(True)
        plt.savefig('silhouette_scores.png')
        plt.close()
        
        return optimal_clusters
    
    def perform_affinity_propagation(self, damping=0.5, max_iter=200, min_cluster_size_percent=1.0):
        """
        Perform affinity propagation clustering on the embeddings
        
        Args:
            damping (float): Damping factor (between 0.5 and 1)
            max_iter (int): Maximum number of iterations
            min_cluster_size_percent (float): Minimum cluster size as percentage of total entries
            
        Returns:
            numpy.ndarray: Cluster labels
        """
        if self.embeddings is None:
            print("Please generate embeddings first")
            return None
        
        print("Performing affinity propagation clustering...")
        self.cluster_model = AffinityPropagation(damping=damping, max_iter=max_iter, random_state=42)
        self.clusters = self.cluster_model.fit_predict(self.embeddings)
        
        # Filter out small clusters (less than min_cluster_size_percent of total entries)
        if min_cluster_size_percent > 0:
            min_cluster_size = int(len(self.embeddings) * (min_cluster_size_percent / 100))
            unique_clusters, counts = np.unique(self.clusters, return_counts=True)
            
            # Create a mapping to reassign small clusters to -1 (noise/outliers)
            cluster_mapping = {}
            for cluster_id in unique_clusters:
                if counts[cluster_id] < min_cluster_size:
                    cluster_mapping[cluster_id] = -1
                else:
                    cluster_mapping[cluster_id] = cluster_id
            
            # Apply the mapping
            self.clusters = np.array([cluster_mapping[cluster_id] for cluster_id in self.clusters])
            
            # Remove noise points (-1) from silhouette score calculation
            valid_indices = self.clusters != -1
            valid_clusters = self.clusters[valid_indices]
            valid_embeddings = self.embeddings[valid_indices]
            
            print(f"Filtered out {np.sum(self.clusters == -1)} entries in small clusters")
        
        # Calculate silhouette score only on valid clusters
        valid_clusters = self.clusters[self.clusters != -1]
        if len(np.unique(valid_clusters)) > 1:  # Silhouette score requires at least 2 clusters
            valid_embeddings = self.embeddings[self.clusters != -1]
            silhouette_avg = silhouette_score(valid_embeddings, valid_clusters)
            print(f"Silhouette Score: {silhouette_avg:.4f}")
        else:
            print("Not enough valid clusters for silhouette score calculation")
        
        n_clusters = len(np.unique(valid_clusters))
        print(f"Number of valid clusters found: {n_clusters}")
        
        return self.clusters
    
    def perform_clustering(self, n_clusters=None, damping=0.5, max_iter=200, min_cluster_size_percent=1.0):
        """
        Perform clustering on the embeddings using the selected method
        
        Args:
            n_clusters (int): Number of clusters for K-means (optional)
            damping (float): Damping factor for affinity propagation
            max_iter (int): Maximum iterations for affinity propagation
            min_cluster_size_percent (float): Minimum cluster size as percentage of total entries (for affinity only)
            
        Returns:
            numpy.ndarray: Cluster labels
        """
        if self.embeddings is None:
            print("Please generate embeddings first")
            return None
        
        if self.clustering_method == 'kmeans':
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters()
            
            print(f"Performing K-means clustering with {n_clusters} clusters...")
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = self.cluster_model.fit_predict(self.embeddings)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(self.embeddings, self.clusters)
            print(f"Silhouette Score: {silhouette_avg:.4f}")
            
        elif self.clustering_method == 'affinity':
            print("Performing affinity propagation clustering...")
            self.cluster_model = AffinityPropagation(damping=damping, max_iter=max_iter, random_state=42)
            self.clusters = self.cluster_model.fit_predict(self.embeddings)
            
            # Filter out small clusters (less than min_cluster_size_percent of total entries)
            if min_cluster_size_percent > 0:
                min_cluster_size = int(len(self.embeddings) * (min_cluster_size_percent / 100))
                unique_clusters, counts = np.unique(self.clusters, return_counts=True)
                
                # Create a mapping to reassign small clusters to -1 (noise/outliers)
                cluster_mapping = {}
                for cluster_id in unique_clusters:
                    if counts[cluster_id] < min_cluster_size:
                        cluster_mapping[cluster_id] = -1
                    else:
                        cluster_mapping[cluster_id] = cluster_id
                
                # Apply the mapping
                self.clusters = np.array([cluster_mapping[cluster_id] for cluster_id in self.clusters])
                
                print(f"Filtered out {np.sum(self.clusters == -1)} entries in small clusters")
            
            # Calculate silhouette score only on valid clusters
            valid_clusters = self.clusters[self.clusters != -1]
            if len(np.unique(valid_clusters)) > 1:  # Silhouette score requires at least 2 clusters
                valid_embeddings = self.embeddings[self.clusters != -1]
                silhouette_avg = silhouette_score(valid_embeddings, valid_clusters)
                print(f"Silhouette Score: {silhouette_avg:.4f}")
            else:
                print("Not enough valid clusters for silhouette score calculation")
            
            n_clusters = len(np.unique(valid_clusters))
            print(f"Number of valid clusters found: {n_clusters}")
        
        else:
            print(f"Unknown clustering method: {self.clustering_method}")
            return None
        
        return self.clusters
    
    def visualize_clusters(self):
        """
        Visualize clusters using PCA for dimensionality reduction
        """
        if self.embeddings is None or self.clusters is None:
            print("Please generate embeddings and perform clustering first")
            return
        
        print("Visualizing clusters...")
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        reduced_embeddings = pca.fit_transform(self.embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                             c=self.clusters, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Semantic Clusters Visualization (PCA-reduced)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)
        plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Cluster visualization saved as 'cluster_visualization.png'")
    
    def get_cluster_summary(self, df, text_column='Texto'):
        """
        Get summary of each cluster with representative texts
        
        Args:
            df (pandas.DataFrame): Original dataframe
            text_column (str): Name of the text column
            
        Returns:
            dict: Cluster summaries
        """
        if self.clusters is None:
            print("Please perform clustering first")
            return None
        
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = self.clusters
        
        cluster_summary = {}
        unique_clusters = np.unique(self.clusters)
        
        for cluster_id in unique_clusters:
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # For K-means, use cluster centers; for affinity propagation, use exemplars
            if hasattr(self.cluster_model, 'cluster_centers_'):
                # K-means approach
                cluster_center = self.cluster_model.cluster_centers_[cluster_id]
                distances = np.linalg.norm(self.embeddings - cluster_center, axis=1)
                closest_indices = np.argsort(distances)[:5]  # Top 5 closest texts
            elif hasattr(self.cluster_model, 'cluster_centers_indices_'):
                # Affinity propagation approach - use exemplars
                exemplar_indices = self.cluster_model.cluster_centers_indices_
                cluster_indices = np.where(self.clusters == cluster_id)[0]
                if len(cluster_indices) > 0:
                    # Use the exemplar text and some random samples from the cluster
                    exemplar_idx = exemplar_indices[cluster_id]
                    exemplar_text = df_with_clusters.iloc[exemplar_idx][text_column]
                    # Get 4 other random samples from the cluster
                    other_indices = np.random.choice(cluster_indices, min(4, len(cluster_indices)), replace=False)
                    closest_indices = [exemplar_idx] + list(other_indices)
                else:
                    closest_indices = []
            else:
                # Fallback: use random samples
                cluster_indices = np.where(self.clusters == cluster_id)[0]
                closest_indices = np.random.choice(cluster_indices, min(5, len(cluster_indices)), replace=False)
            
            representative_texts = df_with_clusters.iloc[closest_indices][text_column].tolist()
            
            cluster_summary[cluster_id] = {
                'size': len(cluster_data),
                'representative_texts': representative_texts
            }
        
        return cluster_summary
    
    def generate_cluster_names_with_llm(self, cluster_summary, model="qwen3:30b-a3b", ollama_url="http://10.1.141.16:11434"):
        """
        Generate meaningful names for clusters using LLM
        
        Args:
            cluster_summary (dict): Cluster summary from get_cluster_summary
            model (str): Ollama model to use
            ollama_url (str): Ollama server URL
            
        Returns:
            dict: Cluster names mapping
        """
        print(f"\nGenerating meaningful cluster names using {model}...")
        
        cluster_names = {}
        
        for cluster_id, summary in cluster_summary.items():
            if cluster_id == -1:  # Skip noise cluster
                cluster_names[cluster_id] = "Noise/Outliers"
                continue
                
            representative_texts = summary['representative_texts']
            if not representative_texts:
                cluster_names[cluster_id] = f"Cluster_{cluster_id}"
                continue
                
            # Prepare prompt for LLM
            prompt = f"""Analyze the following group of Spanish text documents from an administrative context and provide a concise, meaningful name in Spanish that describes the common theme or category. The name should be 2-5 words maximum.

Text examples:
{chr(10).join([f'{i+1}. {text[:200]}...' for i, text in enumerate(representative_texts[:3])])}

Provide only the category name in Spanish, nothing else."""
            
            try:
                # Call Ollama API
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    cluster_name = result['response'].strip()
                    # Clean up the response to get just the category name
                    cluster_name = cluster_name.split('\n')[0].strip()
                    cluster_name = cluster_name.replace('"', '').replace("'", "")
                    cluster_names[cluster_id] = cluster_name
                    print(f"Cluster {cluster_id}: {cluster_name}")
                else:
                    print(f"Error generating name for cluster {cluster_id}: {response.status_code}")
                    cluster_names[cluster_id] = f"Cluster_{cluster_id}"
                    
            except Exception as e:
                print(f"Error calling LLM for cluster {cluster_id}: {e}")
                cluster_names[cluster_id] = f"Cluster_{cluster_id}"
            
            # Add small delay to avoid overwhelming the server
            time.sleep(1)
        
        return cluster_names
    
    def save_results(self, df, output_file='clustered_results.csv', cluster_names=None):
        """
        Save clustering results to CSV file with optional cluster names
        
        Args:
            df (pandas.DataFrame): Original dataframe
            output_file (str): Output file path
            cluster_names (dict): Optional mapping of cluster IDs to names
        """
        if self.clusters is None:
            print("Please perform clustering first")
            return
        
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = self.clusters
        
        # Add cluster names if provided
        if cluster_names:
            df_with_clusters['cluster_name'] = df_with_clusters['cluster'].map(cluster_names)
        
        df_with_clusters.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Results saved to {output_file}")
    
    def run_pipeline(self, file_path, text_column='Texto', n_clusters=None):
        """
        Run complete clustering pipeline
        
        Args:
            file_path (str): Path to CSV file
            text_column (str): Name of text column
            n_clusters (int): Number of clusters (optional)
        """
        print("Starting semantic clustering pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return
        
        # Preprocess text
        processed_texts = self.preprocess_text(df[text_column])
        
        # Generate embeddings
        self.generate_embeddings(processed_texts)
        
        # Perform clustering
        self.perform_clustering(n_clusters)
        
        # Visualize clusters
        self.visualize_clusters()
        
        # Get cluster summary
        cluster_summary = self.get_cluster_summary(df, text_column)
        
        # Print cluster summary
        print("\n" + "="*50)
        print("CLUSTER SUMMARY")
        print("="*50)
        for cluster_id, summary in cluster_summary.items():
            print(f"\nCluster {cluster_id} (Size: {summary['size']}):")
            print("Representative texts:")
            for i, text in enumerate(summary['representative_texts'], 1):
                print(f"  {i}. {text}")
        
        # Save results
        self.save_results(df)
        
        print("\nPipeline completed successfully!")

# Example usage
if __name__ == "__main__":
    print("Semantic Clustering Demo")
    print("=" * 50)
    
    # Initialize the cluster generator
    cluster_generator = SemanticClusterGenerator()
    
    # Load data
    df = cluster_generator.load_data('dataframe.csv')
    if df is None:
        exit()
    
    # Preprocess text
    processed_texts = cluster_generator.preprocess_text(df['Texto'])
    
    # Generate embeddings
    cluster_generator.generate_embeddings(processed_texts)
    
    # Option 1: K-means clustering
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING")
    print("="*50)
    cluster_generator.set_clustering_method('kmeans')
    cluster_generator.perform_clustering()
    cluster_generator.visualize_clusters()
    
    # Get and print cluster summary for K-means
    kmeans_summary = cluster_generator.get_cluster_summary(df, 'Texto')
    print("\nK-means Cluster Summary:")
    for cluster_id, summary in kmeans_summary.items():
        print(f"\nCluster {cluster_id} (Size: {summary['size']}):")
        for i, text in enumerate(summary['representative_texts'], 1):
            print(f"  {i}. {text}")
    
    # Save K-means results
    cluster_generator.save_results(df, 'kmeans_clustered_results.csv')
    
    # Option 2: Affinity Propagation clustering with minimum cluster size filter
    print("\n" + "="*50)
    print("AFFINITY PROPAGATION CLUSTERING (WITH MINIMUM CLUSTER SIZE FILTER)")
    print("="*50)
    cluster_generator.set_clustering_method('affinity')
    cluster_generator.perform_clustering(damping=0.7, min_cluster_size_percent=1.0)
    cluster_generator.visualize_clusters()
    
    # Get and print cluster summary for Affinity Propagation
    affinity_summary = cluster_generator.get_cluster_summary(df, 'Texto')
    print("\nAffinity Propagation Cluster Summary:")
    for cluster_id, summary in affinity_summary.items():
        print(f"\nCluster {cluster_id} (Size: {summary['size']}):")
        for i, text in enumerate(summary['representative_texts'], 1):
            print(f"  {i}. {text}")
    
    # Generate meaningful cluster names using LLM with gpt-oss:latest model
    cluster_names = cluster_generator.generate_cluster_names_with_llm(affinity_summary, model="gpt-oss:latest")
    
    # Save Affinity Propagation results with cluster names
    cluster_generator.save_results(df, 'affinity_clustered_results.csv', cluster_names)
    
    # Print final cluster names
    print("\n" + "="*50)
    print("FINAL CLUSTER NAMES")
    print("="*50)
    for cluster_id, name in cluster_names.items():
        print(f"Cluster {cluster_id}: {name}")
    
    print("\nBoth clustering methods completed successfully!")
    print("Results saved to:")
    print("- kmeans_clustered_results.csv")
    print("- affinity_clustered_results.csv (with LLM-generated cluster names)")
    print("- cluster_visualization.png (for the last method run)")
