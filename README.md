# Semantic Clustering for Administrative Documents

A Python-based semantic clustering system that automatically categorizes Spanish administrative documents using natural language processing and machine learning techniques.

## Description

This project implements a semantic clustering pipeline that processes Spanish administrative documents from social services departments. It uses sentence transformers to generate semantic embeddings and applies multiple clustering algorithms (K-means and Affinity Propagation) to automatically categorize documents into meaningful groups. The system can also generate descriptive cluster names using LLM integration.

The project is particularly useful for:
- Automating document categorization in administrative workflows
- Identifying patterns and themes in large document collections
- Improving information retrieval and organization in public administration
- Analyzing grant applications, subsidy requests, and social service documents

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Dependencies

Install the required packages:

```bash
pip install pandas numpy scikit-learn sentence-transformers matplotlib requests
```

### Optional Dependencies for LLM Integration

If you want to use the LLM-based cluster naming feature, you'll need access to an Ollama server:

```bash
# Install Ollama (follow instructions from https://ollama.ai/)
# Then pull the desired model:
ollama pull qwen3:30b-a3b
# or
ollama pull gpt-oss:latest
```

## Usage

### Basic Usage

1. **Prepare your data**: Ensure your CSV file has a text column (default: 'Texto') containing the documents to cluster.

2. **Run the clustering pipeline**:

```python
from semantic_clustering import SemanticClusterGenerator

# Initialize the cluster generator
cluster_generator = SemanticClusterGenerator()

# Run complete pipeline
cluster_generator.run_pipeline('path/to/your/data.csv', text_column='Texto')
```

### Advanced Usage

```python
from semantic_clustering import SemanticClusterGenerator

# Initialize with custom model
cluster_generator = SemanticClusterGenerator(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Load and preprocess data
df = cluster_generator.load_data('dataframe.csv')
processed_texts = cluster_generator.preprocess_text(df['Texto'])

# Generate embeddings
cluster_generator.generate_embeddings(processed_texts)

# Option 1: K-means clustering
cluster_generator.set_clustering_method('kmeans')
cluster_generator.perform_clustering(n_clusters=5)  # Optional: specify number of clusters
cluster_generator.visualize_clusters()

# Option 2: Affinity Propagation clustering
cluster_generator.set_clustering_method('affinity')
cluster_generator.perform_clustering(damping=0.7, min_cluster_size_percent=1.0)
cluster_generator.visualize_clusters()

# Get cluster summary
cluster_summary = cluster_generator.get_cluster_summary(df, 'Texto')

# Generate meaningful cluster names using LLM
cluster_names = cluster_generator.generate_cluster_names_with_llm(
    cluster_summary, 
    model="gpt-oss:latest",
    ollama_url="http://localhost:11434"  # Your Ollama server URL
)

# Save results
cluster_generator.save_results(df, 'clustered_results.csv', cluster_names)
```

### Command Line Usage

The script can also be run directly:

```bash
python semantic_clustering.py
```

This will execute the demo pipeline using the default settings.

## Features

### Core Features

- **Multiple Clustering Algorithms**: Supports both K-means and Affinity Propagation clustering
- **Semantic Embeddings**: Uses state-of-the-art multilingual sentence transformers
- **Optimal Cluster Detection**: Automatically finds optimal number of clusters for K-means using silhouette scores
- **Cluster Visualization**: Generates 2D PCA visualizations of clusters
- **Data Preprocessing**: Automatic text cleaning and duplicate removal
- **Comprehensive Output**: Saves clustering results with cluster assignments and summaries

### Advanced Features

- **LLM Integration**: Generates meaningful Spanish cluster names using local LLMs
- **Flexible Configuration**: Customizable clustering parameters and model selection
- **Robust Error Handling**: Graceful handling of data loading and processing errors
- **Performance Metrics**: Calculates silhouette scores for clustering quality assessment
- **Minimum Cluster Size Filter**: Filters out small clusters in Affinity Propagation

### Output Files

The system generates several output files:

- `clustered_results.csv`: Main results with cluster assignments
- `kmeans_clustered_results.csv`: Results from K-means clustering
- `affinity_clustered_results.csv`: Results from Affinity Propagation with LLM-generated names
- `cluster_visualization.png`: 2D visualization of clusters
- `silhouette_scores.png`: Silhouette score analysis for K-means

## Configuration

### Model Configuration

```python
# Available sentence transformer models
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'  # Default
# model_name = 'paraphrase-multilingual-mpnet-base-v2'
# model_name = 'distiluse-base-multilingual-cased-v2'
```

### Clustering Parameters

**K-means Parameters:**
- `n_clusters`: Number of clusters (optional, auto-detected if not specified)
- `random_state`: Random seed for reproducibility

**Affinity Propagation Parameters:**
- `damping`: Damping factor (0.5 to 1.0, default: 0.7)
- `max_iter`: Maximum iterations (default: 200)
- `min_cluster_size_percent`: Minimum cluster size as percentage of total (default: 1.0)

### LLM Configuration

```python
# Ollama configuration
ollama_url = "http://localhost:11434"  # Default Ollama server
model = "gpt-oss:latest"  # or "qwen3:30b-a3b"
```

### Data Configuration

- **Text Column**: Default is 'Texto', can be customized
- **Encoding**: UTF-8
- **Duplicate Handling**: Automatic duplicate removal

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project is designed for processing Spanish administrative documents but can be adapted for other languages and document types by changing the sentence transformer model and text preprocessing steps.
