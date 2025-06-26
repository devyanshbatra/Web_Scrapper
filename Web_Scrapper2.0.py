import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from typing import List, Dict, Tuple
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MLKeywordSearch:
    def __init__(self, language='english'):
        """
        Initialize ML-based keyword search system
        
        Args:
            language: Language for stopwords and processing
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = None
        self.documents = []
        self.document_vectors = None
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for ML analysis
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and processed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        processed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def fit_documents(self, documents: List[Dict]):
        """
        Fit the ML model on a collection of documents
        
        Args:
            documents: List of document dictionaries with 'content' key
        """
        self.documents = documents
        
        # Preprocess all documents
        processed_docs = []
        for doc in documents:
            content = doc.get('content', '') + ' ' + doc.get('title', '')
            processed_content = self.preprocess_text(content)
            processed_docs.append(processed_content)
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        self.document_vectors = self.tfidf_matrix.toarray()
        
        print(f"Fitted {len(documents)} documents")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def search_by_keywords(self, keywords: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search documents using keyword similarity
        
        Args:
            keywords: Search keywords
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit_documents first.")
        
        # Preprocess query
        processed_query = self.preprocess_text(keywords)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                results.append((self.documents[idx], similarities[idx]))
        
        return results
    
    def extract_key_terms(self, n_terms: int = 20) -> List[Tuple[str, float]]:
        """
        Extract key terms from the document collection
        
        Args:
            n_terms: Number of top terms to extract
            
        Returns:
            List of (term, importance_score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit_documents first.")
        
        # Calculate mean TF-IDF scores for each term
        mean_scores = np.mean(self.tfidf_matrix.toarray(), axis=0)
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Create term-score pairs
        term_scores = list(zip(feature_names, mean_scores))
        
        # Sort by score and return top terms
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        return term_scores[:n_terms]
    
    def cluster_documents(self, n_clusters: int = 5) -> Dict:
        """
        Cluster documents based on content similarity
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with cluster information
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit_documents first.")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.tfidf_matrix)
        
        # Organize results
        clusters = {}
        for i in range(n_clusters):
            cluster_docs = [
                self.documents[j] for j in range(len(self.documents))
                if cluster_labels[j] == i
            ]
            clusters[f'cluster_{i}'] = {
                'documents': cluster_docs,
                'size': len(cluster_docs)
            }
        
        return clusters
    
    def find_similar_documents(self, doc_index: int, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find documents similar to a given document
        
        Args:
            doc_index: Index of the reference document
            top_k: Number of similar documents to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit_documents first.")
        
        # Get document vector
        doc_vector = self.tfidf_matrix[doc_index:doc_index+1]
        
        # Calculate similarities
        similarities = cosine_similarity(doc_vector, self.tfidf_matrix).flatten()
        
        # Exclude the document itself
        similarities[doc_index] = 0
        
        # Get top similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.documents[idx], similarities[idx]))
        
        return results
    
    def topic_modeling(self, n_topics: int = 5, n_words: int = 10) -> Dict:
        """
        Perform topic modeling using Latent Dirichlet Allocation
        
        Args:
            n_topics: Number of topics to extract
            n_words: Number of words per topic
            
        Returns:
            Dictionary with topic information
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit_documents first.")
        
        # Perform LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(self.tfidf_matrix)
        
        # Extract topics
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f'topic_{topic_idx}'] = {
                'words': top_words,
                'weights': [topic[i] for i in top_words_idx]
            }
        
        return topics

# Example usage
if __name__ == "__main__":
    # Sample documents for testing
    sample_docs = [
        {
            'title': 'Machine Learning Basics',
            'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.',
            'url': 'example1.com'
        },
        {
            'title': 'Web Scraping Guide',
            'content': 'Web scraping is the process of extracting data from websites using automated tools and scripts.',
            'url': 'example2.com'
        },
        {
            'title': 'Python Programming',
            'content': 'Python is a versatile programming language used for web development, data science, and automation.',
            'url': 'example3.com'
        }
    ]
    
    # Initialize and fit the model
    ml_search = MLKeywordSearch()
    ml_search.fit_documents(sample_docs)
    
    # Search for documents
    results = ml_search.search_by_keywords("machine learning algorithms", top_k=3)
    
    print("Search Results:")
    for doc, score in results:
        print(f"Score: {score:.3f} - {doc['title']}")
    
    # Extract key terms
    key_terms = ml_search.extract_key_terms(10)
    print(f"\nKey Terms: {key_terms[:5]}")
