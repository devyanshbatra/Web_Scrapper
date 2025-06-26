import json
import os
from web_scraper import WebScraper
from ml_keyword_search import MLKeywordSearch
from typing import List, Dict
import argparse

class WebScrapingMLApp:
    def __init__(self):
        """Initialize the web scraping ML application"""
        self.scraper = WebScraper(delay=1.0)
        self.ml_search = MLKeywordSearch()
        self.scraped_data = []
        
    def scrape_and_analyze(self, urls: List[str]) -> Dict:
        """
        Scrape URLs and perform ML analysis
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Dictionary containing all results
        """
        print("=== Starting Web Scraping ===")
        
        # Scrape websites
        self.scraped_data = self.scraper.scrape_multiple_pages(urls)
        
        if not self.scraped_data:
            print("No data scraped successfully!")
            return {}
        
        print(f"Successfully scraped {len(self.scraped_data)} pages")
        
        # Fit ML model
        print("\n=== Training ML Model ===")
        self.ml_search.fit_documents(self.scraped_data)
        
        # Perform analysis
        results = {
            'scraped_data': self.scraped_data,
            'analysis': self._perform_analysis()
        }
        
        return results
    
    def _perform_analysis(self) -> Dict:
        """Perform comprehensive ML analysis"""
        analysis = {}
        
        # Extract key terms
        print("Extracting key terms...")
        key_terms = self.ml_search.extract_key_terms(15)
        analysis['key_terms'] = key_terms
        
        # Cluster documents if we have enough
        if len(self.scraped_data) >= 3:
            print("Clustering documents...")
            clusters = self.ml_search.cluster_documents(
                n_clusters=min(3, len(self.scraped_data))
            )
            analysis['clusters'] = clusters
        
        # Topic modeling if we have enough documents
        if len(self.scraped_data) >= 3:
            print("Performing topic modeling...")
            topics = self.ml_search.topic_modeling(
                n_topics=min(3, len(self.scraped_data)),
                n_words=8
            )
            analysis['topics'] = topics
        
        return analysis
    
    def search_content(self, keywords: str, top_k: int = 5) -> List[Dict]:
        """
        Search scraped content using ML-based keyword matching
        
        Args:
            keywords: Search keywords
            top_k: Number of top results
            
        Returns:
            List of search results
        """
        if not self.scraped_data:
            print("No scraped data available. Please scrape some URLs first.")
            return []
        
        print(f"Searching for: '{keywords}'")
        results = self.ml_search.search_by_keywords(keywords, top_k)
        
        search_results = []
        for doc, score in results:
            search_results.append({
                'title': doc['title'],
                'url': doc['url'],
                'similarity_score': float(score),
                'content_preview': doc['content'][:300] + '...' if len(doc['content']) > 300 else doc['content']
            })
        
        return search_results
    
    def find_similar_pages(self, page_index: int, top_k: int = 3) -> List[Dict]:
        """Find pages similar to a given page"""
        if not self.scraped_data or page_index >= len(self.scraped_data):
            return []
        
        results = self.ml_search.find_similar_documents(page_index, top_k)
        
        similar_pages = []
        for doc, score in results:
            similar_pages.append({
                'title': doc['title'],
                'url': doc['url'],
                'similarity_score': float(score),
                'content_preview': doc['content'][:200] + '...'
            })
        
        return similar_pages
    
    def save_results(self, results: Dict, filename: str = 'scraping_results.json'):
        """Save all results to file"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_converted = convert_types(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
    
    def interactive_search(self):
        """Interactive search interface"""
        if not self.scraped_data:
            print("No data available for search. Please scrape some URLs first.")
            return
        
        print("\n=== Interactive Search Mode ===")
        print("Enter keywords to search (or 'quit' to exit):")
        
        while True:
            query = input("\nSearch query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = self.search_content(query, top_k=3)
            
            if results:
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['title']}")
                    print(f"   URL: {result['url']}")
                    print(f"   Similarity: {result['similarity_score']:.3f}")
                    print(f"   Preview: {result['content_preview']}")
            else:
                print("No relevant results found.")

def main():
    """Main function to run the application"""
    app = WebScrapingMLApp()
    
    # Example URLs - replace with your target URLs
    example_urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://quotes.toscrape.com/",
        "https://books.toscrape.com/"
    ]
    
    print("Web Scraping + ML Keyword Search Application")
    print("=" * 50)
    
    # Scrape and analyze
    results = app.scrape_and_analyze(example_urls)
    
    if results:
        # Display key findings
        print("\n=== Key Terms Found ===")
        for term, score in results['analysis'].get('key_terms', [])[:10]:
            print(f"{term}: {score:.3f}")
        
        # Display topics if available
        if 'topics' in results['analysis']:
            print("\n=== Topics Discovered ===")
            for topic_name, topic_data in results['analysis']['topics'].items():
                print(f"{topic_name}: {', '.join(topic_data['words'][:5])}")
        
        # Save results
        app.save_results(results)
        
        # Example searches
        print("\n=== Example Searches ===")
        
        search_queries = ["python programming", "web development", "data science"]
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            search_results = app.search_content(query, top_k=2)
            
            for result in search_results:
                print(f"  - {result['title']} (Score: {result['similarity_score']:.3f})")
        
        # Interactive search
        try:
            app.interactive_search()
        except KeyboardInterrupt:
            print("\nExiting...")
    
    print("\nApplication completed!")

if __name__ == "__main__":
    main()
