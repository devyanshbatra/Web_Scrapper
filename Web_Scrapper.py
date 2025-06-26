import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import json

class WebScraper:
    def __init__(self, delay: float = 1.0):
        """
        Initialize the web scraper with rate limiting
        
        Args:
            delay: Delay between requests in seconds
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        """
        Scrape a single webpage and extract text content
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing scraped data or None if failed
        """
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post', '.entry-content', 'section'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text() for elem in elements])
                    break
            
            # If no specific content found, get all text
            if not content:
                content = soup.get_text()
            
            # Clean the text
            content = self._clean_text(content)
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ''
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                if self._is_valid_url(absolute_url):
                    links.append(absolute_url)
            
            scraped_data = {
                'url': url,
                'title': title_text,
                'content': content,
                'description': description,
                'links': links[:10],  # Limit to first 10 links
                'word_count': len(content.split()),
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            time.sleep(self.delay)  # Rate limiting
            return scraped_data
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def scrape_multiple_pages(self, urls: List[str]) -> List[Dict]:
        """
        Scrape multiple webpages
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped data dictionaries
        """
        results = []
        for url in urls:
            result = self.scrape_page(url)
            if result:
                results.append(result)
        return results
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        return text.strip()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not a fragment or mailto link"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.scheme in ['http', 'https']
        except:
            return False
    
    def save_results(self, results: List[Dict], filename: str = 'scraped_data.json'):
        """Save scraping results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")

# Example usage
if __name__ == "__main__":
    scraper = WebScraper(delay=1.0)
    
    # Example URLs to scrape
    urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://quotes.toscrape.com/"
    ]
    
    results = scraper.scrape_multiple_pages(urls)
    
    for result in results:
        print(f"\nTitle: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Word Count: {result['word_count']}")
        print(f"Content Preview: {result['content'][:200]}...")
    
    scraper.save_results(results)
