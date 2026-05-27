import requests  
import json  
  
BASE_URL = "http://localhost:18026"  
  
def check_wiki_cache(owner: str, repo: str, repo_type: str = "github", language: str = "en"):  
    """Check if a repository is cached."""  
    params = {  
        "owner": owner,  
        "repo": repo,  
        "repo_type": repo_type,  
        "language": language  
    }  
    response = requests.get(f"{BASE_URL}/api/wiki_cache", params=params)  
    return response.json() if response.status_code == 200 else None  
  
def ask_question(repo_url: str, question: str, provider: str = "google"):  
    """Ask a question about a repository using the streaming endpoint."""  
    payload = {  
        "repo_url": repo_url,  
        "messages": [{"role": "user", "content": question}],  
        "provider": provider  
    }  
      
    response = requests.post(f"{BASE_URL}/chat/completions/stream", json=payload, stream=True)  
      
    # Process streaming response  
    full_response = ""  
    for chunk in response.iter_content(chunk_size=None):  
        if chunk:  
            full_response += chunk.decode('utf-8')  
      
    return full_response  
  
def list_processed_projects():  
    """List all processed projects."""  
    response = requests.get(f"{BASE_URL}/api/processed_projects")  
    return response.json() if response.status_code == 200 else None  
  
# Example usage  
if __name__ == "__main__":  
    repo_url = "https://github.com/AsyncFuncAI/deepwiki-open"  
    owner = "AsyncFuncAI"  
    repo = "deepwiki-open"  
      
    # Check if cached  
    cache_data = check_wiki_cache(owner, repo)  
    if cache_data:  
        print("Repository is cached!")  
    else:  
        print("Repository not cached")  
      
    # Ask a question  
    question = "What does this repository do?"  
    answer = ask_question(repo_url, question)  
    print(f"Answer: {answer}")  
      
    # List all projects  
    projects = list_processed_projects()  
    print(f"Processed projects: {projects}")