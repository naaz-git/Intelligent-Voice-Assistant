from serpapi import GoogleSearch
import os
import pickle

def searchBlogsFunc(query, sites):
    SERP_API_KEY    = os.getenv("SERP_API_KEY")
    site_query      = " OR ".join([f"site:{site}" for site in sites])  # Combine sites with OR
    full_query      = f"{query} {site_query}"

    params = {
        "q": full_query,
        "hl": "en",
        "gl": "us",
        "api_key": SERP_API_KEY,
    }

    searchObject    = None
    try:
        if os.path.exists('./searchObject.pkl'):
            with open('./searchObject.pkl', "rb") as file:
                searchObject    = pickle.load(file)
        else:
            raise FileNotFoundError(f"The file '{'./searchObject.pkl'}' does not exist.")
    except:
        searchObject            = GoogleSearch(params)
        with open('./searchObject.pkl', 'wb') as file:
            pickle.dump(searchObject, file)

    results         = searchObject.get_dict()

    # Extract and print search results
    output = []
    if "organic_results" in results:
        for result in results["organic_results"]:
            output.append((result.get('title', 'No title'), result.get('link', 'No link'), result.get('snippet', 'No description') )) 
            print(f"Title: {result.get('title', 'No title')}")
            print(f"Link: {result.get('link', 'No link')}")
            print(f"Description: {result.get('snippet', 'No description')}")
            print("-" * 50)
    else:
        print("No results found.")
    return output
