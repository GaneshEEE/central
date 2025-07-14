import google.generativeai as genai
import requests
import os

# === YOUR API KEYS ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")          # Now loaded from environment variable
GOOGLE_API_KEY = "AIzaSyBiI_i8JbSsAVFNiaMBcLw_jABTrS_HXrA"          # Replace this
SEARCH_ENGINE_ID = "d35cd327525264522"                              # Replace this

# === CONFIGURE GEMINI ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")

# === GOOGLE SEARCH FUNCTION ===
def search_web_google(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        if not results:
            return ""

        snippets = []
        for item in results:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            snippets.append(f"{title}\n{snippet}\n{link}")
        return "\n\n".join(snippets)

    except Exception as e:
        return f"❌ Google Search error: {e}"

# === HYBRID RAG ===
def hybrid_rag(prompt):
    web_context = search_web_google(prompt)

    if not web_context.strip():
        response = model.generate_content(prompt)
        return response.parts[0].text.strip() if response.parts else "⚠️ No answer from Gemini."

    final_prompt = f"""
Use the following web search results to answer the question accurately.
If relevant, include links as citations.

Web Results:
{web_context}

Question: {prompt}

Answer:
"""

    try:
        response = model.generate_content(final_prompt)
        text = response.parts[0].text.strip() if response.parts else "⚠️ No answer from Gemini."

        if "do not contain" in text.lower():
            response = model.generate_content(prompt)
            return response.parts[0].text.strip() if response.parts else "⚠️ No fallback answer from Gemini."

        return text

    except Exception as e:
        return f"❌ Gemini error: {e}"

# === MAIN ===
if __name__ == "__main__":
    user_prompt = input("Ask me anything: ")
    print("Please wait while I fetch the best answer...\n")
    answer = hybrid_rag(user_prompt)
    print("\n🤖 Gemini says:\n", answer)
