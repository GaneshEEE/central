import os
import io
import re
import csv
import json
import time
import traceback
import warnings
import requests
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fpdf import FPDF
from docx import Document
from dotenv import load_dotenv
from atlassian import Confluence
import google.generativeai as genai
from bs4 import BeautifulSoup
from io import BytesIO
import difflib
import base64
import openai
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

app = FastAPI(title="Confluence AI Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "https://central-1rua.onrender.com",  # Add your Render URL
        "https://central-frontend-gbvy.onrender.com",  # Add frontend domain
        "*"  # For development, you can allow all origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment
GEMINI_API_KEY = os.getenv("GENAI_API_KEY_1") or os.getenv("GENAI_API_KEY_2")
if not GEMINI_API_KEY:
    raise ValueError("No Gemini API key found in environment variables. Please set GENAI_API_KEY_1 or GENAI_API_KEY_2 in your .env file.")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Initialize ChromaDB client (singleton)
chroma_client = chromadb.Client(Settings(persist_directory=".chroma_db"))

# Embedding function using OpenAI
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_embedding(text: str) -> list:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    response = openai.Embedding.create(
        input=text,
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
    return response["data"][0]["embedding"]

# ChromaDB collection for Confluence pages
CHROMA_COLLECTION_NAME = "confluence_pages"

def get_chroma_collection():
    if CHROMA_COLLECTION_NAME not in [c.name for c in chroma_client.list_collections()]:
        chroma_client.create_collection(CHROMA_COLLECTION_NAME)
    return chroma_client.get_collection(CHROMA_COLLECTION_NAME)

def index_confluence_pages(confluence, space_key: str):
    collection = get_chroma_collection()
    pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
    for page in pages:
        page_id = page["id"]
        title = page["title"]
        page_data = confluence.get_page_by_id(page_id, expand="body.storage")
        raw_html = page_data["body"]["storage"]["value"]
        text_content = clean_html(raw_html)
        doc_id = f"{space_key}:{page_id}"
        # Check if already indexed
        if not collection.get(ids=[doc_id])["ids"]:
            embedding = get_embedding(text_content)
            collection.add(
                documents=[text_content],
                metadatas=[{"title": title, "space_key": space_key, "page_id": page_id}],
                ids=[doc_id],
                embeddings=[embedding]
            )

def retrieve_relevant_context(query: str, top_k: int = 3) -> str:
    collection = get_chroma_collection()
    embedding = get_embedding(query)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    titles = [meta.get("title", "") for meta in results.get("metadatas", [[]])[0]]
    context = "\n\n".join([f"Title: {t}\n{d}" for t, d in zip(titles, docs)])
    return context

# Pydantic models for request/response
class SearchRequest(BaseModel):
    space_key: str
    page_titles: List[str]
    query: str

class VideoRequest(BaseModel):
    video_url: Optional[str] = None
    space_key: str
    page_title: str
    question: Optional[str] = None

class CodeRequest(BaseModel):
    space_key: str
    page_title: str
    instruction: str
    target_language: Optional[str] = None

class ImpactRequest(BaseModel):
    space_key: str
    old_page_title: str
    new_page_title: str
    question: Optional[str] = None

class TestRequest(BaseModel):
    space_key: str
    code_page_title: str
    test_input_page_title: Optional[str] = None
    question: Optional[str] = None

class ImageRequest(BaseModel):
    space_key: str
    page_title: str
    image_url: str

class ImageSummaryRequest(BaseModel):
    space_key: str
    page_title: str
    image_url: str
    summary: str
    question: str

class ChartRequest(BaseModel):
    space_key: str
    page_title: str
    image_url: str
    chart_type: str
    filename: str
    format: str

class ExportRequest(BaseModel):
    content: str
    format: str
    filename: str

class SaveToConfluenceRequest(BaseModel):
    space_key: Optional[str] = None
    page_title: str
    content: str

# Helper functions
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    no_emoji = emoji_pattern.sub(r'', text)
    return no_emoji.encode('latin-1', 'ignore').decode('latin-1')

def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n")

def init_confluence():
    try:
        return Confluence(
            url=os.getenv('CONFLUENCE_BASE_URL'),
            username=os.getenv('CONFLUENCE_USER_EMAIL'),
            password=os.getenv('CONFLUENCE_API_KEY'),
            timeout=10
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Confluence initialization failed: {str(e)}")

# Export functions
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return io.BytesIO(pdf.output(dest='S').encode('latin1'))

def create_docx(text):
    doc = Document()
    for line in text.split('\n'):
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def create_csv(text):
    output = io.StringIO()
    writer = csv.writer(output)
    for line in text.strip().split('\n'):
        writer.writerow([line])
    return io.BytesIO(output.getvalue().encode())

def create_json(text):
    return io.BytesIO(json.dumps({"response": text}, indent=4).encode())

def create_html(text):
    html = f"<html><body><pre>{text}</pre></body></html>"
    return io.BytesIO(html.encode())

def create_txt(text):
    return io.BytesIO(text.encode())


def extract_timestamps_from_summary(summary):
    timestamps = []
    lines = summary.splitlines()
    collecting = False
    for line in lines:
        if "**Timestamps:**" in line or "Timestamps:" in line:
            collecting = True
            continue
        if collecting:
            if not line.strip() or line.strip().startswith("**"):
                break
            # match lines like "* [00:00-00:05] sentence" or "[00:00-00:05] sentence"
            match = re.match(r"^\*?\s*\[(\d{1,2}:\d{2}-\d{1,2}:\d{2})\]\s*(.*)", line.strip())
            if match:
                timestamp_text = f"[{match.group(1)}] {match.group(2)}"
                timestamps.append(timestamp_text)
            elif line.strip().startswith("*") or line.strip().startswith("-"):
                # fallback for bullet points
                timestamps.append(line.strip().lstrip("* -").strip())
            elif line.strip():
                # fallback for any non-empty line
                timestamps.append(line.strip())
    return timestamps

def auto_detect_space(confluence, space_key: Optional[str] = None) -> str:
    """
    If space_key is provided and valid, return it.
    If not provided, auto-detect:
      - If only one space exists, return its key.
      - If multiple, raise error to specify.
    """
    if space_key:
        return space_key
    spaces = confluence.get_all_spaces(start=0, limit=100)["results"]
    if len(spaces) == 1:
        return spaces[0]["key"]
    raise HTTPException(status_code=400, detail="Multiple spaces found. Please specify a space_key.")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Confluence AI Assistant API", "status": "running"}

@app.get("/spaces")
async def get_spaces():
    """Get all available Confluence spaces"""
    try:
        confluence = init_confluence()
        
        spaces = confluence.get_all_spaces(start=0, limit=100)["results"]
        space_options = [{"name": s['name'], "key": s['key']} for s in spaces]
        
        return {"spaces": space_options}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pages/{space_key}")
async def get_pages(space_key: Optional[str] = None):
    """Get all pages from a specific space (auto-detect if not provided)"""
    try:
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, space_key)
        
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
        page_titles = [p["title"] for p in pages]
        
        return {"pages": page_titles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def ai_powered_search(request: SearchRequest, req: Request):
    """AI Powered Search functionality with RAG"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))

        # Index pages if not already indexed
        index_confluence_pages(confluence, space_key)

        # RAG: Retrieve relevant context
        rag_context = retrieve_relevant_context(request.query, top_k=3)

        full_context = ""
        selected_pages = []
        # Get pages
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
        selected_pages = [p for p in pages if p["title"] in request.page_titles]
        if not selected_pages:
            raise HTTPException(status_code=400, detail="No pages found")
        # Extract content from selected pages
        for page in selected_pages:
            page_id = page["id"]
            page_data = confluence.get_page_by_id(page_id, expand="body.storage")
            raw_html = page_data["body"]["storage"]["value"]
            text_content = clean_html(raw_html)
            full_context += f"\n\nTitle: {page['title']}\n{text_content}"
        # Generate AI response
        prompt = (
            f"Use the following retrieved context to help answer the question.\n"
            f"Retrieved context:\n{rag_context}\n\n"
            f"Confluence page context:\n{full_context}\n\n"
            f"Question: {request.query}\n"
            f"Instructions: Begin with the answer based on the context above. Then, if applicable, supplement with general knowledge."
        )
        response = ai_model.generate_content(prompt)
        ai_response = response.text.strip()
        page_titles = [p["title"] for p in selected_pages]
        grounding = f"This answer is based on the following Confluence page(s): {', '.join(page_titles)}."
        return {
            "response": f"{ai_response}\n\n{grounding}",
            "pages_analyzed": len(selected_pages),
            "page_titles": page_titles,
            "grounding": grounding
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/video-summarizer")
async def video_summarizer(request: VideoRequest, req: Request):
    """Video Summarizer functionality using AssemblyAI and Gemini"""
    import requests
    import tempfile
    import subprocess
    import shutil
    confluence = init_confluence()
    space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))

    # Get page info
    pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
    selected_page = next((p for p in pages if p["title"] == request.page_title), None)
    if not selected_page:
        raise HTTPException(status_code=400, detail="Page not found")
    page_id = selected_page["id"]

    # Get attachments
    attachments = confluence.get(f"/rest/api/content/{page_id}/child/attachment?limit=50")
    video_attachment = None
    for att in attachments.get("results", []):
        if att["title"].lower().endswith(".mp4"):
            video_attachment = att
            break
    if not video_attachment:
        raise HTTPException(status_code=404, detail="No .mp4 video attachment found on this page.")

    # Download video
    video_url = video_attachment["_links"]["download"]
    full_url = f"{os.getenv('CONFLUENCE_BASE_URL').rstrip('/')}{video_url}"
    video_name = video_attachment["title"].replace(" ", "_")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, video_name)
        audio_path = os.path.join(tmpdir, "audio.mp3")
        # Download video file
        video_data = confluence._session.get(full_url).content
        with open(video_path, "wb") as f:
            f.write(video_data)
        # Extract audio using ffmpeg
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_path
            ], check=True, capture_output=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ffmpeg audio extraction failed: {e}")
        # Upload audio to AssemblyAI
        assemblyai_api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not assemblyai_api_key:
            raise HTTPException(status_code=500, detail="AssemblyAI API key not configured. Please set ASSEMBLYAI_API_KEY in your environment variables.")
        headers = {"authorization": assemblyai_api_key}
        with open(audio_path, "rb") as f:
            upload_response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                data=f
            )
        if upload_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to upload audio to AssemblyAI")
        audio_url = upload_response.json()["upload_url"]
        # Submit for transcription
        transcript_request = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "auto_chapters": True,
            "auto_highlights": True,
            "entity_detection": True,
            "sentiment_analysis": True
        }
        transcript_response = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json=transcript_request,
            headers={**headers, "content-type": "application/json"}
        )
        if transcript_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to submit audio for transcription")
        transcript_id = transcript_response.json()["id"]
        # Poll for completion
        while True:
            polling_response = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers
            )
            if polling_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to get transcription status")
            status = polling_response.json()["status"]
            if status == "completed":
                break
            elif status == "error":
                raise HTTPException(status_code=500, detail="Transcription failed")
            time.sleep(3)
        transcript_data = polling_response.json()
        transcript_text = transcript_data.get("text", "")
        if not transcript_text:
            raise HTTPException(status_code=500, detail="No transcript text returned from AssemblyAI")
        
        # Initialize Gemini AI model for text generation
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        
        # Q&A
        if request.question:
            qa_prompt = (
                f"Based on the following video transcript, answer this question: {request.question}\n\n"
                f"Transcript: {transcript_text[:3000]}\n\n"
                f"Provide a detailed answer based on the video content."
            )
            qa_response = ai_model.generate_content(qa_prompt)
            grounding = f"This answer is based on the transcript of the Confluence page: {request.page_title}."
            return {"answer": f"{qa_response.text.strip()}\n\n{grounding}", "grounding": grounding}
        
        # Generate quotes
        quote_prompt = (
            "Extract 3-5 powerful or interesting quotes from the transcript.\n"
            "Format each quote on a new line starting with a dash (-).\n"
            f"Transcript:\n{transcript_text[:3000]}"
        )
        quotes_response = ai_model.generate_content(quote_prompt).text.strip()
        # Split quotes into individual items
        quotes = [quote.strip().lstrip("- ").strip() for quote in quotes_response.split('\n') if quote.strip()]
        
        # Generate summary WITHOUT timestamps
        summary_prompt = (
            "detailed paragraph summarizing the video content.\n"
            "Do NOT include any timestamps in the summary.\n"
            f"Transcript:\n{transcript_text[:3000]}"
        )
        summary = ai_model.generate_content(summary_prompt).text.strip()
        
        # Generate timestamps separately
        timestamp_prompt = (
            "Extract 5-7 important moments from the following transcript.\n"
            "Format each moment as: [MM:SS-MM:SS] Description of what happens\n"
            "Example: [00:15-00:30] Speaker introduces the main topic\n"
            "Return only the timestamps, one per line.\n\n"
            f"Transcript:\n{transcript_text[:3000]}"
        )
        timestamps_response = ai_model.generate_content(timestamp_prompt).text.strip()
        # Split timestamps into individual items
        timestamps = [ts.strip() for ts in timestamps_response.split('\n') if ts.strip()]
        
        return {
            "summary": summary,
            "quotes": quotes,
            "timestamps": timestamps,
            "qa": [],
            "page_title": request.page_title,
            "transcript": transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text,
            "video_url": full_url
        }


@app.post("/code-assistant")
async def code_assistant(request: CodeRequest, req: Request):
    """Code Assistant functionality with RAG"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))

        # Index pages if not already indexed
        index_confluence_pages(confluence, space_key)

        # RAG: Retrieve relevant context
        rag_context = retrieve_relevant_context(request.instruction or request.page_title, top_k=3)

        # Get page content
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
        selected_page = next((p for p in pages if p["title"] == request.page_title), None)
        if not selected_page:
            raise HTTPException(status_code=400, detail="Page not found")
        page_id = selected_page["id"]
        page_content = confluence.get_page_by_id(page_id, expand="body.storage")
        context = page_content["body"]["storage"]["value"]
        # Extract visible code
        soup = BeautifulSoup(context, "html.parser")
        for tag in soup.find_all(['pre', 'code']):
            code_text = tag.get_text()
            if code_text.strip():
                cleaned_code = code_text
                break
        else:
            cleaned_code = soup.get_text(separator="\n").strip()
        # Detect language
        def detect_language_from_content(content: str) -> str:
            if "<?xml" in content:
                return "xml"
            if "<html" in content.lower() or "<!DOCTYPE html>" in content:
                return "html"
            if content.strip().startswith("{") or content.strip().startswith("["):
                return "json"
            if re.search(r"\bclass\s+\w+", content) and "public" in content:
                return "java"
            if "#include" in content:
                return "cpp"
            if "def " in content:
                return "python"
            if "function" in content or "=>" in content:
                return "javascript"
            return "text"
        detected_lang = detect_language_from_content(cleaned_code)
        # Generate summary
        summary_prompt = (
            f"Use the following retrieved context to help summarize the code.\n"
            f"Retrieved context:\n{rag_context}\n\n"
            f"The following is content (possibly code or structure) from a Confluence page:\n\n{context}\n\n"
            "Summarize in detailed paragraph"
        )
        summary_response = ai_model.generate_content(summary_prompt)
        summary = summary_response.text.strip()
        # Modify code if instruction provided
        modified_code = None
        if request.instruction:
            alteration_prompt = (
                f"Use the following retrieved context to help modify the code.\n"
                f"Retrieved context:\n{rag_context}\n\n"
                f"The following is a piece of code extracted from a Confluence page:\n\n{cleaned_code}\n\n"
                f"Please modify this code according to the following instruction:\n'{request.instruction}'\n\n"
                "Return the modified code only. No explanation or extra text."
            )
            altered_response = ai_model.generate_content(alteration_prompt)
            modified_code = re.sub(r"^```[a-zA-Z]*\n|```$", "", altered_response.text.strip(), flags=re.MULTILINE)
        # Convert to another language if requested
        converted_code = None
        if request.target_language and request.target_language != detected_lang:
            input_code = modified_code if modified_code else cleaned_code
            convert_prompt = (
                f"Use the following retrieved context to help convert the code.\n"
                f"Retrieved context:\n{rag_context}\n\n"
                f"The following is a code structure or data snippet:\n\n{input_code}\n\n"
                f"Convert this into equivalent {request.target_language} code. Only show the converted code."
            )
            lang_response = ai_model.generate_content(convert_prompt)
            converted_code = re.sub(r"^```[a-zA-Z]*\n|```$", "", lang_response.text.strip(), flags=re.MULTILINE)
        grounding = f"This answer is based on the code/content from the Confluence page: {request.page_title}."
        return {
            "summary": f"{summary}\n\n{grounding}",
            "original_code": cleaned_code,
            "detected_language": detected_lang,
            "modified_code": (f"{modified_code}\n\n{grounding}" if modified_code else None),
            "converted_code": (f"{converted_code}\n\n{grounding}" if converted_code else None),
            "target_language": request.target_language,
            "grounding": grounding
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/impact-analyzer")
async def impact_analyzer(request: ImpactRequest, req: Request):
    """Impact Analyzer functionality with RAG"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))

        # Index pages if not already indexed
        index_confluence_pages(confluence, space_key)

        # RAG: Retrieve relevant context
        rag_context = retrieve_relevant_context(request.question or request.old_page_title + ' ' + request.new_page_title, top_k=3)

        # Get pages
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
        old_page = next((p for p in pages if p["title"] == request.old_page_title), None)
        new_page = next((p for p in pages if p["title"] == request.new_page_title), None)
        if not old_page or not new_page:
            raise HTTPException(status_code=400, detail="One or both pages not found")
        # Extract content from pages
        def extract_content(content):
            soup = BeautifulSoup(content, 'html.parser')
            code_blocks = soup.find_all('ac:structured-macro', {'ac:name': 'code'})
            if code_blocks:
                return '\n'.join(
                    block.find('ac:plain-text-body').text
                    for block in code_blocks if block.find('ac:plain-text-body')
                )
            return soup.get_text(separator="\n").strip()
        old_raw = confluence.get_page_by_id(old_page["id"], expand="body.storage")["body"]["storage"]["value"]
        new_raw = confluence.get_page_by_id(new_page["id"], expand="body.storage")["body"]["storage"]["value"]
        old_content = extract_content(old_raw)
        new_content = extract_content(new_raw)
        if not old_content or not new_content:
            raise HTTPException(status_code=400, detail="No content found in one or both pages")
        # Generate diff
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        diff = difflib.unified_diff(old_lines, new_lines, fromfile=request.old_page_title, tofile=request.new_page_title, lineterm='')
        full_diff_text = '\n'.join(diff)
        # Calculate metrics
        lines_added = sum(1 for l in full_diff_text.splitlines() if l.startswith('+') and not l.startswith('+++'))
        lines_removed = sum(1 for l in full_diff_text.splitlines() if l.startswith('-') and not l.startswith('---'))
        total_lines = len(old_lines) or 1
        percent_change = round(((lines_added + lines_removed) / total_lines) * 100, 2)
        # Generate AI analysis
        def clean_and_truncate_prompt(text, max_chars=10000):
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            return text[:max_chars]
        safe_diff = clean_and_truncate_prompt(full_diff_text)
        # Impact analysis
        impact_prompt = f"""Use the following retrieved context to help analyze the impact.\nRetrieved context:\n{rag_context}\n\nWrite 2 paragraphs summarizing the overall impact of the following changes between two versions of a document.\n\nCover only:\n- What was changed\n- Which parts of the content are affected\n- Why this matters\n\nKeep it within 20 sentences.\n\nChanges:\n{safe_diff}"""
        impact_response = ai_model.generate_content(impact_prompt)
        impact_text = impact_response.text.strip()
        # Recommendations
        rec_prompt = f"""Use the following retrieved context to help suggest improvements.\nRetrieved context:\n{rag_context}\n\nAs a senior analyst, write 2 paragraphs suggesting improvements for the following changes.\n\nFocus on:\n- Content quality\n- Clarity and completeness\n- Any possible enhancements\n\nLimit to 20 sentences.\n\nChanges:\n{safe_diff}"""
        rec_response = ai_model.generate_content(rec_prompt)
        rec_text = rec_response.text.strip()
        # Risk analysis
        risk_prompt = f"Use the following retrieved context to help assess risk.\nRetrieved context:\n{rag_context}\n\nAssess the risk of each change in this document diff with severity tags (Low, Medium, High):\n\n{safe_diff}"
        risk_response = ai_model.generate_content(risk_prompt)
        raw_risk = risk_response.text.strip()
        risk_text = re.sub(
            r'\b(Low|Medium|High)\b',
            lambda m: {
                'Low': '🟢 Low',
                'Medium': '🟡 Medium',
                'High': '🔴 High'
            }[m.group(0)],
            raw_risk
        )
        # Generate structured risk factors (new dynamic part)
        risk_factors_prompt = f"""
        Use the following retrieved context to help extract risk factors.\nRetrieved context:\n{rag_context}\n\nAnalyze the following code/content diff and extract a structured list of key risk factors introduced by these changes.\n\nFocus on identifying:\n- Broken or removed validation\n- Modified authentication/authorization checks\n- Logical regressions\n- Removed error handling\n- Performance or scalability risks\n- Security vulnerabilities\n- Stability or maintainability concerns\n\nWrite each risk factor as 1 line. Avoid repeating obvious stats like line count.\n\nDiff:\n{safe_diff}\n"""
        risk_factors_response = ai_model.generate_content(risk_factors_prompt)
        risk_factors = risk_factors_response.text.strip().split("\n")
        risk_factors = [re.sub(r"^[\*\-•\s]+", "", line).strip() for line in risk_factors if line.strip()]
        # Q&A if question provided
        qa_answer = None
        grounding = f"This answer is based on the diff between Confluence pages: {request.old_page_title} and {request.new_page_title}."
        if request.question:
            context = (
                f"Summary: {impact_text[:1000]}\n"
                f"Recommendations: {rec_text[:1000]}\n"
                f"Risks: {risk_text[:1000]}\n"
                f"Changes: +{lines_added}, -{lines_removed}, ~{percent_change}%"
            )
            qa_prompt = f"""Use the following retrieved context to help answer the question.\nRetrieved context:\n{rag_context}\n\nYou are an expert AI assistant. Based on the report below, answer the user's question clearly.\n\n{context}\n\nQuestion: {request.question}\n\nAnswer:"""
            qa_response = ai_model.generate_content(qa_prompt)
            qa_answer = f"{qa_response.text.strip()}\n\n{grounding}"
        return {
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "files_changed": 1,
            "percentage_change": percent_change,
            "impact_analysis": f"{impact_text}\n\n{grounding}",
            "recommendations": f"{rec_text}\n\n{grounding}",
            "risk_analysis": f"{risk_text}\n\n{grounding}",
            "risk_level": "low" if percent_change < 10 else "medium" if percent_change < 30 else "high",
            "risk_score": min(10, max(1, round(percent_change / 10))),
            "risk_factors": risk_factors,
            "answer": qa_answer,
            "diff": full_diff_text,
            "grounding": grounding
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-support")
async def test_support(request: TestRequest, req: Request):
    """Test Support Tool functionality with RAG"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        print(f"Test support request: {request}")  # Debug log
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))

        # Index pages if not already indexed
        index_confluence_pages(confluence, space_key)

        # RAG: Retrieve relevant context
        rag_context = retrieve_relevant_context(request.question or request.code_page_title, top_k=3)

        # Get code page
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=50)
        code_page = next((p for p in pages if p["title"] == request.code_page_title), None)
        if not code_page:
            raise HTTPException(status_code=400, detail="Code page not found")
        print(f"Found code page: {code_page['title']}")  # Debug log
        code_data = confluence.get_page_by_id(code_page["id"], expand="body.storage")
        code_content = code_data["body"]["storage"]["value"]
        print(f"Code content length: {len(code_content)}")  # Debug log
        # Generate test strategy
        prompt_strategy = f"""Use the following retrieved context to help generate the test strategy.\nRetrieved context:\n{rag_context}\n\nThe following is a code snippet:\n\n{code_content[:2000]}\n\nPlease generate a **structured test strategy** for the above code using the following format. \n\nMake sure each section heading is **clearly labeled** and includes a **percentage estimate** of total testing effort and the total of all percentage values across Unit Test, Integration Test, and End-to-End (E2E) Test must add up to exactly **100%**. Each subpoint should be short (1–2 lines max). Use bullet points for clarity.\n\n---\n\n\n## Unit Test (xx%)\n- **Coverage Areas**:  \n  - What functions or UI elements are directly tested?  \n- **Edge Cases**:  \n  - List 2–3 specific edge conditions or unusual inputs.\n\n## Integration Test (xx%)\n- **Integrated Modules**:  \n  - What parts of the system work together and need testing as a unit?  \n- **Data Flow Validation**:  \n  - How does data move between components or layers?\n\n## End-to-End (E2E) Test (xx%)\n- **User Scenarios**:  \n  - Provide 2–3 user flows that simulate real usage.  \n- **System Dependencies**:  \n  - What systems, APIs, or services must be operational?\n\n## Test Data Management\n- **Data Requirements**:  \n  - What test data (e.g., users, tokens, inputs) is needed?  \n- **Data Setup & Teardown**:  \n  - How is test data created and removed?\n\n## Automation Strategy\n- **Frameworks/Tools**:  \n  - Recommend tools for each test level.  \n- **CI/CD Integration**:  \n  - How will tests be included in automated pipelines?\n\n## Risk Areas Identified\n- **Complex Logic**:  \n  - Highlight any logic that's error-prone or tricky.  \n- **Third-Party Dependencies**:  \n  - Any reliance on external APIs or libraries?  \n- **Security/Critical Flows**:  \n  - Mention any data protection or authentication flows.\n\n## Additional Considerations\n- **Security**:  \n  - Are there vulnerabilities or security-sensitive operations?  \n- **Accessibility**:  \n  - Are there any compliance or usability needs?  \n- **Performance**:  \n  - Should speed, responsiveness, or load handling be tested?\n\n---\n\nPlease format your response exactly like this structure, using proper markdown headings, short bullet points, and estimated test effort percentages. """
        response_strategy = ai_model.generate_content(prompt_strategy)
        strategy_text = response_strategy.text.strip()
        print(f"Strategy generated: {len(strategy_text)} chars")  # Debug log
        # Generate cross-platform testing
        prompt_cross_platform = f"""Use the following retrieved context to help generate the cross-platform strategy.\nRetrieved context:\n{rag_context}\n\nYou are a cross-platform UI testing expert. Analyze the following frontend code and generate a detailed cross-platform test strategy using the structure below. Your insights should be **relevant to the code**, not generic. Code:\n\n{code_content[:2000]}\n\nFollow the format strictly and customize values based on the code analysis. Avoid repeating default phrases — provide actual testing considerations derived from the code.\n\n---\n\n\n## Platform Coverage Assessment\n\n### Web Browsers\n- **Chrome**: [Insert expected behavior or issues specific to the code]  \n- **Firefox**: [Insert any rendering quirks, compatibility notes, or enhancements]  \n- **Safari**: [Highlight any issues with WebKit or mobile Safari]  \n- **Edge**: [Mention compatibility or layout differences]  \n- **Mobile Browsers**: [Describe responsive behavior, touch issues, or layout breaks]  \n\n### Operating Systems\n- **Windows**: [Describe any dependency or rendering issues noticed]  \n- **macOS**: [Note differences in rendering, fonts, or interactions]  \n- **Linux**: [Mention support in containerized or open environments]  \n- **Mobile iOS**: [Identify areas needing testing on Safari iOS or WebView]  \n- **Android**: [Highlight performance, scrolling, or viewport concerns]  \n\n### Device Categories\n- **Desktop**: [List full UI/feature behavior on large screens]  \n- **Tablet**: [Mention any layout shifting, input mode support, or constraints]  \n- **Mobile**: [List responsiveness issues or changes in UI behavior]  \n- **Accessibility**: [Accessibility tags, ARIA usage, screen reader compatibility]  \n\n## Testing Approach\n\n### Automated Cross-Platform Testing\n- **Browser Stack Integration**: [Which browsers/devices to target and why]  \n- **Device Farm Testing**: [Recommend real-device scenarios to validate]  \n- **Performance Benchmarking**: [How platform differences might affect performance]  \n\n### Manual Testing Strategy\n- **User Acceptance Testing**: [Suggest user workflows to validate on each platform]  \n- **Accessibility Testing**: [Mention checks like tab order, ARIA roles, color contrast]  \n- **Localization Testing**: [If text/UI is dynamic, how to test translations or RTL]  \n\n## Platform-Specific Considerations\n\n### Performance Optimization\n- **Mobile**: [Mention any heavy assets, unused JS/CSS, or optimizations needed]  \n- **Desktop**: [Advanced UI behaviors or feature flags that only show on desktop]  \n- **Tablets**: [Navigation patterns or split-view compatibility]  \n\n### Security Implications\n- **iOS**: [Any app/webview permissions or secure storage issues]  \n- **Android**: [Issues with file access, permissions, or deep linking]  \n- **Web**: [CSP, HTTPS enforcement, token handling or XSS risks]  \n\n---\n\nRespond **exactly** in this format with dynamic insights, no extra text outside the structure. """
        response_cross_platform = ai_model.generate_content(prompt_cross_platform)
        cross_text = response_cross_platform.text.strip()
        print(f"Cross-platform generated: {len(cross_text)} chars")  # Debug log
        # Sensitivity analysis if test input page provided
        sensitivity_text = None
        if request.test_input_page_title:
            test_input_page = next((p for p in pages if p["title"] == request.test_input_page_title), None)
            if test_input_page:
                test_data = confluence.get_page_by_id(test_input_page["id"], expand="body.storage")
                test_input_content = test_data["body"]["storage"]["value"]
                prompt_sensitivity = f"""Use the following retrieved context to help analyze sensitivity.\nRetrieved context:\n{rag_context}\n\nYou are a data privacy expert. Classify sensitive fields (PII, credentials, financial) and provide masking suggestions.Also, don't include comments if any code is present.\n\nData:\n{test_input_content[:2000]}"""
                response_sensitivity = ai_model.generate_content(prompt_sensitivity)
                sensitivity_text = response_sensitivity.text.strip()
                print(f"Sensitivity generated: {len(sensitivity_text)} chars")  # Debug log
        # Q&A if question provided
        ai_response = None
        grounding = f"This answer is based on the code/content from the Confluence page: {request.code_page_title}."
        if request.question:
            context = f"📘 Test Strategy:\n{strategy_text}\n🌐 Cross-Platform Testing:\n{cross_text}"
            if sensitivity_text:
                context += f"\n🔒 Sensitivity Analysis:\n{sensitivity_text}"
            prompt_chat = f"""Use the following retrieved context to help answer the question.\nRetrieved context:\n{rag_context}\n\nBased on the following content:\n{context}\n\nAnswer this user query: \"{request.question}\" """
            response_chat = ai_model.generate_content(prompt_chat)
            ai_response = f"{response_chat.text.strip()}\n\n{grounding}"
            print(f"Q&A generated: {len(ai_response)} chars")  # Debug log
        result = {
            "test_strategy": f"{strategy_text}\n\n{grounding}",
            "cross_platform_testing": f"{cross_text}\n\n{grounding}",
            "sensitivity_analysis": (f"{sensitivity_text}\n\n{grounding}" if sensitivity_text else None),
            "ai_response": ai_response,
            "grounding": grounding
        }
        print(f"Returning result: {result}")  # Debug log
        return result
    except Exception as e:
        print(f"Test support error: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{space_key}/{page_title}")
async def get_images(space_key: Optional[str] = None, page_title: str = ""):
    """Get all images from a specific page"""
    try:
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, space_key)
        
        # Get page content
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
        page = next((p for p in pages if p["title"].strip().lower() == page_title.strip().lower()), None)
        
        if not page:
            raise HTTPException(status_code=404, detail=f"Page '{page_title}' not found")
        
        page_id = page["id"]
        html_content = confluence.get_page_by_id(page_id=page_id, expand="body.export_view")["body"]["export_view"]["value"]
        soup = BeautifulSoup(html_content, "html.parser")
        base_url = os.getenv("CONFLUENCE_BASE_URL")
        
        image_urls = list({
            base_url + img["src"] if img["src"].startswith("/") else img["src"]
            for img in soup.find_all("img") if img.get("src")
        })
        
        return {"images": image_urls}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image-summary")
async def image_summary(request: ImageRequest, req: Request):
    """Generate AI summary for an image"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))
        
        # Download image
        auth = (os.getenv('CONFLUENCE_USER_EMAIL'), os.getenv('CONFLUENCE_API_KEY'))
        response = requests.get(request.image_url, auth=auth)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Failed to fetch image")
        
        image_bytes = response.content
        
        # Upload to Gemini
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            uploaded = genai.upload_file(
                path=tmp.name,
                mime_type="image/png",
                display_name=f"confluence_image_{request.page_title}.png"
            )
        
        prompt = (
            "You are analyzing a technical image from a documentation page. "
            "If it's a chart or graph, explain what is shown in detail. "
            "If it's code, summarize what the code does. "
            "Avoid mentioning filenames or metadata. Provide an informative analysis in 1 paragraph."
        )
        
        response = ai_model.generate_content([uploaded, prompt])
        summary = response.text.strip()
        grounding = "Grounding: This answer is based on the provided image content."
        
        return {"summary": f"{summary}\n\n{grounding}", "grounding": grounding}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image-qa")
async def image_qa(request: ImageSummaryRequest, req: Request):
    """Generate AI response for a question about an image"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))
        
        # Download image
        auth = (os.getenv('CONFLUENCE_USER_EMAIL'), os.getenv('CONFLUENCE_API_KEY'))
        response = requests.get(request.image_url, auth=auth)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Failed to fetch image")
        
        image_bytes = response.content
        
        # Upload to Gemini
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(image_bytes)
            tmp_img.flush()
            uploaded_img = genai.upload_file(
                path=tmp_img.name,
                mime_type="image/png",
                display_name=f"qa_image_{request.page_title}.png"
            )
        
        full_prompt = (
            "You're analyzing a technical image extracted from documentation. "
            "Answer the user's question based on the visual content of the image, "
            "as well as the summary below.\n\n"
            f"Summary:\n{request.summary}\n\n"
            f"User Question:\n{request.question}"
        )
        
        ai_response = ai_model.generate_content([uploaded_img, full_prompt])
        answer = ai_response.text.strip()
        grounding = "Grounding: This answer is based on the provided image and summary content."
        
        return {"answer": f"{answer}\n\n{grounding}", "grounding": grounding}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-chart")
async def create_chart(request: ChartRequest, req: Request):
    """Create chart from image data"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))
        
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from io import StringIO
        
        # Download image
        auth = (os.getenv('CONFLUENCE_USER_EMAIL'), os.getenv('CONFLUENCE_API_KEY'))
        response = requests.get(request.image_url, auth=auth)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Failed to fetch image")
        
        image_bytes = response.content
        
        # Upload to Gemini for data extraction
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(image_bytes)
            tmp_img.flush()
            uploaded_img = genai.upload_file(
                path=tmp_img.name,
                mime_type="image/png",
                display_name=f"chart_image_{request.page_title}.png"
            )
        
        graph_prompt = (
            "You're looking at a Likert-style bar chart image or table. Extract the full numeric table represented by the chart.\n"
            "Return only the raw CSV table: no markdown, no comments, no code blocks.\n"
            "The first column must be the response category (e.g., Strongly Agree), followed by columns for group counts (e.g., Students, Lecturers, Staff, Total).\n"
            "Ensure all values are numeric and the CSV is properly aligned. Do NOT summarize—just output the table."
        )
        
        graph_response = ai_model.generate_content([uploaded_img, graph_prompt])
        csv_text = graph_response.text.strip()
        
        # Clean CSV data
        def clean_ai_csv(raw_text):
            lines = raw_text.strip().splitlines()
            clean_lines = [
                line.strip() for line in lines
                if ',' in line and not line.strip().startswith("```") and not line.lower().startswith("here")
            ]
            header = clean_lines[0].split(",")
            cleaned_data = [clean_lines[0]]
            for line in clean_lines[1:]:
                if line.split(",")[0] != header[0]:
                    cleaned_data.append(line)
            return "\n".join(cleaned_data)
        
        cleaned_csv = clean_ai_csv(csv_text)
        df = pd.read_csv(StringIO(cleaned_csv))
        
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=df.columns[1:], how='all', inplace=True)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Failed to extract chart data from image")
        
        # Create chart based on type
        if request.chart_type == "Grouped Bar":
            melted = df.melt(id_vars=[df.columns[0]], var_name="Group", value_name="Count")
            plt.figure(figsize=(10, 6))
            sns.barplot(data=melted, x=melted.columns[0], y="Count", hue="Group")
            plt.xticks(rotation=45)
            plt.title("Grouped Bar Chart")
            plt.tight_layout()
        elif request.chart_type == "Stacked Bar":
            df_plot = df.set_index(df.columns[0])
            plt.figure(figsize=(10, 6))
            df_plot.drop(columns="Total", errors="ignore").plot(kind='bar', stacked=True)
            plt.title("Stacked Bar Chart")
            plt.xticks(rotation=45)
            plt.ylabel("Count")
            plt.tight_layout()
        elif request.chart_type == "Line":
            df_plot = df.set_index(df.columns[0])
            plt.figure(figsize=(10, 6))
            df_plot.drop(columns="Total", errors="ignore").plot(marker='o')
            plt.title("Line Chart")
            plt.xticks(rotation=45)
            plt.ylabel("Count")
            plt.tight_layout()
        elif request.chart_type == "Pie":
            plt.figure(figsize=(7, 6))
            label_col = df.columns[0]
            if "Total" in df.columns:
                data = df["Total"]
            else:
                data = df.iloc[:, 1:].sum(axis=1)
            plt.pie(data, labels=df[label_col], autopct="%1.1f%%", startangle=140)
            plt.title("Pie Chart (Total Responses)")
            plt.tight_layout()
        
        # Save chart to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format=request.format.lower(), bbox_inches="tight")
        buf.seek(0)
        chart_bytes = buf.getvalue()
        
        # Convert to base64 for response
        chart_base64 = base64.b64encode(chart_bytes).decode()
        
        return {
            "chart_data": chart_base64,
            "mime_type": f"image/{request.format.lower()}",
            "filename": f"{request.filename}.{request.format.lower()}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export")
async def export_content(request: ExportRequest, req: Request):
    """Export content in various formats"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        if request.format == "pdf":
            buffer = create_pdf(request.content)
            file_data = buffer.getvalue()
            return {"file": base64.b64encode(file_data).decode('utf-8'), "mime": "application/pdf", "filename": f"{request.filename}.pdf"}
        elif request.format == "docx":
            buffer = create_docx(request.content)
            file_data = buffer.getvalue()
            return {"file": base64.b64encode(file_data).decode('utf-8'), "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "filename": f"{request.filename}.docx"}
        elif request.format == "csv":
            buffer = create_csv(request.content)
            file_data = buffer.getvalue()
            return {"file": file_data.decode('utf-8'), "mime": "text/csv", "filename": f"{request.filename}.csv"}
        elif request.format == "json":
            buffer = create_json(request.content)
            file_data = buffer.getvalue()
            return {"file": file_data.decode('utf-8'), "mime": "application/json", "filename": f"{request.filename}.json"}
        elif request.format == "html":
            buffer = create_html(request.content)
            file_data = buffer.getvalue()
            return {"file": file_data.decode('utf-8'), "mime": "text/html", "filename": f"{request.filename}.html"}
        else:  # txt/markdown
            buffer = create_txt(request.content)
            file_data = buffer.getvalue()
            return {"file": file_data.decode('utf-8'), "mime": "text/plain", "filename": f"{request.filename}.txt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-to-confluence")
async def save_to_confluence(request: SaveToConfluenceRequest, req: Request):
    """
    Update the content of a Confluence page (storage format).
    """
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, request.space_key)
        # Get page by title, expand body.storage
        page = confluence.get_page_by_title(space=space_key, title=request.page_title, expand='body.storage')
        if not page:
            raise HTTPException(status_code=404, detail="Page not found")
        page_id = page["id"]
        # Append new content to existing content
        existing_content = page["body"]["storage"]["value"]
        updated_body = existing_content + "<hr/>" + request.content
        # Update page
        confluence.update_page(
            page_id=page_id,
            title=request.page_title,
            body=updated_body,
            representation="storage"
        )
        return {"message": "Page updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify backend is working"""
    return {"message": "Backend is working", "status": "ok"}

def get_actual_api_key_from_identifier(identifier: str) -> str:
    if identifier and identifier.startswith('GENAI_API_KEY_'):
        key = os.getenv(identifier)
        print(f"Using API key identifier: {identifier}, value: {key}")  # This will appear in Render logs
        if key:
            return key
    fallback = os.getenv('GENAI_API_KEY_1')
    print(f"Falling back to GENAI_API_KEY_1, value: {fallback}")
    return fallback

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
