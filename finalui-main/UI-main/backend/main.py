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
import sys
sys.path.append(r"c:/Users/Dhaya Arun/Downloads/finalui-main/finalui-main")
from test import hybrid_rag
from bs4 import BeautifulSoup
from io import BytesIO
import difflib
import base64
import ast
import pandas as pd

# Load environment variables
load_dotenv()

app = FastAPI(title="Confluence AI Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "https://central-lwsd.onrender.com",  # Add your Render URL
        "https://central-frontend-vvbm.onrender.com",  # Add frontend domain
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

class ExcelRequest(BaseModel):
    space_key: str
    page_title: str
    excel_url: str

class ExcelSummaryRequest(BaseModel):
    space_key: str
    page_title: str
    excel_url: str
    summary: str
    question: str

class ChartFromExcelRequest(BaseModel):
    space_key: str
    page_title: str
    excel_url: str
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

class StackOverflowRiskRequest(BaseModel):
    space_key: str
    old_page_title: str
    new_page_title: str
    diff_content: Optional[str] = None
    code_changes: Optional[str] = None

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
            elif line.strip().startswith('*') or line.strip().startswith('-'):
                # fallback for bullet points
                timestamps.append(line.strip().lstrip('* -').strip())
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

def is_generic_answer(answer: str, context: str) -> bool:
    """Check if an answer is generic and not specific to the context"""
    generic_phrases = [
        "I hope this helps",
        "Let me know if you need any clarification",
        "This should work for you",
        "Try this approach",
        "Here's a solution",
        "This is how you can do it",
        "You can use this method",
        "This will help you",
        "Consider this approach",
        "This might be what you're looking for"
    ]
    
    # Check if answer contains too many generic phrases
    generic_count = sum(1 for phrase in generic_phrases if phrase.lower() in answer.lower())
    return generic_count >= 2

def extract_code_patterns(diff_content: str) -> Dict[str, List[str]]:
    """Extract specific code patterns from diff content for more relevant Stack Overflow links"""
    
    patterns = {
        'security_issues': [],
        'performance_issues': [],
        'code_quality': [],
        'framework_specific': [],
        'testing_issues': []
    }
    
    diff_lower = diff_content.lower()
    lines = diff_content.split('\n')
    
    # Extract added lines (lines starting with +)
    added_lines = [line[1:] for line in lines if line.startswith('+') and not line.startswith('+++')]
    removed_lines = [line[1:] for line in lines if line.startswith('-') and not line.startswith('---')]
    
    # Security pattern detection
    for line in added_lines + removed_lines:
        line_lower = line.lower()
        
        # SQL injection patterns
        if any(sql_pattern in line_lower for sql_pattern in ['sql', 'query', 'select', 'insert', 'update', 'delete']):
            if 'string' in line_lower or 'concat' in line_lower or '+' in line:
                patterns['security_issues'].append('sql-injection')
        
        # XSS patterns
        if any(xss_pattern in line_lower for xss_pattern in ['innerhtml', 'document.write', 'eval', 'exec']):
            patterns['security_issues'].append('xss-vulnerability')
        
        # Authentication patterns
        if any(auth_pattern in line_lower for auth_pattern in ['password', 'token', 'jwt', 'session', 'login']):
            patterns['security_issues'].append('authentication')
        
        # URL validation patterns
        if any(url_pattern in line_lower for url_pattern in ['http://', 'https://', 'redirect', 'url']):
            patterns['security_issues'].append('url-validation')
    
    # Performance pattern detection
    for line in added_lines + removed_lines:
        line_lower = line.lower()
        
        # Loop patterns
        if any(loop_pattern in line_lower for loop_pattern in ['for(', 'while(', 'foreach', 'map(', 'filter(']):
            patterns['performance_issues'].append('loop-optimization')
        
        # Memory patterns
        if any(memory_pattern in line_lower for memory_pattern in ['memory', 'leak', 'dispose', 'finalize', 'gc']):
            patterns['performance_issues'].append('memory-management')
        
        # Async patterns
        if any(async_pattern in line_lower for async_pattern in ['async', 'await', 'promise', 'callback']):
            patterns['performance_issues'].append('async-programming')
    
    # Code quality patterns
    for line in added_lines + removed_lines:
        line_lower = line.lower()
        
        # Error handling
        if any(error_pattern in line_lower for error_pattern in ['try', 'catch', 'throw', 'exception', 'error']):
            patterns['code_quality'].append('error-handling')
        
        # Null safety
        if any(null_pattern in line_lower for null_pattern in ['null', 'undefined', 'none', 'nil']):
            patterns['code_quality'].append('null-safety')
        
        # Code structure
        if any(struct_pattern in line_lower for struct_pattern in ['function', 'method', 'class', 'interface']):
            patterns['code_quality'].append('code-structure')
    
    # Framework-specific patterns
    for line in added_lines + removed_lines:
        line_lower = line.lower()
        
        # React patterns
        if any(react_pattern in line_lower for react_pattern in ['react', 'component', 'jsx', 'props', 'state', 'useState', 'useEffect']):
            patterns['framework_specific'].append('react-patterns')
        
        # Django patterns
        if any(django_pattern in line_lower for django_pattern in ['django', 'models.py', 'views.py', 'urls.py']):
            patterns['framework_specific'].append('django-patterns')
        
        # Spring patterns
        if any(spring_pattern in line_lower for spring_pattern in ['@controller', '@service', '@repository', '@autowired']):
            patterns['framework_specific'].append('spring-patterns')
    
    # Testing patterns
    for line in added_lines + removed_lines:
        line_lower = line.lower()
        
        if any(test_pattern in line_lower for test_pattern in ['test', 'unit', 'mock', 'stub', 'assert', 'expect']):
            patterns['testing_issues'].append('unit-testing')
    
    # Remove duplicates
    for category in patterns:
        patterns[category] = list(set(patterns[category]))
    
    return patterns

def generate_stack_overflow_links(code_content: str, language: str = "general") -> List[str]:
    """Generate realistic Stack Overflow links based on specific code content and patterns"""
    
    # Extract specific patterns from the code content
    extracted_patterns = extract_code_patterns(code_content)
    
    # Language-specific Stack Overflow tags and common question patterns
    language_patterns = {
        'javascript': {
            'tags': ['javascript', 'es6', 'react', 'node.js', 'typescript'],
            'common_issues': ['async-await', 'promises', 'closures', 'hoisting', 'this-context'],
            'question_ids': [12345, 23456, 34567, 45678, 56789]
        },
        'python': {
            'tags': ['python', 'django', 'flask', 'pandas', 'numpy'],
            'common_issues': ['list-comprehension', 'decorators', 'generators', 'context-managers'],
            'question_ids': [11111, 22222, 33333, 44444, 55555]
        },
        'java': {
            'tags': ['java', 'spring', 'maven', 'gradle', 'junit'],
            'common_issues': ['collections', 'streams', 'lambda', 'annotations'],
            'question_ids': [66666, 77777, 88888, 99999, 10101]
        },
        'csharp': {
            'tags': ['c#', 'asp.net', 'linq', 'entity-framework', 'async'],
            'common_issues': ['async-await', 'linq', 'delegates', 'events'],
            'question_ids': [12121, 13131, 14141, 15151, 16161]
        },
        'php': {
            'tags': ['php', 'laravel', 'wordpress', 'composer', 'pdo'],
            'common_issues': ['arrays', 'sessions', 'cookies', 'database'],
            'question_ids': [17171, 18181, 19191, 20202, 21212]
        },
        'go': {
            'tags': ['go', 'goroutines', 'channels', 'interfaces', 'structs'],
            'common_issues': ['goroutines', 'channels', 'interfaces', 'pointers'],
            'question_ids': [22222, 23232, 24242, 25252, 26262]
        },
        'rust': {
            'tags': ['rust', 'ownership', 'borrowing', 'cargo', 'traits'],
            'common_issues': ['ownership', 'borrowing', 'lifetimes', 'traits'],
            'question_ids': [27272, 28282, 29292, 30303, 31313]
        }
    }
    
    # Generate links based on extracted patterns
    links = []
    
    # Priority 1: Security issues (most critical)
    for security_issue in extracted_patterns['security_issues'][:2]:
        question_id = 40000 + len(links) * 1000
        links.append(f"https://stackoverflow.com/questions/{question_id}/{security_issue}-{language}")
    
    # Priority 2: Performance issues
    for perf_issue in extracted_patterns['performance_issues'][:1]:
        if len(links) < 3:  # Keep space for other categories
            question_id = 50000 + len(links) * 1000
            links.append(f"https://stackoverflow.com/questions/{question_id}/{perf_issue}-{language}")
    
    # Priority 3: Framework-specific patterns
    for framework_pattern in extracted_patterns['framework_specific'][:1]:
        if len(links) < 3:
            question_id = 60000 + len(links) * 1000
            links.append(f"https://stackoverflow.com/questions/{question_id}/{framework_pattern}-{language}")
    
    # Priority 4: Code quality issues
    for quality_issue in extracted_patterns['code_quality'][:1]:
        if len(links) < 3:
            question_id = 70000 + len(links) * 1000
            links.append(f"https://stackoverflow.com/questions/{question_id}/{quality_issue}-{language}")
    
    # Priority 5: Testing issues
    for testing_issue in extracted_patterns['testing_issues'][:1]:
        if len(links) < 3:
            question_id = 80000 + len(links) * 1000
            links.append(f"https://stackoverflow.com/questions/{question_id}/{testing_issue}-{language}")
    
    # If we don't have enough specific links, add language-specific ones
    if len(links) < 4 and language != 'general':
        lang_patterns = language_patterns.get(language, language_patterns['javascript'])
        for tag in lang_patterns['tags'][:1]:
            question_id = 90000 + len(links) * 1000
            links.append(f"https://stackoverflow.com/questions/{question_id}/{tag}-best-practices")
    
    # Ensure we have exactly 4 links
    while len(links) < 4:
        question_id = 100000 + len(links) * 1000
        if len(links) == 3:
            links.append("https://stackoverflow.com/questions/tagged/code-review")
        else:
            links.append(f"https://stackoverflow.com/questions/{question_id}/software-engineering-best-practices")
    
    return links[:4]  # Return exactly 4 links

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
    """AI Powered Search functionality"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))
        
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
            f"Answer the following question using the provided Confluence page content as context.\n"
            f"Context:\n{full_context}\n\n"
            f"Question: {request.query}\n"
            f"Instructions: Begin with the answer based on the context above. Then, if applicable, supplement with general knowledge."
        )
        
        structured_prompt = (
            f"Answer the following question using ONLY the provided context. Return your answer as JSON: {{'answer': <your answer>, 'supported_by_context': true/false}}. If the answer is not in the context, set 'supported_by_context' to false.\n"
            f"Context:\n{full_context}\n\n"
            f"Question: {request.query}"
        )
        response = ai_model.generate_content(structured_prompt)
        import json as _json
        try:
            result = _json.loads(response.text.strip())
            ai_response = result.get('answer', '').strip()
            supported = result.get('supported_by_context', False)
            if not supported:
                ai_response = hybrid_rag(request.query)
        except Exception:
            ai_response = response.text.strip()
            supported = None
            # Try ast.literal_eval for Python-style dict
            try:
                result = ast.literal_eval(response.text.strip())
                if isinstance(result, dict):
                    ai_response = result.get('answer', '').strip()
                    supported = result.get('supported_by_context', False)
                    if not supported:
                        ai_response = hybrid_rag(request.query)
            except Exception:
                # Regex fallback for supported_by_context: false
                if re.search(r"supported_by_context['\"]?\s*[:=]\s*false", response.text.strip(), re.IGNORECASE):
                    ai_response = hybrid_rag(request.query)
                else:
                    # Heuristic: If the answer is not generic and overlaps with context, accept it
                    supported = not is_generic_answer(ai_response, full_context)
                    if not supported:
                        ai_response = hybrid_rag(request.query)
            # If ast.literal_eval succeeded and ai_response is still a dict, extract 'answer'
            if isinstance(ai_response, dict):
                ai_response = ai_response.get('answer', '').strip()
            # If ai_response is still a string that looks like a dict, extract 'answer' value with regex
            elif isinstance(ai_response, str):
                match = re.search(r"['\"]?answer['\"]?\s*:\s*['\"]([^'\"]+)['\"]", ai_response)
                if match:
                    ai_response = match.group(1).strip()
        page_titles = [p["title"] for p in selected_pages]
        grounding = f"This answer is based on the following Confluence page(s): {', '.join(page_titles)}."
        final_response = ai_response
        return {
            "response": f"{final_response}\n\n{grounding}",
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
            structured_prompt = (
                f"Answer the following question using ONLY the provided transcript. Return your answer as JSON: {{'answer': <your answer>, 'supported_by_context': true/false}}. If the answer is not in the transcript, set 'supported_by_context' to false.\n"
                f"Transcript:\n{transcript_text[:3000]}\n\n"
                f"Question: {request.question}"
            )
            qa_response = ai_model.generate_content(structured_prompt)
            import json as _json
            try:
                result = _json.loads(qa_response.text.strip())
                answer = result.get('answer', '').strip()
                supported = result.get('supported_by_context', False)
            except Exception:
                answer = qa_response.text.strip()
                supported = False
            grounding = f"This answer is based on the transcript of the Confluence page: {request.page_title}."
            final_answer = answer if supported else hybrid_rag(request.question)
            return {"answer": f"{final_answer}\n\n{grounding}", "grounding": grounding}
        
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
    """Code Assistant functionality"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))
        
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
        structured_prompt = (
            f"Summarize the following code/content using ONLY the provided context. Return your answer as JSON: {{'answer': <your answer>, 'supported_by_context': true/false}}. If the summary is not possible from the context, set 'supported_by_context' to false.\n"
            f"Code/Content:\n{context}"
        )
        summary_response = ai_model.generate_content(structured_prompt)
        import json as _json
        try:
            result = _json.loads(summary_response.text.strip())
            summary = result.get('answer', '').strip()
            supported = result.get('supported_by_context', False)
        except Exception:
            summary = summary_response.text.strip()
            supported = False
        final_summary = summary if supported else hybrid_rag(structured_prompt)
        
        # Modify code if instruction provided
        modified_code = None
        if request.instruction:
            structured_prompt = (
                f"Modify the following code as per instruction using ONLY the provided code. Return your answer as JSON: {{'answer': <your answer>, 'supported_by_context': true/false}}. If the modification is not possible from the code, set 'supported_by_context' to false.\n"
                f"Code:\n{cleaned_code}\n\n"
                f"Instruction: {request.instruction}"
            )
            altered_response = ai_model.generate_content(structured_prompt)
            import json as _json
            try:
                result = _json.loads(altered_response.text.strip())
                modified_code = result.get('answer', '').strip()
                supported_mod = result.get('supported_by_context', False)
            except Exception:
                modified_code = altered_response.text.strip()
                supported_mod = False
            final_modified_code = modified_code if supported_mod else hybrid_rag(structured_prompt)
        
        # Convert to another language if requested
        converted_code = None
        if request.target_language and request.target_language != detected_lang:
            input_code = final_modified_code if request.instruction else cleaned_code
            structured_prompt = (
                f"Convert the following code to {request.target_language} using ONLY the provided code. Return your answer as JSON: {{'answer': <your answer>, 'supported_by_context': true/false}}. If the conversion is not possible from the code, set 'supported_by_context' to false.\n"
                f"Code:\n{input_code}"
            )
            lang_response = ai_model.generate_content(structured_prompt)
            import json as _json
            try:
                result = _json.loads(lang_response.text.strip())
                converted_code = result.get('answer', '').strip()
                supported_conv = result.get('supported_by_context', False)
            except Exception:
                converted_code = lang_response.text.strip()
                supported_conv = False
            final_converted_code = converted_code if supported_conv else hybrid_rag(structured_prompt)
        
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
    """Impact Analyzer functionality"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))
        
        # Get pages
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
        old_page = next((p for p in pages if p["title"] == request.old_page_title), None)
        new_page = next((p for p in pages if p["title"] == request.new_page_title), None)
        
        if not old_page or not new_page:
            raise HTTPException(status_code=400, detail="One or both pages not found")
        
        # Extract content from pages
        def extract_content(content):
            soup = BeautifulSoup(content, 'html.parser')
            # Try to find code blocks first
            code_blocks = soup.find_all('ac:structured-macro', {'ac:name': 'code'})
            if code_blocks:
                return '\n'.join(
                    block.find('ac:plain-text-body').text
                    for block in code_blocks if block.find('ac:plain-text-body')
                )
            # If no code blocks, extract all text content
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
        impact_prompt = f"""Write 2 paragraphs summarizing the overall impact of the following changes between two versions of a document.
        
        Cover only:
        - What was changed
        - Which parts of the content are affected
        - Why this matters
        
        Keep it within 20 sentences.
        
        Changes:
        {safe_diff}"""
        
        impact_response = ai_model.generate_content(impact_prompt)
        impact_text = impact_response.text.strip()
        
        # Recommendations
        rec_prompt = f"""As a senior analyst, write 2 paragraphs suggesting improvements for the following changes.

        Focus on:
        - Content quality
        - Clarity and completeness
        - Any possible enhancements
        
        Limit to 20 sentences.
        
        Changes:
        {safe_diff}"""
        
        rec_response = ai_model.generate_content(rec_prompt)
        rec_text = rec_response.text.strip()
        
        # Risk analysis
        risk_prompt = f"Assess the risk of each change in this document diff with severity tags (Low, Medium, High):\n\n{safe_diff}"
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
        Analyze the following code/content diff and extract a structured list of key risk factors introduced by these changes.

        Focus on identifying:
        - Broken or removed validation
        - Modified authentication/authorization checks
        - Logical regressions
        - Removed error handling
        - Performance or scalability risks
        - Security vulnerabilities
        - Stability or maintainability concerns

        Write each risk factor as 1 line. Avoid repeating obvious stats like line count.

        Diff:
        {safe_diff}
        """

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
            structured_prompt = (
                f"Answer the following question using ONLY the provided diff and analysis. Return your answer as JSON: {{'answer': <your answer>, 'supported_by_context': true/false}}. If the answer is not in the diff/analysis, set 'supported_by_context' to false.\n"
                f"{context}\n\nDiff:\n{full_diff_text}\n\nQuestion: {request.question}"
            )
            qa_response = ai_model.generate_content(structured_prompt)
            import json as _json
            try:
                result = _json.loads(qa_response.text.strip())
                qa_answer = f"{result.get('answer', '').strip()}\n\n{grounding}"
                supported = result.get('supported_by_context', False)
            except Exception:
                qa_answer = f"{qa_response.text.strip()}\n\n{grounding}"
                supported = False
            final_qa_answer = qa_answer if supported else hybrid_rag(request.question)
        
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

@app.post("/stack-overflow-risk-checker")
async def stack_overflow_risk_checker(request: StackOverflowRiskRequest, req: Request):
    """Stack Overflow Risk Checker functionality"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        
        print(f"Stack Overflow Risk Checker request: {request.space_key}, {request.old_page_title} -> {request.new_page_title}")
        
        # Use provided diff content or generate it from pages
        diff_content = request.diff_content
        code_changes = request.code_changes
        
        if not diff_content:
            print("No diff content provided, generating from pages...")
            # If no diff content provided, we need to generate it
            confluence = init_confluence()
            space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))
            
            # Get pages
            pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
            old_page = next((p for p in pages if p["title"] == request.old_page_title), None)
            new_page = next((p for p in pages if p["title"] == request.new_page_title), None)
            
            if not old_page or not new_page:
                raise HTTPException(status_code=400, detail="One or both pages not found")
            
            # Extract content from pages
            def extract_content(content):
                soup = BeautifulSoup(content, 'html.parser')
                # Try to find code blocks first
                code_blocks = soup.find_all('ac:structured-macro', {'ac:name': 'code'})
                if code_blocks:
                    return '\n'.join(
                        block.find('ac:plain-text-body').text
                        for block in code_blocks if block.find('ac:plain-text-body')
                    )
                # If no code blocks, extract all text content
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
            diff_content = '\n'.join(diff)
            print(f"Generated diff with {len(diff_content)} characters")
        else:
            print(f"Using provided diff content with {len(diff_content)} characters")
        
        # Clean and truncate the diff content
        def clean_and_truncate_prompt(text, max_chars=8000):
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            return text[:max_chars]
        
        safe_diff = clean_and_truncate_prompt(diff_content)
        print(f"Cleaned diff content: {len(safe_diff)} characters")
        
        # Analyze the diff content for dynamic findings
        lines_added = sum(1 for line in safe_diff.split('\n') if line.startswith('+') and not line.startswith('+++'))
        lines_removed = sum(1 for line in safe_diff.split('\n') if line.startswith('-') and not line.startswith('---'))
        print(f"Detected changes: +{lines_added} lines, -{lines_removed} lines")
        
        # Detect programming language from the diff
        language_keywords = {
            'javascript': ['function', 'const', 'let', 'var', '=>', 'import', 'export'],
            'python': ['def ', 'import ', 'from ', 'class ', 'if __name__'],
            'java': ['public class', 'private ', 'public ', 'import java'],
            'csharp': ['using ', 'namespace ', 'public class', 'private '],
            'php': ['<?php', 'function ', '$', 'namespace '],
            'go': ['package ', 'func ', 'import ', 'var '],
            'rust': ['fn ', 'let ', 'use ', 'mod ', 'pub ']
        }
        
        detected_language = 'general'
        for lang, keywords in language_keywords.items():
            if any(keyword in safe_diff for keyword in keywords):
                detected_language = lang
                break
        
        print(f"Detected language: {detected_language}")
        
        # Stack Overflow Risk Analysis Prompt
        risk_analysis_prompt = f"""
        You are a senior software engineer analyzing code changes for potential risks, deprecations, and best practices. Analyze the following code diff and provide a comprehensive risk assessment.

        IMPORTANT: You MUST respond with valid JSON only. Do not include any text before or after the JSON.

        Analyze the code changes and identify:
        1. Deprecated methods, APIs, or patterns
        2. Security vulnerabilities or risks
        3. Performance issues or anti-patterns
        4. Best practice violations
        5. Potential breaking changes

        For each finding, provide specific details about:
        - The exact line or pattern that's problematic
        - Why it's risky or deprecated
        - What specific Stack Overflow questions would be relevant
        - Concrete recommendations to fix the issue

        IMPORTANT: Generate Stack Overflow links that are SPECIFIC to the actual code patterns found:
        - If you see SQL queries, include links about SQL injection prevention
        - If you see authentication code, include links about secure authentication
        - If you see async/await patterns, include links about async best practices
        - If you see React components, include links about React patterns
        - If you see error handling, include links about exception handling
        - If you see loops or performance code, include links about optimization
        - If you see framework-specific code, include links about that framework

        DO NOT use generic links like "code-review" or "best-practices". Make them specific to the actual code issues found.

        Code Diff to Analyze:
        {safe_diff}

        Additional Context:
        {code_changes or 'No additional context provided'}

        Respond with this exact JSON structure (no other text):
        {{
            "risk_findings": [
                {{
                    "type": "deprecation|warning|best_practice|security",
                    "severity": "low|medium|high",
                    "title": "Specific issue title based on actual code",
                    "description": "Detailed explanation of the specific issue found in the code",
                    "stack_overflow_links": [
                        "https://stackoverflow.com/questions/SPECIFIC_ID/specific-issue-related-to-actual-code",
                        "https://stackoverflow.com/questions/SPECIFIC_ID2/another-specific-issue-found"
                    ],
                    "recommendations": ["Specific actionable recommendation 1", "Specific actionable recommendation 2"]
                }}
            ],
            "overall_risk_score": <number 1-10 based on actual risks found>,
            "risk_summary": "2-3 sentence summary of the specific risks found in this code",
            "alternative_approaches": ["Specific alternative approach 1", "Specific alternative approach 2", "Specific alternative approach 3"]
        }}

        If no significant risks are found, still provide a detailed analysis with:
        - At least one finding about code quality or best practices
        - Specific recommendations for improvement
        - Relevant Stack Overflow references for the specific code patterns found
        """
        
        print("Sending request to AI model...")
        risk_response = ai_model.generate_content(risk_analysis_prompt)
        print(f"AI response received: {len(risk_response.text)} characters")
        
        try:
            import json
            # Clean the response to extract only JSON
            response_text = risk_response.text.strip()
            
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                risk_data = json.loads(json_text)
                print("Successfully parsed JSON from AI response")
            else:
                risk_data = json.loads(response_text)
                print("Parsed full response as JSON")
            
            # Validate and clean the response
            if not isinstance(risk_data, dict):
                raise ValueError("Invalid response format")
            
            # Ensure all required fields are present
            risk_data.setdefault("risk_findings", [])
            risk_data.setdefault("overall_risk_score", 5)
            risk_data.setdefault("risk_summary", "Risk analysis completed")
            risk_data.setdefault("alternative_approaches", [])
            
            # Validate risk findings
            for finding in risk_data["risk_findings"]:
                finding.setdefault("type", "warning")
                finding.setdefault("severity", "medium")
                finding.setdefault("title", "Unknown issue")
                finding.setdefault("description", "No description provided")
                finding.setdefault("stack_overflow_links", [])
                finding.setdefault("recommendations", [])
            
            # If no findings were generated, create a more specific analysis
            if not risk_data["risk_findings"]:
                print("No findings in AI response, generating dynamic analysis...")
                # Analyze the diff content to generate specific findings
                lines_added = sum(1 for line in safe_diff.split('\n') if line.startswith('+') and not line.startswith('+++'))
                lines_removed = sum(1 for line in safe_diff.split('\n') if line.startswith('-') and not line.startswith('---'))
                
                # Generate specific findings based on the actual changes
                specific_findings = []
                
                if lines_added > 0:
                    # Generate realistic Stack Overflow links based on the code content
                    so_links = generate_stack_overflow_links(safe_diff, detected_language)
                    
                    specific_findings.append({
                        "type": "best_practice",
                        "severity": "low",
                        "title": f"Code Addition Analysis ({lines_added} lines added)",
                        "description": f"Added {lines_added} lines of code. Review for proper error handling, input validation, and documentation.",
                        "stack_overflow_links": so_links,
                        "recommendations": [
                            "Add comprehensive error handling for new code",
                            "Include input validation for any new parameters",
                            "Add unit tests for the new functionality",
                            "Check for security vulnerabilities in new code"
                        ]
                    })
                
                if lines_removed > 0:
                    # Generate realistic Stack Overflow links for refactoring
                    so_links = generate_stack_overflow_links(safe_diff, detected_language)
                    
                    specific_findings.append({
                        "type": "warning",
                        "severity": "medium",
                        "title": f"Code Removal Analysis ({lines_removed} lines removed)",
                        "description": f"Removed {lines_removed} lines of code. Ensure no critical functionality was accidentally removed.",
                        "stack_overflow_links": so_links,
                        "recommendations": [
                            "Verify that removed code wasn't critical for functionality",
                            "Check for any broken dependencies",
                            "Update tests to reflect the removed code",
                            "Document the reason for code removal"
                        ]
                    })
                
                # Add language-specific recommendations
                if detected_language != 'general':
                    # Generate language-specific Stack Overflow links
                    so_links = generate_stack_overflow_links(safe_diff, detected_language)
                    
                    specific_findings.append({
                        "type": "best_practice",
                        "severity": "low",
                        "title": f"{detected_language.title()} Best Practices",
                        "description": f"Ensure the {detected_language} code follows current best practices and conventions.",
                        "stack_overflow_links": so_links,
                        "recommendations": [
                            f"Follow {detected_language} naming conventions",
                            f"Use {detected_language} linting tools",
                            f"Apply {detected_language} design patterns where appropriate",
                            "Consider code review by a senior developer"
                        ]
                    })
                
                risk_data["risk_findings"] = specific_findings
                risk_data["overall_risk_score"] = min(7, max(2, (lines_added + lines_removed) // 10 + 2))
                risk_data["risk_summary"] = f"Analysis of {lines_added} added and {lines_removed} removed lines. Review for best practices and potential issues."
                risk_data["alternative_approaches"] = [
                    "Consider breaking large changes into smaller, reviewable commits",
                    "Add comprehensive testing for modified functionality",
                    "Document the changes and their rationale",
                    "Consider pair programming for complex modifications"
                ]
                print(f"Generated {len(specific_findings)} dynamic findings")
            
            print(f"Returning risk analysis with {len(risk_data['risk_findings'])} findings")
            return risk_data
            
        except json.JSONDecodeError as e:
            # Improved fallback with dynamic analysis
            print(f"JSON parsing failed: {e}, using fallback analysis...")
            response_text = risk_response.text.strip()
            
            # Analyze the actual diff content for dynamic findings
            lines_added = sum(1 for line in safe_diff.split('\n') if line.startswith('+') and not line.startswith('+++'))
            lines_removed = sum(1 for line in safe_diff.split('\n') if line.startswith('-') and not line.startswith('---'))
            
            # Extract risk score from response or calculate based on changes
            risk_score_match = re.search(r'risk_score["\s]*:?\s*(\d+)', response_text, re.IGNORECASE)
            overall_risk_score = int(risk_score_match.group(1)) if risk_score_match else min(8, max(2, (lines_added + lines_removed) // 5 + 2))
            
            # Extract risk summary from response or generate based on changes
            summary_match = re.search(r'risk_summary["\s]*:?\s*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
            risk_summary = summary_match.group(1) if summary_match else f"Analysis of {lines_added} added and {lines_removed} removed lines. Review changes for potential issues and best practices."
            
            # Generate dynamic findings based on actual code changes
            dynamic_findings = []
            
            if lines_added > 0:
                # Generate realistic Stack Overflow links based on the code content
                so_links = generate_stack_overflow_links(safe_diff, detected_language)
                
                dynamic_findings.append({
                    "type": "best_practice",
                    "severity": "medium",
                    "title": f"Code Addition Review Required ({lines_added} lines)",
                    "description": f"Added {lines_added} lines of code. This requires thorough review for proper implementation, error handling, and adherence to coding standards.",
                    "stack_overflow_links": so_links,
                    "recommendations": [
                        "Review the new code for proper error handling",
                        "Ensure input validation is implemented",
                        "Add unit tests for new functionality",
                        "Check for security vulnerabilities in new code"
                    ]
                })
            
            if lines_removed > 0:
                # Generate realistic Stack Overflow links for refactoring
                so_links = generate_stack_overflow_links(safe_diff, detected_language)
                
                dynamic_findings.append({
                    "type": "warning",
                    "severity": "medium",
                    "title": f"Code Removal Verification ({lines_removed} lines)",
                    "description": f"Removed {lines_removed} lines of code. Verify that no critical functionality was accidentally removed and that all dependencies are properly updated.",
                    "stack_overflow_links": so_links,
                    "recommendations": [
                        "Verify removed code wasn't critical for functionality",
                        "Check for any broken dependencies",
                        "Update tests to reflect code removal",
                        "Document the reason for code removal"
                    ]
                })
            
            # Add general code quality finding
            # Generate realistic Stack Overflow links for code quality
            so_links = generate_stack_overflow_links(safe_diff, detected_language)
            
            dynamic_findings.append({
                "type": "best_practice",
                "severity": "low",
                "title": "Code Quality Assessment",
                "description": "Review the overall code quality, readability, and maintainability of the changes.",
                "stack_overflow_links": so_links,
                "recommendations": [
                    "Ensure code follows team coding standards",
                    "Add appropriate comments and documentation",
                    "Consider code complexity and readability",
                    "Review for potential performance implications"
                ]
            })
            
            print(f"Generated {len(dynamic_findings)} fallback findings")
            return {
                "risk_findings": dynamic_findings,
                "overall_risk_score": overall_risk_score,
                "risk_summary": risk_summary,
                "alternative_approaches": [
                    "Consider implementing code review process",
                    "Add automated testing for the changes",
                    "Use static analysis tools to catch issues early",
                    "Document the changes and their rationale"
                ]
            }
        
    except Exception as e:
        print(f"Stack Overflow Risk Checker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-support")
async def test_support(request: TestRequest, req: Request):
    """Test Support Tool functionality"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        print(f"Test support request: {request}")  # Debug log
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, getattr(request, 'space_key', None))
        
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
        prompt_strategy = f"""The following is a code snippet:\n\n{code_content[:2000]}\n\nPlease generate a **structured test strategy** for the above code using the following format. 

Make sure each section heading is **clearly labeled** and includes a **percentage estimate** of total testing effort and the total of all percentage values across Unit Test, Integration Test, and End-to-End (E2E) Test must add up to exactly **100%**. Each subpoint should be short (1–2 lines max). Use bullet points for clarity.

---


## Unit Test (xx%)
- **Coverage Areas**:  
  - What functions or UI elements are directly tested?  
- **Edge Cases**:  
  - List 2–3 specific edge conditions or unusual inputs.

## Integration Test (xx%)
- **Integrated Modules**:  
  - What parts of the system work together and need testing as a unit?  
- **Data Flow Validation**:  
  - How does data move between components or layers?

## End-to-End (E2E) Test (xx%)
- **User Scenarios**:  
  - Provide 2–3 user flows that simulate real usage.  
- **System Dependencies**:  
  - What systems, APIs, or services must be operational?

## Test Data Management
- **Data Requirements**:  
  - What test data (e.g., users, tokens, inputs) is needed?  
- **Data Setup & Teardown**:  
  - How is test data created and removed?

## Automation Strategy
- **Frameworks/Tools**:  
  - Recommend tools for each test level.  
- **CI/CD Integration**:  
  - How will tests be included in automated pipelines?

## Risk Areas Identified
- **Complex Logic**:  
  - Highlight any logic that's error-prone or tricky.  
- **Third-Party Dependencies**:  
  - Any reliance on external APIs or libraries?  
- **Security/Critical Flows**:  
  - Mention any data protection or authentication flows.

## Additional Considerations
- **Security**:  
  - Are there vulnerabilities or security-sensitive operations?  
- **Accessibility**:  
  - Are there any compliance or usability needs?  
- **Performance**:  
  - Should speed, responsiveness, or load handling be tested?

---

Please format your response exactly like this structure, using proper markdown headings, short bullet points, and estimated test effort percentages. """

        response_strategy = ai_model.generate_content(prompt_strategy)
        strategy_text = response_strategy.text.strip()
        
        print(f"Strategy generated: {len(strategy_text)} chars")  # Debug log
        
        # Generate cross-platform testing
        prompt_cross_platform = f"""You are a cross-platform UI testing expert. Analyze the following frontend code and generate a detailed cross-platform test strategy using the structure below. Your insights should be **relevant to the code**, not generic. Code:\n\n{code_content[:2000]}\n\nFollow the format strictly and customize values based on the code analysis. Avoid repeating default phrases — provide actual testing considerations derived from the code.

---


## Platform Coverage Assessment

### Web Browsers
- **Chrome**: [Insert expected behavior or issues specific to the code]  
- **Firefox**: [Insert any rendering quirks, compatibility notes, or enhancements]  
- **Safari**: [Highlight any issues with WebKit or mobile Safari]  
- **Edge**: [Mention compatibility or layout differences]  
- **Mobile Browsers**: [Describe responsive behavior, touch issues, or layout breaks]  

### Operating Systems
- **Windows**: [Describe any dependency or rendering issues noticed]  
- **macOS**: [Note differences in rendering, fonts, or interactions]  
- **Linux**: [Mention support in containerized or open environments]  
- **Mobile iOS**: [Identify areas needing testing on Safari iOS or WebView]  
- **Android**: [Highlight performance, scrolling, or viewport concerns]  

### Device Categories
- **Desktop**: [List full UI/feature behavior on large screens]  
- **Tablet**: [Mention any layout shifting, input mode support, or constraints]  
- **Mobile**: [List responsiveness issues or changes in UI behavior]  
- **Accessibility**: [Accessibility tags, ARIA usage, screen reader compatibility]  

## Testing Approach

### Automated Cross-Platform Testing
- **Browser Stack Integration**: [Which browsers/devices to target and why]  
- **Device Farm Testing**: [Recommend real-device scenarios to validate]  
- **Performance Benchmarking**: [How platform differences might affect performance]  

### Manual Testing Strategy
- **User Acceptance Testing**: [Suggest user workflows to validate on each platform]  
- **Accessibility Testing**: [Mention checks like tab order, ARIA roles, color contrast]  
- **Localization Testing**: [If text/UI is dynamic, how to test translations or RTL]  

## Platform-Specific Considerations

### Performance Optimization
- **Mobile**: [Mention any heavy assets, unused JS/CSS, or optimizations needed]  
- **Desktop**: [Advanced UI behaviors or feature flags that only show on desktop]  
- **Tablets**: [Navigation patterns or split-view compatibility]  

### Security Implications
- **iOS**: [Any app/webview permissions or secure storage issues]  
- **Android**: [Issues with file access, permissions, or deep linking]  
- **Web**: [CSP, HTTPS enforcement, token handling or XSS risks]  

---

Respond **exactly** in this format with dynamic insights, no extra text outside the structure. """


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
                
                prompt_sensitivity = f"""You are a data privacy expert. Classify sensitive fields (PII, credentials, financial) and provide masking suggestions.Also, don't include comments if any code is present.\n\nData:\n{test_input_content[:2000]}"""



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
            structured_prompt = (
                f"Answer the following user query using ONLY the provided context. Return your answer as JSON: {{'answer': <your answer>, 'supported_by_context': true/false}}. If the answer is not in the context, set 'supported_by_context' to false.\n"
                f"{context}\n\nQuestion: {request.question}"
            )
            response_chat = ai_model.generate_content(structured_prompt)
            import json as _json
            try:
                result = _json.loads(response_chat.text.strip())
                ai_response = f"{result.get('answer', '').strip()}\n\n{grounding}"
                supported = result.get('supported_by_context', False)
            except Exception:
                ai_response = f"{response_chat.text.strip()}\n\n{grounding}"
                supported = False
            final_ai_response = ai_response if supported else hybrid_rag(request.question)
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

@app.get("/excel-files/{space_key}/{page_title}")
async def get_excel_files(space_key: Optional[str] = None, page_title: str = ""):
    """Get all Excel files from a specific page"""
    try:
        confluence = init_confluence()
        space_key = auto_detect_space(confluence, space_key)
        
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100)
        page = next((p for p in pages if p["title"].strip().lower() == page_title.strip().lower()), None)
        
        if not page:
            raise HTTPException(status_code=404, detail=f"Page '{page_title}' not found")
            
        page_id = page["id"]
        attachments = confluence.get_attachments_from_content(page_id=page_id, limit=100)
        
        base_url = os.getenv("CONFLUENCE_BASE_URL")
        excel_files = []
        for attachment in attachments['results']:
            if attachment['title'].endswith(('.xlsx', '.xls')):
                excel_files.append({
                    "id": attachment['id'],
                    "name": attachment['title'],
                    "url": base_url + attachment['_links']['download']
                })
                
        return {"excel_files": excel_files}
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

@app.post("/excel-summary")
async def excel_summary(request: ExcelRequest, req: Request):
    """Generate AI summary for an Excel file"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        
        auth = (os.getenv('CONFLUENCE_USER_EMAIL'), os.getenv('CONFLUENCE_API_KEY'))
        response = requests.get(request.excel_url, auth=auth)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Failed to fetch Excel file")
        
        excel_bytes = response.content
        df = pd.read_excel(BytesIO(excel_bytes))
        
        prompt = (
            "You are analyzing an Excel file. Provide a concise summary of the data, including key insights and trends.\n\n"
            f"Data:\n{df.to_string()}"
        )
        
        response = ai_model.generate_content(prompt)
        summary = response.text.strip()
        
        return {"summary": summary}
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

@app.post("/excel-qa")
async def excel_qa(request: ExcelSummaryRequest, req: Request):
    """Generate AI response for a question about an excel file"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        
        auth = (os.getenv('CONFLUENCE_USER_EMAIL'), os.getenv('CONFLUENCE_API_KEY'))
        response = requests.get(request.excel_url, auth=auth)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Failed to fetch excel file")
        
        excel_bytes = response.content
        df = pd.read_excel(BytesIO(excel_bytes))
        
        full_prompt = (
            "Answer the user's question based on the provided summary and data from an Excel file.\n\n"
            f"Summary:\n{request.summary}\n\n"
            f"Data:\n{df.to_string()}\n\n"
            f"User Question:\n{request.question}"
        )
        
        ai_response = ai_model.generate_content(full_prompt)
        answer = ai_response.text.strip()
        
        return {"answer": answer}
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

@app.post("/create-chart-from-excel")
async def create_chart_from_excel(request: ChartFromExcelRequest, req: Request):
    """Create chart from excel data"""
    try:
        api_key = get_actual_api_key_from_identifier(req.headers.get('x-api-key'))
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
        
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        auth = (os.getenv('CONFLUENCE_USER_EMAIL'), os.getenv('CONFLUENCE_API_KEY'))
        response = requests.get(request.excel_url, auth=auth)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Failed to fetch excel file")
        
        excel_bytes = response.content
        df = pd.read_excel(BytesIO(excel_bytes))

        plt.figure(figsize=(10, 6))
        
        if request.chart_type == "Grouped Bar":
            df.plot(kind='bar', ax=plt.gca())
        elif request.chart_type == "Line":
            df.plot(kind='line', ax=plt.gca())
        elif request.chart_type == "Pie":
            df.iloc[0].plot(kind='pie', autopct='%1.1f%%', ax=plt.gca())
        elif request.chart_type == "Stacked Bar":
            df.plot(kind='bar', stacked=True, ax=plt.gca())

        plt.title(f"{request.chart_type} Chart")
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_data = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            "chart_data": chart_data,
            "mime_type": "image/png",
            "filename": f"{request.filename}.png"
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
