# Confluence AI Assistant

A comprehensive AI-powered tool suite for Confluence workspaces, featuring advanced analysis, code assistance, and risk assessment capabilities.

## Features

### 🎯 AI Powered Search
- Semantic search across multiple Confluence pages
- Intelligent content analysis and summarization
- Context-aware responses

### 🎥 Video Summarizer
- Extract key insights from video content
- Generate timestamps and quotes
- Interactive Q&A about video content

### 💻 Code Assistant
- Multi-language code analysis and conversion
- Intelligent code suggestions and improvements
- Support for JavaScript, TypeScript, Python, Java, C#, Go, Rust, PHP

### 📊 Impact Analyzer
- **Version comparison and diff analysis**
- **Risk assessment and impact evaluation**
- **Stack Overflow Risk Checker** - NEW!
  - Scans code changes for deprecated patterns
  - Queries Stack Overflow for warnings and best practices
  - Identifies security vulnerabilities and alternative approaches
  - Provides actionable recommendations with SO references

### 🧪 Test Support Tool
- Comprehensive test strategy generation
- Cross-platform testing recommendations
- Sensitivity analysis for test scenarios

### 🖼️ Image Insights & Chart Builder
- AI-powered image analysis and Q&A
- Excel data processing and visualization
- Dynamic chart generation (bar, line, pie, stacked)

## Stack Overflow Risk Checker

The **Stack Overflow Risk Checker** is a powerful new feature that helps developers identify potential issues in their code changes by simulating Stack Overflow research.

### How it works:
1. **Code Analysis**: Analyzes diff content between two versions
2. **Risk Detection**: Identifies deprecated methods, security issues, and best practice violations
3. **SO Simulation**: Simulates Stack Overflow queries to find relevant warnings
4. **Recommendations**: Provides actionable advice with alternative approaches

### Features:
- **Risk Scoring**: Overall risk assessment (1-10 scale)
- **Categorized Findings**: Deprecation, warnings, best practices, security issues
- **Stack Overflow Links**: Simulated relevant SO references
- **Alternative Approaches**: Suggests better implementation methods
- **Export Support**: Include risk analysis in exported reports

## Installation

### Frontend Setup
   ```bash
cd UI-main
   npm install
npm run dev
   ```

### Backend Setup
   ```bash
cd UI-main/backend
   pip install -r requirements.txt
python main.py
   ```

### Full Stack Development
```bash
cd UI-main
npm run dev:full
```

## Environment Variables

Create a `.env` file in the backend directory:
```
GENAI_API_KEY_1=your_gemini_api_key_1
GENAI_API_KEY_2=your_gemini_api_key_2
CONFLUENCE_URL=your_confluence_url
CONFLUENCE_USERNAME=your_username
CONFLUENCE_API_TOKEN=your_api_token
```

## Usage

1. **Launch the application** and select your Confluence space
2. **Choose a feature** from the navigation bar
3. **Configure settings** specific to each tool
4. **Run analysis** and review results
5. **Export or save** results to Confluence

## API Endpoints

### Core Endpoints
- `POST /search` - AI-powered search
- `POST /video-summarizer` - Video analysis
- `POST /code-assistant` - Code assistance
- `POST /impact-analyzer` - Impact analysis
- `POST /stack-overflow-risk-checker` - **NEW!** Risk assessment
- `POST /test-support` - Test strategy generation
- `POST /image-summary` - Image analysis
- `POST /excel-summary` - Excel processing
- `POST /create-chart` - Chart generation

### Utility Endpoints
- `GET /spaces` - List Confluence spaces
- `GET /pages/{space_key}` - List pages in space
- `POST /export` - Export content in various formats
- `POST /save-to-confluence` - Save results to Confluence

## Technologies Used

- **Frontend**: React, TypeScript, Tailwind CSS, Vite
- **Backend**: FastAPI, Python, Gemini AI
- **APIs**: Confluence API, Stack Overflow API (simulated)
- **Styling**: Tailwind CSS with custom scrollbar support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
