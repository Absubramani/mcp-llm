 # 🤖 AI Assistant — Gmail & Google Drive Agent

A production-level conversational AI assistant that manages your Gmail and Google Drive using natural language. Built with LangChain, MCP (Model Context Protocol), Groq (LLaMA 3.3 70B), and Streamlit.

---

## ✨ Features

### Gmail
- List, read, and summarize emails
- Send emails with CC, BCC support
- Reply and forward emails
- Delete and restore emails
- Search emails by keyword
- Download email attachments
- Save attachments to Google Drive
- Send emails with Drive file attachments

### Google Drive
- List, search, and view files
- Read and summarize all file types
- Create folders and files
- Upload, download, move, copy, rename files
- Delete and restore files/folders
- Share files with specific users
- Get file info

### Supported File Types for Reading/Summarizing
- Google Docs, Sheets, Slides
- PDF, DOCX, XLSX, PPTX
- TXT, MD, CSV, HTML, JSON, XML
- Python, JS, YAML, SH, SQL

### AI Behavior
- Conversational — asks for missing info before acting
- Follow-up context — remembers ids and filenames from chat history
- Spelling correction — understands informal and misspelled input
- Structured responses — bold, emojis, clean formatting
- Multi-user — each user has isolated Google credentials
- Streaming responses — tokens render live like ChatGPT

---

## 🗂️ Project Structure
```
mcp-llm/
├── app.py                  # Streamlit UI + OAuth callback
├── requirements.txt        # Python dependencies
├── oauth-client.json       # Google OAuth credentials (not committed)
├── .env                    # Environment variables (not committed)
│
├── agent/
│   ├── orchestrator.py     # LLM selection, agent execution, streaming
│   ├── prompt.py           # System prompt — all LLM behavior rules
│   ├── tool_executor.py    # MCP tool execution + argument sanitization
│   ├── tool_schema.py      # Fetch tool schemas from MCP servers
│   ├── logger.py           # Structured file logging
│   └── auth.py             # Google OAuth flow + token management
│
├── mcp_servers/
│   ├── drive_server.py     # Google Drive MCP server (15+ tools)
│   └── gmail_server.py     # Gmail MCP server (11+ tools)
│
├── logs/                   # Auto-created — daily log files
└── tokens/                 # Auto-created — per-user OAuth tokens
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/mcp-llm.git
cd mcp-llm
```

### 2. Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Set up Google OAuth
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable **Gmail API** and **Google Drive API**
4. Go to **APIs & Services → Credentials**
5. Create **OAuth 2.0 Client ID** → Web Application
6. Add `http://localhost:8501` as Authorized Redirect URI
7. Download the JSON and save as `oauth-client.json` in project root

### 4. Create `.env` file
```env
LLM_PROVIDER=groq

GROQ_API_KEY_1=your_first_groq_key
GROQ_API_KEY_2=your_second_groq_key
GROQ_API_KEY_3=your_third_groq_key
```

Get free Groq API keys at [console.groq.com](https://console.groq.com)

> Set `LLM_PROVIDER=ollama` to use local Ollama instead of Groq.

### 5. Run
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `groq` or `ollama` |
| `GROQ_API_KEY_1` | If Groq | First Groq API key |
| `GROQ_API_KEY_2` | Optional | Second key — fallback |
| `GROQ_API_KEY_3` | Optional | Third key — fallback |

---

## 🤖 LLM Architecture
```
User Input
    ↓
LangChain Agent (create_openai_tools_agent)
    ↓
Groq Key 1 → Key 2 → Key 3 → Ollama (fallback chain)
    ↓
MCP Tool Call
    ↓
drive_server.py / gmail_server.py
    ↓
Google API
    ↓
Structured Response
```

**Fallback logic:**
- Rate limited → try next Groq key
- All Groq keys exhausted → fall to Ollama
- Keys reset daily

---

## 📝 Logging

Logs are written daily to `logs/app_YYYY-MM-DD.log`:
```
2026-03-11 10:23:01 | INFO     | REQUEST  | 'list my last 3 emails'
2026-03-11 10:23:01 | INFO     | LLM      | Using Groq Key 1
2026-03-11 10:23:03 | INFO     | TOOL     | [OK] list_emails | args={} | 1.23s
2026-03-11 10:23:04 | INFO     | RESPONSE | 3.45s | tools=[list_emails] | 'Here are your latest emails...'
```

---

## 🧪 Testing

Use `test.txt` — covers all features in order:
```
Gmail → Spelling → Conversational → Follow-up → Drive → 
File Types → Upload → Error Handling → Conversation → 
Multi-user → Session Persistence
```

Run tests manually via the chat UI.

---

## 🔒 Security

- OAuth tokens stored locally per user — never shared
- Credentials passed via temp files to MCP servers — deleted after each request
- No API keys exposed in UI
- Path traversal attempts handled by Drive server
- Off-topic requests blocked by prompt rules

---

## 🚀 Ollama Setup (Optional)

For local LLM without Groq:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.1:8b

# Set in .env
LLM_PROVIDER=ollama
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `langchain` | Agent framework |
| `langchain-groq` | Groq LLM integration |
| `langchain-ollama` | Ollama LLM integration |
| `mcp` | Model Context Protocol |
| `google-api-python-client` | Gmail + Drive APIs |
| `pdfplumber` | PDF reading |
| `python-docx` | DOCX reading |
| `openpyxl` | XLSX reading |
| `python-pptx` | PPTX reading |
| `PyJWT` | OAuth token decoding |

---