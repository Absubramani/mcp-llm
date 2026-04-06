# 🤖 AI Assistant — Gmail, Google Drive & GitHub Agent

A conversational AI assistant that manages your Gmail, Google Drive, and GitHub using natural language. Built with LangChain, MCP (Model Context Protocol), Groq (LLaMA 3.3 70B), and Streamlit.

---

## ✨ Features

### 📧 Gmail
- List, read, and summarize emails
- Send emails with CC, BCC support
- Reply and forward emails
- Delete and restore emails
- Search emails by keyword
- Download email attachments
- Save attachments to Google Drive
- Send emails with Drive file attachments
- Schedule emails for future delivery
- List and cancel scheduled emails
- Mark emails as read / unread
- Save, update, send, and delete drafts
- Pagination — load more emails

### 📁 Google Drive
- List, search, and view files
- Read and summarize all file types
- Create folders and files
- Upload, download, move, copy, rename files
- Delete and restore files/folders
- Share files with specific users
- Get file info
- List recently modified files

### 🐙 GitHub
- List, search, and browse repositories
- Read files from any repo (always shown in syntax-highlighted code blocks)
- List, read, and create issues
- List branches and create new branches
- List, create, and **merge** pull requests (merge / squash / rebase)
- Search GitHub repositories
- Create new repositories

### 📋 GitHub Projects (v2)
- List all project boards
- **Create new GitHub Projects** (linked to a repository)
- View all issues on a project board with status, dates, and assignees
- Create issues directly on a project board (added to Backlog automatically)
- Update issues by title — set assignee, labels, start/end dates, move to column — all in one prompt
- Move issues between columns (Backlog → Ready → In Progress → In Review → Done)
- Update custom date fields (start date, end date)

### 🤖 AI Behavior
- Conversational — asks for missing info before acting
- Follow-up context — remembers ids and filenames from chat history
- Spelling correction — understands informal and misspelled input
- Structured responses — bold, emojis, clean formatting
- Multi-user — each user has isolated Google and GitHub credentials
- Streaming responses — tokens render live like ChatGPT
- Smart fallback — when rate-limited, returns results already obtained from completed tool calls

### 📄 Supported File Types for Reading/Summarizing
- Google Docs, Sheets, Slides
- PDF, DOCX, XLSX, PPTX
- TXT, MD, CSV, HTML, JSON, XML
- Python, JS, TS, YAML, SH, SQL, and more
- GitHub repo files (always displayed in syntax-highlighted code blocks)

---

## 🗂️ Project Structure

```
mcp-llm/
├── app.py                  # Streamlit UI + Google & GitHub OAuth callbacks
├── scheduler.py            # Long-running scheduled email delivery process
├── requirements.txt        # Python dependencies
├── run.sh                  # Starts app + scheduler together
├── oauth-client.json       # Google OAuth credentials (not committed)
├── .env                    # Environment variables (not committed)
│
├── agent/
│   ├── orchestrator.py     # LLM selection, agent execution, streaming, smart fallback
│   ├── prompt.py           # System prompt — all LLM behavior rules
│   ├── tool_executor.py    # MCP tool execution + argument sanitization + type coercion
│   ├── tool_schema.py      # Fetch tool schemas from MCP servers
│   ├── logger.py           # Structured file logging
│   └── auth.py             # Google + GitHub OAuth flow + token management
│
├── mcp_servers/
│   ├── drive_server.py     # Google Drive MCP server (15+ tools)
│   ├── gmail_server.py     # Gmail MCP server (20+ tools incl. scheduling + drafts)
│   └── github_server.py    # GitHub MCP server (25+ tools incl. Projects v2)
│
├── logs/                   # Auto-created — daily log files
├── tokens/                 # Auto-created — per-user OAuth tokens
└── scheduled_emails.db     # Auto-created — scheduled email job queue
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Absubramani/mcp-llm.git
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
6. Add `http://localhost:8501` as an Authorized Redirect URI
7. Download the JSON and save as `oauth-client.json` in the project root

### 4. Set up GitHub OAuth
1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **New OAuth App**
3. Set **Homepage URL** to `http://localhost:8501`
4. Set **Authorization Callback URL** to `http://localhost:8501`
5. Copy the **Client ID** and **Client Secret**

### 5. Create `.env` file
```env
LLM_PROVIDER=groq

GROQ_API_KEY_1=your_first_groq_key
GROQ_API_KEY_2=your_second_groq_key
GROQ_API_KEY_3=your_third_groq_key

# Optional — Mistral fallback (recommended)
MISTRAL_API_KEY=your_mistral_key

# GitHub OAuth
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

Get free Groq API keys at [console.groq.com](https://console.groq.com)  
Get Mistral API keys at [console.mistral.ai](https://console.mistral.ai)

> Set `LLM_PROVIDER=ollama` to use local Ollama instead of Groq.

### 6. Run
```bash
# Recommended — starts both app and scheduler together
bash run.sh
```

Or manually in two terminals:
```bash
# Terminal 1 — scheduler (required for scheduled emails)
python scheduler.py

# Terminal 2 — app
streamlit run app.py
```

> ⚠️ If you only run `streamlit run app.py` without `scheduler.py`, the app works but scheduled emails will never be delivered.

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `groq` or `ollama` |
| `GROQ_API_KEY_1` | If Groq | First Groq API key |
| `GROQ_API_KEY_2` | Optional | Second key — fallback |
| `GROQ_API_KEY_3` | Optional | Third key — fallback |
| `MISTRAL_API_KEY` | Optional | Mistral fallback key (recommended) |
| `GITHUB_CLIENT_ID` | For GitHub | GitHub OAuth app Client ID |
| `GITHUB_CLIENT_SECRET` | For GitHub | GitHub OAuth app Client Secret |
| `GROQ_MODEL` | Optional | Override model (default: `llama-3.3-70b-versatile`) |

---

## 🤖 LLM Architecture

```
User Input
    ↓
LangChain Agent (create_openai_tools_agent)
    ↓
Groq Key 1 → Key 2 → Key 3 → Mistral → Ollama  (fallback chain)
    ↓
MCP Tool Call  (one at a time)
    ↓
drive_server.py / gmail_server.py / github_server.py
    ↓
Google API / GitHub REST + GraphQL API
    ↓
Structured Response (streamed token-by-token)
```

**Fallback logic:**
- Rate limited or quota exceeded → try next Groq key immediately
- All Groq keys exhausted → fall to Mistral
- Mistral unavailable → fall to Ollama
- If tools already ran before rate limit hit → return smart formatted reply from tool results (no re-execution)
- Keys reset daily (100K tokens/day per key on free tier)

> **Important:** Use `llama-3.3-70b-versatile` only (the default). Do NOT use `llama-3.1-8b-instant` — its 6,000 TPM limit is too small for this app's prompt size.

---

## 🐙 GitHub Integration

Connect GitHub by clicking **Connect GitHub** in the side menu (☰). This triggers an OAuth flow — no personal access tokens needed.

**Short repo names work automatically** — just say `mcp-llm` and the tool resolves it to `Absubramani/mcp-llm` without asking.

**Example prompts:**
```
list my github repos
read requirements.txt from mcp-llm
list issues in test-ai-assistant
create a bug issue in test-ai-assistant titled "Login breaks on mobile" labels: bug
create a pull request in test-ai-assistant
merge PR #5 in test-ai-assistant using squash
```

---

## 📋 GitHub Projects (v2)

Full project board management via GitHub's GraphQL API.

**Create a project:**
```
create a project called Sprint Board linked to Absubramani/test-ai-assistant
```

**Manage issues on the board:**
```
show issues on my test project board
create an issue 'Implement OAuth' in test-ai-assistant add to test project
assign me to that issue, label: feature, start today, end Friday, move to Ready
move 'Implement OAuth' to Done
```

The `update_project_issue_by_title` tool handles assignee + labels + dates + column move in a **single call** — no need to know internal `item_id`.

> **Note:** New GitHub Projects v2 start as a Table view. To get Kanban-style columns (Backlog, Ready, In Progress, etc.), open the project URL and click **+ New view → Board**.

---

## 📅 Scheduled Emails

The scheduler runs as a companion process. It checks the job queue every 30 seconds and sends emails at the exact scheduled time.

```
gmail_server.py  →  writes job to scheduled_emails.db
scheduler.py     →  reads jobs every 30s → sends at exact time
```

Jobs survive app restarts — if the scheduler was offline when a job was due, it sends immediately on next startup.

**Example usage:**
```
schedule email to someone@gmail.com at tomorrow 9am
schedule email to someone@gmail.com at Friday 3pm
show my scheduled emails
cancel my scheduled email
```

---

## 📝 Logging

Logs are written daily to `logs/app_YYYY-MM-DD.log`:
```
2026-04-02 12:02:52 | INFO     | LLM      | Using Groq Key 1
2026-04-02 12:02:58 | INFO     | TOOL     | [OK] create_project_issue | args={...} | 4.79s
2026-04-02 12:03:41 | INFO     | RESPONSE | 50.57s | tools=[create_project_issue] | '✅ Issue #14...'
2026-04-02 12:04:08 | WARNING  | LLM      | Groq Key 1 failed — Rate limited → falling back
2026-04-02 12:04:08 | INFO     | LLM      | Using Groq Key 2
```

---

## 🧪 Testing

Use `test.txt` — covers all features in order:

```
Gmail → Schedule Email → Spelling → Conversational → Follow-up →
Drive → File Types → Upload → Error Handling → Conversation →
GitHub Repos → File Reading → Issues → Pull Requests → PR Merge →
Create Repo → GitHub Projects → Create Project → Board Management →
Update Issues → Natural Language Tests
```

Run all tests manually via the chat UI.

---

## 🔒 Security

- Google OAuth tokens stored locally per user — never shared between users
- GitHub OAuth tokens stored locally per user — never shared between users
- Credentials passed via temp files to MCP servers — deleted after each request
- No API keys exposed in UI
- Path traversal attempts handled by Drive server
- Off-topic requests blocked by prompt scope rules
- CSRF protection on GitHub OAuth via state parameter verification

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

> Ollama is a last-resort fallback. It handles simple requests but may struggle with complex multi-step GitHub operations due to smaller context window.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI + OAuth redirect handling |
| `langchain` | Agent framework |
| `langchain-groq` | Groq LLM integration |
| `langchain-openai` | Mistral LLM integration |
| `langchain-ollama` | Ollama LLM integration |
| `mcp` | Model Context Protocol server/client |
| `google-api-python-client` | Gmail + Drive APIs |
| `PyGithub` | GitHub REST API (repos, issues, PRs, branches) |
| `requests` | GitHub GraphQL API (Projects v2) |
| `apscheduler` | Scheduled email delivery |
| `dateparser` | Natural language time parsing |
| `pdfplumber` | PDF reading |
| `python-docx` | DOCX reading |
| `openpyxl` | XLSX reading |
| `python-pptx` | PPTX reading |
| `PyJWT` | OAuth token decoding |

---

## 💬 Example Prompts

### Gmail
```
list my last 5 emails
summarize the email from Glassdoor
forward that to someone@gmail.com
schedule email to team@company.com at Monday 9am
save draft to boss@company.com subject "Q2 Report"
mark all unread emails as read
```

### Google Drive
```
summarize AI_ML Milestones.docx
send that file to colleague@gmail.com
create a folder called Q2 Reports
upload this file to Projects folder
list recent files
```

### GitHub
```
list my repos
read requirements.txt from mcp-llm
list open issues in test-ai-assistant
create a bug issue in test-ai-assistant titled "Login breaks on mobile" labels: bug
create a pull request in test-ai-assistant
merge PR #5 in test-ai-assistant using squash
create a public repo called my-new-project with readme
```

### GitHub Projects
```
create a project called Sprint 1 linked to test-ai-assistant
show issues on my test project board
create an issue 'Implement OAuth' in test-ai-assistant add to test project
assign me to that issue, label: feature, start today, end Friday, move to Ready
move 'Implement OAuth' to Done
update end date for 'Implement OAuth' to next Monday
```

---

## 🛠️ Known Limitations

- **Groq free tier:** 100,000 tokens/day per API key. Heavy usage exhausts the daily quota. Use 3 keys for maximum coverage throughout the day.
- **GitHub Projects v2:** New projects start as a Table view. Board view (Kanban columns) must be added manually from the project URL by clicking **+ New view → Board**.
- **Ollama fallback:** Complex multi-step GitHub Project operations may not work reliably with the 8B model due to context limitations.
- **File content from GitHub:** Always displayed inside syntax-highlighted code blocks to prevent Markdown rendering issues (e.g. `#` comment lines rendering as giant headers).

---