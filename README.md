# 🤖 AI Assistant — Gmail, Google Drive & GitHub Agent

A conversational AI assistant that manages your Gmail, Google Drive, and GitHub using natural language. Built with a **dual-LLM architecture**, LangChain, MCP (Model Context Protocol), Groq, and Streamlit.

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
- Get file info and list recently modified files

### 🐙 GitHub
- List, search, and browse repositories
- Read files from any repo (syntax-highlighted code blocks)
- List, read, and create issues
- List branches and create new branches
- List, create, and merge pull requests (merge / squash / rebase)
- Search GitHub repositories
- Create new repositories

### 📋 GitHub Projects (v2)
- List all project boards
- Create new GitHub Projects (linked to a repository)
- View all issues with status, dates, and assignees
- Create issues directly on a project board (added to Backlog)
- Update issues by title — assignee, labels, dates, column — all in one prompt
- Move issues between columns (Backlog → Ready → In Progress → In Review → Done)
- Update custom date fields

### 🤖 Dual-LLM Architecture (Production)
- **Router LLM** (`llama-3.1-8b-instant`, dedicated `GROQ_ROUTER_KEY`):
  - Scope check — rejects off-topic requests instantly, before any tool loads
  - Intent detection — separates greetings from actions
  - Input normalization — fixes spelling, expands abbreviations
  - Section routing — decides which tools (gmail/drive/github) to load
- **Agent LLM** (`llama-3.3-70b-versatile`, `GROQ_API_KEY_2` + `GROQ_API_KEY_3`):
  - Executes tools and streams the response
  - Fallback chain: Key 2 → Key 3 → Mistral → Ollama
- **Key isolation** — Router and Agent use completely separate Groq keys, so they never exhaust each other's quota
- Streaming responses — tokens render live
- Smart fallback — when rate-limited after tool calls, returns results already obtained

### 📄 Supported File Types
- Google Docs, Sheets, Slides
- PDF, DOCX, XLSX, PPTX
- TXT, MD, CSV, HTML, JSON, XML
- Python, JS, TS, YAML, SH, SQL, and more

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
│   ├── router.py           # Router LLM — scope, intent, normalization, section routing
│   ├── orchestrator.py     # Dual-LLM wiring, agent execution, streaming, smart fallback
│   ├── prompt.py           # Agent LLM system prompt — all tool behavior rules
│   ├── tool_executor.py    # MCP tool execution + argument sanitization
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

# KEY_1 → Router LLM only (fast classification, never touches agent quota)
GROQ_ROUTER_KEY=your_groq_key_1

# KEY_2 → Agent LLM primary
GROQ_API_KEY_2=your_groq_key_2

# KEY_3 → Agent LLM fallback
GROQ_API_KEY_3=your_groq_key_3

# Optional — Mistral fallback for agent (recommended)
MISTRAL_API_KEY=your_mistral_key

OLLAMA_BASE_URL=http://localhost:11434

# GitHub OAuth
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

> **Why 3 separate keys?** The Router LLM (`GROQ_ROUTER_KEY`) and Agent LLM (`GROQ_API_KEY_2/3`) use completely separate Groq keys so they can never exhaust each other's daily quota (100K tokens/day per key on free tier).

Get free Groq API keys at [console.groq.com](https://console.groq.com)  
Get Mistral API keys at [console.mistral.ai](https://console.mistral.ai)

### 6. Run
```bash
bash run.sh
```

Or manually:
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
| `GROQ_ROUTER_KEY` | Yes (if Groq) | Dedicated key for Router LLM only |
| `GROQ_API_KEY_2` | Yes (if Groq) | Agent LLM primary key |
| `GROQ_API_KEY_3` | Optional | Agent LLM fallback key |
| `MISTRAL_API_KEY` | Optional | Agent LLM Mistral fallback |
| `OLLAMA_BASE_URL` | Optional | Ollama base URL (default: `http://localhost:11434`) |
| `GITHUB_CLIENT_ID` | For GitHub | GitHub OAuth app Client ID |
| `GITHUB_CLIENT_SECRET` | For GitHub | GitHub OAuth app Client Secret |
| `GROQ_MODEL` | Optional | Override agent model (default: `llama-3.3-70b-versatile`) |

---

## 🤖 Dual-LLM Architecture

```
User Input
    ↓
┌─────────────────────────────────────────────┐
│  Router LLM  (llama-3.1-8b-instant)         │
│  Key: GROQ_ROUTER_KEY (dedicated)           │
│  Speed: ~0.3s                               │
│                                             │
│  ├─ in_scope = false  → return ❌ instantly  │
│  ├─ is_conversational → skip tools, greet   │
│  ├─ cleaned_input     → normalized input    │
│  └─ sections          → [gmail/drive/github]│
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Agent LLM  (llama-3.3-70b-versatile)       │
│  Keys: GROQ_API_KEY_2 → KEY_3 → Mistral    │
│        → Ollama                             │
│                                             │
│  Executes MCP tool calls one at a time      │
│  Streams response token by token            │
└─────────────────────────────────────────────┘
    ↓
drive_server.py / gmail_server.py / github_server.py
    ↓
Google API / GitHub REST + GraphQL API
```

**Why key isolation matters:**
- Groq free tier = 100K tokens/day per key
- Router uses ~200 tokens per call (tiny, 8b model)
- Agent uses ~2,000–5,000 tokens per call (large, 70b model)
- Without isolation: both compete on the same key → quota exhausted faster
- With isolation: Router always has a fresh key, Agent has its own two keys
- If Router key fails → graceful fallback to all-sections (never crashes)

**Agent fallback chain:**
- `GROQ_API_KEY_2` rate limited → try `GROQ_API_KEY_3`
- All Groq keys exhausted → fall to Mistral
- Mistral unavailable → fall to Ollama
- If tools already ran before rate limit → return smart reply from results (no re-execution)

> **Important:** Use `llama-3.3-70b-versatile` only (the default) for the Agent. Do NOT use `llama-3.1-8b-instant` for the Agent — its 6,000 TPM limit is too small for this app's prompt size.

---

## 🐙 GitHub Integration

Connect GitHub by clicking **Connect GitHub** in the side menu (☰). No personal access tokens needed.

**Short repo names work automatically** — just say `mcp-llm` and the tool resolves it to `Absubramani/mcp-llm`.

---

## 📋 GitHub Projects (v2)

Full project board management via GitHub's GraphQL API.

The `update_project_issue_by_title` tool handles assignee + labels + dates + column move in a **single call** — no need to know internal `item_id`.

> **Note:** New GitHub Projects v2 start as a Table view. To get Kanban-style columns, open the project URL and click **+ New view → Board**.

---

## 📅 Scheduled Emails

```
gmail_server.py  →  writes job to scheduled_emails.db
scheduler.py     →  reads jobs every 30s → sends at exact time
```

Jobs survive app restarts — if the scheduler was offline when a job was due, it sends immediately on next startup.

---

## 📝 Logging

Logs written daily to `logs/app_YYYY-MM-DD.log`:
```
2026-04-02 12:02:50 | INFO     | ROUTER   | sections=['github'] cleaned='list my repositories'
2026-04-02 12:02:52 | INFO     | LLM      | Using Groq Key 2
2026-04-02 12:02:58 | INFO     | TOOL     | [OK] list_repos | args={} | 2.1s
2026-04-02 12:03:01 | INFO     | RESPONSE | 11.2s | tools=[list_repos] | '🐙 Your GitHub...'
2026-04-02 12:04:08 | WARNING  | LLM      | Groq Key 2 failed — Rate limited → falling back
2026-04-02 12:04:08 | INFO     | LLM      | Using Groq Key 3
```

---

## 🔒 Security

- Google and GitHub OAuth tokens stored locally per user — never shared between users
- Credentials passed via temp files to MCP servers — deleted after each request
- CSRF protection on GitHub OAuth via state parameter verification
- Off-topic requests rejected by Router LLM before any tool loads — zero wasted compute
- No API keys exposed in UI

---

## 🚀 Ollama Setup (Optional)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
```

Set in `.env`:
```env
LLM_PROVIDER=ollama
```

> Ollama is a last-resort fallback. Complex multi-step GitHub operations may not work reliably with the 8B model.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI + OAuth redirect handling |
| `langchain` | Agent framework |
| `langchain-groq` | Groq LLM (Router + Agent) |
| `langchain-openai` | Mistral fallback |
| `langchain-ollama` | Ollama fallback |
| `mcp` | Model Context Protocol server/client |
| `google-api-python-client` | Gmail + Drive APIs |
| `PyGithub` | GitHub REST API |
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

- **Groq free tier:** 100,000 tokens/day per API key. Use 3 separate keys (Router + 2 Agent) for maximum daily coverage.
- **Router LLM** uses `llama-3.1-8b-instant` which has a 6,000 TPM limit — but since it only classifies (not executes tools), this is rarely hit in practice.
- **GitHub Projects v2:** New projects start as a Table view. Board view must be added manually from the project URL.
- **Ollama fallback:** Complex multi-step GitHub Project operations may not work reliably with the 8B model.
- **File content from GitHub:** Always displayed inside syntax-highlighted code blocks to prevent Markdown rendering issues.