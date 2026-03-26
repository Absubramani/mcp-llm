import os
import json
import threading
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from github import Github, GithubException, Auth

# ================= CONFIG =================

BASE_DIR = Path(__file__).parent.parent
mcp      = FastMCP("GitHub MCP")
_local   = threading.local()


# ================= AUTH =================

def get_github_client() -> Github:
    """Build GitHub client from token."""
    if not hasattr(_local, "github_client") or _local.github_client is None:
        _local.github_client = _build_github_client()
    return _local.github_client


def _build_github_client() -> Github:
    # Try temp creds file first (passed from app via env)
    creds_file = os.environ.get("GITHUB_CREDS_FILE")
    if creds_file and Path(creds_file).exists():
        try:
            with open(creds_file) as f:
                token_data = json.load(f)
            access_token = token_data.get("access_token", "")
            if access_token:
                return Github(auth=Auth.Token(access_token))
        except Exception:
            pass

    # Fallback — PAT from env
    pat = os.environ.get("GITHUB_TOKEN", "")
    if pat:
        return Github(auth=Auth.Token(pat))

    raise RuntimeError("No GitHub credentials found.")


# ================= TOOLS =================

@mcp.tool()
def list_repos(limit: str = "5") -> list:
    """
    List the authenticated user's GitHub repositories.
    limit: number of repos to return (default 5).
    """
    try:
        g     = get_github_client()
        user  = g.get_user()
        repos = list(user.get_repos(sort="updated"))[:int(limit)]
        return [
            {
                "name":     r.full_name,
                "language": r.language or "",
                "stars":    r.stargazers_count,
                "private":  r.private,
                "url":      r.html_url,
            }
            for r in repos
        ]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def create_repo(
    name: str,
    description: str = "",
    private: str = "false",
    auto_init: str = "true",
) -> dict:
    """
    Create a new GitHub repository for the authenticated user.
    name: repository name (no spaces — use hyphens).
    description: short description of the repo (optional).
    private: 'true' to make it private, 'false' for public (default 'false').
    auto_init: 'true' to initialize with a README (default 'true').
    """
    try:
        g    = get_github_client()
        user = g.get_user()

        repo = user.create_repo(
            name        = name,
            description = description or "",
            private     = private.lower() == "true",
            auto_init   = auto_init.lower() == "true",
        )

        return {
            "status":      "success",
            "message":     f"Repository '{repo.full_name}' created successfully.",
            "name":        repo.full_name,
            "url":         repo.html_url,
            "private":     repo.private,
            "description": repo.description or "",
        }

    except GithubException as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def search_repos(query: str, limit: str = "5") -> list:
    """
    Search GitHub repositories by keyword.
    query: search keyword like 'machine learning python'.
    limit: number of results (default 5).
    """
    try:
        g     = get_github_client()
        repos = list(g.search_repositories(query=query))[:int(limit)]
        return [
            {
                "name":        r.full_name,
                "description": r.description or "",
                "stars":       r.stargazers_count,
                "language":    r.language or "",
                "url":         r.html_url,
            }
            for r in repos
        ]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def list_repo_files(repo: str, path: str = "", branch: str = "") -> list:
    """
    List files and folders inside a GitHub repository directory.
    repo: full repo name like 'username/repo-name'.
    path: folder path inside the repo (empty = root directory).
    branch: branch name (optional, defaults to repo default branch).
    Returns files and folders with name, type (file/dir), and size.
    """
    try:
        g = get_github_client()
        r = g.get_repo(repo)

        kwargs = {}
        if branch:
            kwargs["ref"] = branch

        contents = r.get_contents(path or "", **kwargs)

        if not isinstance(contents, list):
            contents = [contents]

        result = []
        for item in contents:
            result.append({
                "name": item.name,
                "path": item.path,
                "type": item.type,   # "file" or "dir"
                "size": item.size if item.type == "file" else None,
            })

        # Sort — folders first, then files alphabetically
        result.sort(key=lambda x: (0 if x["type"] == "dir" else 1, x["name"].lower()))

        if not result:
            return [{"message": f"No files found in '{path or 'root'}' of {repo}."}]

        return result

    except GithubException as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def read_file_from_repo(repo: str, file_path: str, branch: str = "") -> dict:
    """
    Read a file's content from a GitHub repository.
    repo: full repo name like 'username/repo-name'.
    file_path: path to file like 'src/main.py' or 'README.md'.
    branch: branch name (optional, defaults to repo default branch).
    """
    try:
        g      = get_github_client()
        r      = g.get_repo(repo)
        kwargs = {}
        if branch:
            kwargs["ref"] = branch
        content = r.get_contents(file_path, **kwargs)
        text    = content.decoded_content.decode("utf-8", errors="ignore")
        return {
            "file":    file_path,
            "branch":  branch or r.default_branch,
            "content": text[:3000],  # cap at 3000 chars
            "size":    content.size,
        }
    except GithubException as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def list_issues(repo: str, state: str = "open", limit: str = "5") -> list:
    """
    List issues in a repository.
    repo: full repo name like 'username/repo-name'.
    state: 'open', 'closed', or 'all' (default 'open').
    limit: number of issues to return (default 5).
    """
    try:
        g      = get_github_client()
        r      = g.get_repo(repo)
        issues = list(r.get_issues(state=state))[:int(limit)]
        return [
            {
                "number":  i.number,
                "title":   i.title,
                "state":   i.state,
                "author":  i.user.login if i.user else "",
                "created": i.created_at.isoformat() if i.created_at else "",
                "url":     i.html_url,
            }
            for i in issues
            if i.pull_request is None  # exclude PRs from issues list
        ]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def create_issue(repo: str, title: str, body: str = "", labels: str = "") -> dict:
    """
    Create a new issue in a repository.
    repo: full repo name like 'username/repo-name'.
    title: issue title.
    body: issue description (optional).
    labels: comma separated label names (optional).
    """
    try:
        g          = get_github_client()
        r          = g.get_repo(repo)
        label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else []
        issue      = r.create_issue(title=title, body=body, labels=label_list)
        return {
            "status":  "success",
            "message": f"Issue #{issue.number} created successfully.",
            "number":  issue.number,
            "url":     issue.html_url,
        }
    except GithubException as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def read_issue(repo: str, issue_number: str) -> dict:
    """
    Read full details of a specific issue.
    repo: full repo name like 'username/repo-name'.
    issue_number: the issue number.
    """
    try:
        g     = get_github_client()
        r     = g.get_repo(repo)
        issue = r.get_issue(int(issue_number))
        return {
            "number":  issue.number,
            "title":   issue.title,
            "state":   issue.state,
            "author":  issue.user.login if issue.user else "",
            "body":    issue.body or "",
            "created": issue.created_at.isoformat() if issue.created_at else "",
            "url":     issue.html_url,
        }
    except GithubException as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def list_pull_requests(repo: str, state: str = "open", limit: str = "5") -> list:
    """
    List pull requests in a repository.
    repo: full repo name like 'username/repo-name'.
    state: 'open', 'closed', or 'all' (default 'open').
    limit: number of PRs to return (default 5).
    """
    try:
        g   = get_github_client()
        r   = g.get_repo(repo)
        prs = list(r.get_pulls(state=state))[:int(limit)]
        return [
            {
                "number":  pr.number,
                "title":   pr.title,
                "state":   pr.state,
                "author":  pr.user.login if pr.user else "",
                "created": pr.created_at.isoformat() if pr.created_at else "",
                "url":     pr.html_url,
            }
            for pr in prs
        ]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def create_pull_request(
    repo: str,
    title: str,
    head: str,
    base: str = "main",
    body: str = "",
) -> dict:
    """
    Create a new pull request in a repository.
    repo: full repo name like 'username/repo-name'.
    title: pull request title.
    head: the branch containing your changes (source branch).
    base: the branch you want to merge into (default 'main').
    body: pull request description (optional).
    """
    try:
        g    = get_github_client()
        r    = g.get_repo(repo)
        pr   = r.create_pull(
            title = title,
            head  = head,
            base  = base,
            body  = body or "",
        )
        return {
            "status":  "success",
            "message": f"Pull request #{pr.number} created successfully.",
            "number":  pr.number,
            "title":   pr.title,
            "url":     pr.html_url,
            "head":    head,
            "base":    base,
        }
    except GithubException as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run()