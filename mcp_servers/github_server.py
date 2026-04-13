import os
import json
import threading
import requests
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from github import Github, GithubException, Auth

# ================= CONFIG =================

BASE_DIR = Path(__file__).parent.parent
mcp      = FastMCP("GitHub MCP")
_local   = threading.local()


# ================= AUTH =================

def get_github_client() -> Github:
    if not hasattr(_local, "github_client") or _local.github_client is None:
        _local.github_client = _build_github_client()
    return _local.github_client


def _get_access_token() -> str:
    creds_file = os.environ.get("GITHUB_CREDS_FILE")
    if creds_file and Path(creds_file).exists():
        try:
            with open(creds_file) as f:
                token_data = json.load(f)
            token = token_data.get("access_token", "")
            if token:
                return token
        except Exception:
            pass
    raise RuntimeError("No GitHub credentials found.")


def _build_github_client() -> Github:
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
    raise RuntimeError("No GitHub credentials found.")


# ================= HELPERS =================

def _normalize_repo(repo: str) -> str:
    """Fix common spelling errors in repo names."""
    if not repo:
        return repo
    repo = repo.strip()
    repo = repo.replace("Abssubramani/", "Absubramani/")
    repo = repo.replace("abssubramani/", "Absubramani/")
    return repo


def _resolve_repo(repo: str) -> str:
    """
    Resolve repo to full owner/repo format.
    If short name given (no slash), searches user repos by name.
    """
    if not repo:
        return ""
    repo = _normalize_repo(repo)
    if "/" not in repo:
        try:
            g    = get_github_client()
            user = g.get_user()
            for r in user.get_repos(sort="updated"):
                if r.name.lower() == repo.lower():
                    return r.full_name
        except Exception:
            pass
    return repo


def _resolve_project_id(id_or_title: str) -> str:
    """
    If input doesn't look like a node ID (PVT_... or PN_...), find it
    by searching the user's projects list by title.
    """
    if not id_or_title:
        return id_or_title
    s = id_or_title.strip()
    if s.startswith("PVT_") or s.startswith("PN_"):
        return s
    try:
        projects = list_projects()
        for p in projects:
            if isinstance(p, dict) and "title" in p:
                if p["title"].lower() == s.lower():
                    return p.get("id", s)
    except Exception:
        pass
    return s


def _safe_int(val, default: int = 10) -> int:
    """Safely convert string or int to int."""
    try:
        return int(val) if val else default
    except (ValueError, TypeError):
        return default


def _graphql(query: str, variables: dict = None) -> dict:
    """Execute a GitHub GraphQL query."""
    token = _get_access_token()
    resp = requests.post(
        "https://api.github.com/graphql",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        json={"query": query, "variables": variables or {}},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data.get("data", {})


# ================= TOOLS =================

@mcp.tool()
def list_repos(limit: str = "10") -> list:
    """
    List the authenticated user's GitHub repositories.
    limit: number of repos to return (default 10).
    """
    try:
        count = _safe_int(limit, 10)
        g     = get_github_client()
        user  = g.get_user()
        repos = list(user.get_repos(sort="updated"))[:count]
        if not repos:
            return [{"message": "No repositories found."}]
        return [
            {
                "name":        r.full_name,
                "language":    r.language or "",
                "stars":       r.stargazers_count,
                "private":     r.private,
                "url":         r.html_url,
                "description": r.description or "",
            }
            for r in repos
        ]
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
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
    description: short description (optional).
    private: 'true' for private, 'false' for public (default 'false').
    auto_init: 'true' to initialize with README (default 'true').
    """
    try:
        if not name or not name.strip():
            return {"status": "error", "message": "Repository name is required."}
        name = name.strip().replace(" ", "-")
        g    = get_github_client()
        user = g.get_user()
        repo = user.create_repo(
            name        = name,
            description = description or "",
            private     = str(private).lower() == "true",
            auto_init   = str(auto_init).lower() == "true",
        )
        return {
            "status":      "success",
            "message":     f"Repository '{repo.full_name}' created successfully.",
            "name":        repo.full_name,
            "url":         repo.html_url,
            "private":     repo.private,
            "auto_init":   str(auto_init).lower() == "true",
            "description": repo.description or "",
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def search_repos(query: str, limit: str = "5") -> list:
    """
    Search GitHub repositories by keyword.
    query: search keyword like 'machine learning python'.
    limit: number of results (default 5).
    """
    try:
        count = _safe_int(limit, 5)
        g     = get_github_client()
        repos = list(g.search_repositories(query=query))[:count]
        if not repos:
            return [{"message": f"No repositories found for '{query}'."}]
        return [
            {
                "name":        r.full_name,
                "stars":       r.stargazers_count,
                "language":    r.language or "",
                "url":         r.html_url,
                "description": r.description or "",
            }
            for r in repos
        ]
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def list_repo_files(repo: str, path: str = "", branch: str = "") -> list:
    """
    List files and directories in a GitHub repository.
    repo: full repo name like 'username/repo-name' OR short name (auto-resolved).
    path: internal path (empty for root). branch: optional.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return [{"status": "error", "message": "Please provide the repository name."}]
        g        = get_github_client()
        r        = g.get_repo(repo)
        kwargs   = {}
        if branch and branch.strip():
            kwargs["ref"] = branch.strip()
        contents = r.get_contents(path or "", **kwargs)
        if not isinstance(contents, list):
            contents = [contents]
        result = []
        for item in contents:
            result.append({
                "name": item.name,
                "path": item.path,
                "type": item.type,
                "size": item.size if item.type == "file" else None,
            })
        result.sort(key=lambda x: (0 if x["type"] == "dir" else 1, x["name"].lower()))
        if not result:
            return [{"message": f"No files found in '{path or 'root'}' of {repo}."}]
        return result
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def read_file_from_repo(repo: str, file_path: str, branch: str = "") -> dict:
    """
    Read the content of a file from a GitHub repository.
    repo: full repo name like 'username/repo-name' OR short name.
    file_path: path to the file e.g. 'README.md', 'agent/orchestrator.py'.
    branch: optional, defaults to default branch.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        if not file_path or not file_path.strip():
            return {"status": "error", "message": "Please provide the file path."}
        g      = get_github_client()
        r      = g.get_repo(repo)
        kwargs = {}
        if branch and branch.strip():
            kwargs["ref"] = branch.strip()
        content = r.get_contents(file_path.strip(), **kwargs)
        text    = content.decoded_content.decode("utf-8", errors="ignore")
        return {
            "file":    file_path.strip(),
            "repo":    repo,
            "branch":  branch or r.default_branch,
            "content": text[:5000],
            "size":    content.size,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def list_branches(repo: str) -> list:
    """
    List all branches in a repository.
    repo: full repo name like 'username/repo-name' OR short name.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return [{"status": "error", "message": "Please provide the repository name."}]
        g        = get_github_client()
        r        = g.get_repo(repo)
        branches = list(r.get_branches())
        if not branches:
            return [{"message": f"No branches found in {repo}."}]
        return [
            {"name": b.name, "default": b.name == r.default_branch}
            for b in branches
        ]
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def list_issues(repo: str, state: str = "open", limit: str = "20") -> list:
    """
    List issues in a repository.
    repo: full repo name OR short name. state: open/closed/all. limit: default 20.
    """
    try:
        repo  = _resolve_repo(repo)
        if not repo:
            return [{"status": "error", "message": "Please provide the repository name."}]
        state = (state or "open").strip().lower()
        if state not in ("open", "closed", "all"):
            state = "open"
        g      = get_github_client()
        r      = g.get_repo(repo)
        issues = list(r.get_issues(state=state))[:_safe_int(limit, 20)]
        if not issues:
            return [{"message": f"No {state} issues found in {repo}."}]
        return [
            {
                "number":     i.number,
                "title":      i.title,
                "state":      i.state,
                "author":     i.user.login if i.user else "",
                "created_at": i.created_at.isoformat(),
                "url":        i.html_url,
            }
            for i in issues
        ]
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except GithubException as e:
        return [{"status": "error", "message": f"GitHub Error: {str(e)}"}]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def create_issue(
    repo: str,
    title: str,
    body: str = "",
    labels: str = "",
    assignee: str = "",
    start_date: str = "",
    end_date: str = "",
) -> dict:
    """
    Create a new issue in a repository.
    repo: full repo name OR short name. title: required.
    body: optional. labels: comma-separated. assignee: GitHub username.
    start_date/end_date: YYYY-MM-DD format.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        if not title or not title.strip():
            return {"status": "error", "message": "Please provide an issue title."}
        g          = get_github_client()
        r          = g.get_repo(repo)
        label_list = [l.strip() for l in (labels or "").split(",") if l.strip()]
        assignees  = [assignee.strip()] if assignee and assignee.strip() else []
        full_body  = (body or "").rstrip()
        if start_date or end_date:
            if full_body:
                full_body += "\n\n"
            if start_date:
                full_body += f"**Start Date:** {start_date}\n"
            if end_date:
                full_body += f"**End Date:** {end_date}\n"
        create_kwargs = {
            "title":  title.strip(),
            "body":   full_body,
            "labels": label_list,
        }
        if assignees:
            create_kwargs["assignees"] = assignees
        issue = r.create_issue(**create_kwargs)
        return {
            "status":     "success",
            "message":    f"Issue #{issue.number} created successfully.",
            "number":     issue.number,
            "url":        issue.html_url,
            "repo":       repo,
            "assignee":   assignee or "",
            "start_date": start_date or "",
            "end_date":   end_date or "",
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def read_issue(repo: str, issue_number: str) -> dict:
    """
    Read full details of a specific issue.
    repo: full repo name OR short name. issue_number: the number.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        num = _safe_int(issue_number, 0)
        if num == 0:
            return {"status": "error", "message": "Please provide a valid issue number."}
        g     = get_github_client()
        r     = g.get_repo(repo)
        issue = r.get_issue(num)
        return {
            "number":  issue.number,
            "title":   issue.title,
            "state":   issue.state,
            "author":  issue.user.login if issue.user else "",
            "body":    issue.body or "",
            "created": issue.created_at.isoformat() if issue.created_at else "",
            "url":     issue.html_url,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
def add_issue_comment(repo: str, issue_number: str, body: str) -> dict:
    """
    Add a comment to an issue or pull request.
    repo: full repo name OR short name.
    issue_number: the issue or pull request number.
    body: the comment text (markdown supported).
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        num = _safe_int(issue_number, 0)
        if num == 0:
            return {"status": "error", "message": "Please provide a valid issue number."}
        if not body or not body.strip():
            return {"status": "error", "message": "Please provide the comment body."}
        
        g     = get_github_client()
        r     = g.get_repo(repo)
        issue = r.get_issue(num)
        comment = issue.create_comment(body.strip())
        
        return {
            "status":  "success",
            "message": f"Comment added to issue #{num} successfully.",
            "url":     comment.html_url,
            "id":      comment.id,
            "repo":    repo,
            "number":  num,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def list_pull_requests(repo: str, state: str = "open", limit: str = "5") -> list:
    """
    List pull requests in a repository.
    repo: full repo name OR short name. state: open/closed/all. limit: default 5.
    """
    try:
        repo  = _resolve_repo(repo)
        if not repo:
            return [{"status": "error", "message": "Please provide the repository name."}]
        state = (state or "open").strip().lower()
        if state not in ("open", "closed", "all"):
            state = "open"
        g   = get_github_client()
        r   = g.get_repo(repo)
        prs = list(r.get_pulls(state=state))[:_safe_int(limit, 5)]
        if not prs:
            return [{"message": f"No {state} pull requests found in {repo}.", "empty": True}]
        return [
            {
                "number":    pr.number,
                "title":     pr.title,
                "state":     pr.state,
                "author":    pr.user.login if pr.user else "",
                "head":      pr.head.ref,
                "base":      pr.base.ref,
                "created":   pr.created_at.isoformat() if pr.created_at else "",
                "url":       pr.html_url,
                "mergeable": pr.mergeable,
            }
            for pr in prs
        ]
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
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
    repo: full repo name OR short name. title: required.
    head: source branch with changes (must exist). base: target branch (default main).
    body: optional description.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        if not title or not title.strip():
            return {"status": "error", "message": "Please provide a PR title."}
        if not head or not head.strip():
            return {"status": "error", "message": "Please provide the head branch."}
        g = get_github_client()
        r = g.get_repo(repo)
        try:
            r.get_branch(head.strip())
        except GithubException:
            branches = [b.name for b in list(r.get_branches())[:10]]
            return {
                "status":  "error",
                "message": f"Branch '{head}' does not exist in {repo}. Available: {', '.join(branches)}",
            }
        pr = r.create_pull(
            title = title.strip(),
            head  = head.strip(),
            base  = (base or "main").strip(),
            body  = (body or ""),
        )
        return {
            "status":  "success",
            "message": f"Pull request #{pr.number} created successfully.",
            "number":  pr.number,
            "title":   pr.title,
            "url":     pr.html_url,
            "head":    head.strip(),
            "base":    (base or "main").strip(),
            "repo":    repo,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def merge_pull_request(
    repo: str,
    pull_number: str,
    merge_method: str = "merge",
    commit_title: str = "",
    commit_message: str = "",
) -> dict:
    """
    Merge an open pull request.
    repo: full repo name OR short name. pull_number: PR number.
    merge_method: 'merge' (default), 'squash' (combine commits), 'rebase' (linear history).
    commit_title: optional custom commit title.
    commit_message: optional commit message body.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        num = _safe_int(pull_number, 0)
        if num == 0:
            return {"status": "error", "message": "Please provide a valid PR number."}
        method = (merge_method or "merge").strip().lower()
        if method not in ("merge", "squash", "rebase"):
            method = "merge"
        g  = get_github_client()
        r  = g.get_repo(repo)
        pr = r.get_pull(num)
        if pr.state != "open":
            return {"status": "error", "message": f"PR #{num} is already {pr.state} — cannot merge."}
        if pr.mergeable is False:
            return {"status": "error",
                    "message": f"PR #{num} has merge conflicts. Please resolve them on GitHub first."}
        kwargs = {"merge_method": method}
        if commit_title and commit_title.strip():
            kwargs["commit_title"] = commit_title.strip()
        if commit_message and commit_message.strip():
            kwargs["commit_message"] = commit_message.strip()
        result = pr.merge(**kwargs)
        if result.merged:
            return {
                "status":   "success",
                "message":  f"PR #{num} merged successfully via {method}.",
                "merged":   True,
                "sha":      result.sha,
                "method":   method,
                "title":    pr.title,
                "head":     pr.head.ref,
                "base":     pr.base.ref,
                "issue_num": num,
            }
        return {"status": "error", "message": result.message or "Merge failed."}
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def create_branch(repo: str, branch_name: str, base_branch: str = "main") -> dict:
    """
    Create a new branch in a repository.
    repo: full repo name OR short name. branch_name: new branch.
    base_branch: branch to fork from (default main).
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        if not branch_name or not branch_name.strip():
            return {"status": "error", "message": "Please provide a branch name."}
        g = get_github_client()
        r = g.get_repo(repo)
        try:
            source = r.get_branch((base_branch or "main").strip())
        except GithubException:
            return {"status": "error", "message": f"Base branch '{base_branch}' does not exist in {repo}."}
        try:
            r.create_git_ref(ref=f"refs/heads/{branch_name.strip()}", sha=source.commit.sha)
            return {
                "status":  "success",
                "message": f"Branch '{branch_name}' created successfully from '{base_branch}'.",
                "branch":  branch_name.strip(),
                "repo":    repo,
            }
        except GithubException as e:
            if "Reference already exists" in str(e) or getattr(e, "status", 0) == 422:
                return {"status": "error", "message": f"Branch '{branch_name}' already exists."}
            return {"status": "error", "message": str(e)}
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# GITHUB PROJECTS v2 — GraphQL-based tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_projects(repo: str = "", limit: str = "10") -> list:
    """
    List GitHub Projects (v2) for the authenticated user or a specific repo.
    repo: optional 'owner/repo-name'. limit: default 10.
    If repo is empty, lists all projects owned by the authenticated user.
    """
    try:
        count    = _safe_int(limit, 10)
        projects = []

        if repo and repo.strip():
            repo  = _resolve_repo(repo.strip())
            owner, repo_name = repo.split("/", 1)
            query = """
            query($owner: String!, $repo: String!, $limit: Int!) {
              repository(owner: $owner, name: $repo) {
                projectsV2(first: $limit) {
                  nodes {
                    id number title url closed
                    owner { ... on User { login } ... on Organization { login } }
                    repositories(first: 5) { nodes { nameWithOwner } }
                  }
                }
              }
            }
            """
            data     = _graphql(query, {"owner": owner, "repo": repo_name, "limit": count})
            projects = data.get("repository", {}).get("projectsV2", {}).get("nodes", [])
        else:
            query = """
            query($limit: Int!) {
              viewer {
                projectsV2(first: $limit) {
                  nodes {
                    id number title url closed
                    owner { ... on User { login } ... on Organization { login } }
                    repositories(first: 5) { nodes { nameWithOwner } }
                  }
                }
                recentProjects(first: $limit) {
                  nodes {
                    id number title url closed
                    owner { ... on User { login } ... on Organization { login } }
                    repositories(first: 5) { nodes { nameWithOwner } }
                  }
                }
                issues(first: 20, states: OPEN) {
                  nodes {
                    projectItems(first: 10) {
                      nodes {
                        project {
                          id number title url closed
                          owner { ... on User { login } ... on Organization { login } }
                          repositories(first: 5) { nodes { nameWithOwner } }
                        }
                      }
                    }
                  }
                }
                pullRequests(first: 20, states: OPEN) {
                  nodes {
                    projectItems(first: 10) {
                      nodes {
                        project {
                          id number title url closed
                          owner { ... on User { login } ... on Organization { login } }
                          repositories(first: 5) { nodes { nameWithOwner } }
                        }
                      }
                    }
                  }
                }
              }
            }
            """
            data   = _graphql(query, {"limit": count})
            viewer = data.get("viewer", {})
            v2     = viewer.get("projectsV2", {}).get("nodes", []) or []
            recent = viewer.get("recentProjects", {}).get("nodes", []) or []
            issue_projects = []
            for issue in viewer.get("issues", {}).get("nodes", []):
                for item in issue.get("projectItems", {}).get("nodes", []):
                    if item.get("project"):
                        issue_projects.append(item["project"])
            pr_projects = []
            for pr in viewer.get("pullRequests", {}).get("nodes", []):
                for item in pr.get("projectItems", {}).get("nodes", []):
                    if item.get("project"):
                        pr_projects.append(item["project"])
            all_source = v2 + recent + issue_projects + pr_projects
            merged     = {p["id"]: p for p in all_source if p and "id" in p}
            projects   = list(merged.values())

        if not projects:
            return [{"message": "No projects found."}]

        results = []
        for p in projects:
            if p.get("closed"):
                continue
            owner_login  = p.get("owner", {}).get("login", "") if p.get("owner") else ""
            linked_repos = [r["nameWithOwner"] for r in
                            p.get("repositories", {}).get("nodes", []) if r.get("nameWithOwner")]
            results.append({
                "id":     p["id"],
                "number": p["number"],
                "title":  p["title"],
                "url":    p["url"],
                "closed": p["closed"],
                "owner":  owner_login,
                "repos":  ", ".join(linked_repos) if linked_repos else "",
            })
        return results if results else [{"message": "No open projects found."}]

    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def create_project(title: str, repo: str = "") -> dict:
    """
    Create a new GitHub Project (v2) for the authenticated user.
    title: project name (required).
    repo: optional 'owner/repo-name' to link the project to a repository.
    Note: Projects v2 starts as a Table view. User can add a Board view from the project URL.
    """
    try:
        if not title or not title.strip():
            return {"status": "error", "message": "Please provide a project title."}

        # Get viewer node ID
        viewer_query = "query { viewer { id login } }"
        viewer_data  = _graphql(viewer_query)
        owner_id     = viewer_data.get("viewer", {}).get("id", "")
        if not owner_id:
            return {"status": "error", "message": "Could not get your GitHub user ID."}

        # Create the project
        create_mutation = """
        mutation($ownerId: ID!, $title: String!) {
          createProjectV2(input: { ownerId: $ownerId, title: $title }) {
            projectV2 { id number title url }
          }
        }
        """
        create_data = _graphql(create_mutation, {"ownerId": owner_id, "title": title.strip()})
        project     = create_data.get("createProjectV2", {}).get("projectV2", {})
        if not project.get("id"):
            return {"status": "error", "message": "Failed to create project."}

        project_id  = project["id"]
        project_url = project["url"]

        # Optionally link to repo
        repo_linked = ""
        if repo and repo.strip():
            repo = _resolve_repo(repo.strip())
            try:
                parts      = repo.split("/", 1)
                repo_owner = parts[0]
                repo_name  = parts[1] if len(parts) > 1 else repo
                link_query = """
                query($owner: String!, $repo: String!) {
                  repository(owner: $owner, name: $repo) { id }
                }
                """
                repo_data = _graphql(link_query, {"owner": repo_owner, "repo": repo_name})
                repo_id   = repo_data.get("repository", {}).get("id", "")
                if repo_id:
                    link_mutation = """
                    mutation($projectId: ID!, $repoId: ID!) {
                      linkProjectV2ToRepository(
                        input: { projectId: $projectId, repositoryId: $repoId }
                      ) { repository { nameWithOwner } }
                    }
                    """
                    _graphql(link_mutation, {"projectId": project_id, "repoId": repo_id})
                    repo_linked = repo
            except Exception:
                pass  # Linking is optional — project still created

        return {
            "status":     "success",
            "message":    f"Project '{title.strip()}' created successfully.",
            "project_id": project_id,
            "number":     project.get("number"),
            "title":      title.strip(),
            "url":        project_url,
            "repo_linked": repo_linked,
            "note": (
                "Project created! Open the URL to set up your board. "
                "GitHub Projects v2 starts as a Table view — click '+ New view' "
                "and choose 'Board' to get Kanban columns (Backlog, Ready, In Progress, etc.)."
            ),
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def get_project_columns(project_id: str) -> list:
    """
    Get all status columns/options of a GitHub Project v2 board.
    project_id: PVT_... node ID or project title.
    """
    try:
        project_id = _resolve_project_id(project_id)
        query = """
        query($projectId: ID!) {
          node(id: $projectId) {
            ... on ProjectV2 {
              fields(first: 20) {
                nodes {
                  ... on ProjectV2SingleSelectField {
                    id name
                    options { id name }
                  }
                }
              }
            }
          }
        }
        """
        data   = _graphql(query, {"projectId": project_id})
        fields = data.get("node", {}).get("fields", {}).get("nodes", [])
        columns = []
        for field in fields:
            if field.get("name", "").lower() == "status" and "options" in field:
                for opt in field["options"]:
                    columns.append({
                        "column_name":      opt["name"],
                        "column_option_id": opt["id"],
                        "status_field_id":  field["id"],
                    })
                break
        if not columns:
            return [{"message": "No status columns found in this project."}]
        return columns
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def add_issue_to_project(project_id: str, issue_url: str) -> dict:
    """
    Add an existing GitHub issue to a Project v2 board.
    project_id: PVT_... node ID or project title.
    issue_url: full HTML URL of the issue e.g. https://github.com/owner/repo/issues/12
    Returns the project item ID needed for move/update operations.
    """
    try:
        if not project_id or not project_id.strip():
            return {"status": "error", "message": "project_id is required."}
        if not issue_url or not issue_url.strip():
            return {"status": "error", "message": "issue_url is required."}
        project_id = _resolve_project_id(project_id.strip())
        parts      = issue_url.rstrip("/").split("/")
        number     = int(parts[-1])
        repo_name  = parts[-3]
        owner      = parts[-4]
        id_query = """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            issue(number: $number) { id }
          }
        }
        """
        id_data  = _graphql(id_query, {"owner": owner, "repo": repo_name, "number": number})
        issue_id = id_data.get("repository", {}).get("issue", {}).get("id", "")
        if not issue_id:
            return {"status": "error", "message": "Could not find issue node ID."}
        add_query = """
        mutation($projectId: ID!, $contentId: ID!) {
          addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
            item { id }
          }
        }
        """
        add_data = _graphql(add_query, {"projectId": project_id, "contentId": issue_id})
        item_id  = add_data.get("addProjectV2ItemById", {}).get("item", {}).get("id", "")
        if not item_id:
            return {"status": "error", "message": "Failed to add issue to project."}
        return {
            "status":  "success",
            "message": "Issue added to project successfully.",
            "item_id": item_id,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def move_issue_to_column(
    project_id: str,
    item_id: str,
    column_name: str = "Ready",
) -> dict:
    """
    Move a project issue card to a different column (status).
    project_id: PVT_... node ID or project title.
    item_id: project item ID from list_project_issues or add_issue_to_project.
    column_name: e.g. 'Backlog', 'Ready', 'In progress', 'In review', 'Done'.
    """
    try:
        if not project_id or not project_id.strip():
            return {"status": "error", "message": "project_id is required."}
        if not item_id or not item_id.strip():
            return {"status": "error", "message": "item_id is required. Call list_project_issues first to get it."}
        project_id = _resolve_project_id(project_id.strip())
        columns    = get_project_columns(project_id)
        if not columns or (len(columns) == 1 and "message" in columns[0]):
            return {"status": "error", "message": "Could not retrieve project columns."}
        target = next(
            (c for c in columns if c.get("column_name", "").lower() == (column_name or "").lower()),
            None
        )
        if not target:
            available = [c["column_name"] for c in columns]
            return {
                "status":  "error",
                "message": f"Column '{column_name}' not found. Available: {', '.join(available)}",
            }
        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId, itemId: $itemId, fieldId: $fieldId,
            value: { singleSelectOptionId: $optionId }
          }) { projectV2Item { id } }
        }
        """
        _graphql(mutation, {
            "projectId": project_id,
            "itemId":    item_id.strip(),
            "fieldId":   target["status_field_id"],
            "optionId":  target["column_option_id"],
        })
        return {
            "status":  "success",
            "message": f"Issue moved to '{column_name}' successfully.",
            "column":  column_name,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def update_project_issue_fields(
    project_id: str,
    item_id: str,
    start_date: str = "",
    end_date: str = "",
) -> dict:
    """
    Update custom date fields (start date, end date) on a project board item.
    project_id: PVT_... node ID or project title.
    item_id: project item ID from list_project_issues.
    start_date / end_date: YYYY-MM-DD format.
    """
    try:
        if not project_id or not project_id.strip():
            return {"status": "error", "message": "project_id is required."}
        if not item_id or not item_id.strip():
            return {"status": "error", "message": "item_id is required. Call list_project_issues first."}
        project_id = _resolve_project_id(project_id.strip())
        query = """
        query($projectId: ID!) {
          node(id: $projectId) {
            ... on ProjectV2 {
              fields(first: 20) {
                nodes { ... on ProjectV2Field { id name dataType } }
              }
            }
          }
        }
        """
        data      = _graphql(query, {"projectId": project_id})
        fields    = data.get("node", {}).get("fields", {}).get("nodes", [])
        field_map = {
            f.get("name", "").lower(): f.get("id")
            for f in fields if f.get("dataType") == "DATE" and f.get("id")
        }
        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $date: Date!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId, itemId: $itemId, fieldId: $fieldId,
            value: { date: $date }
          }) { projectV2Item { id } }
        }
        """
        updated = []
        if start_date and start_date.strip():
            fid = (field_map.get("start date") or field_map.get("start") or
                   field_map.get("startdate"))
            if fid:
                _graphql(mutation, {"projectId": project_id, "itemId": item_id.strip(),
                                    "fieldId": fid, "date": start_date.strip()})
                updated.append(f"Start Date: {start_date}")
            else:
                updated.append("⚠️ Start Date field not found — skipped")
        if end_date and end_date.strip():
            fid = (field_map.get("end date") or field_map.get("end") or
                   field_map.get("due date") or field_map.get("enddate") or
                   field_map.get("duedate"))
            if fid:
                _graphql(mutation, {"projectId": project_id, "itemId": item_id.strip(),
                                    "fieldId": fid, "date": end_date.strip()})
                updated.append(f"End Date: {end_date}")
            else:
                updated.append("⚠️ End Date field not found — skipped")
        if not updated:
            return {"status": "error", "message": "No date fields found to update."}
        return {
            "status":  "success",
            "message": "Project fields updated: " + ", ".join(updated),
            "updated": updated,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def list_project_issues(project_id: str, limit: str = "20") -> list:
    """
    List all issues/items in a GitHub project board with their status and item_id.
    project_id: project title (e.g. 'test project') or PVT_... node ID. limit: default 20.
    Each result includes item_id needed for move_issue_to_column and update_project_issue_fields.
    """
    try:
        if not project_id or not project_id.strip():
            projects = list_projects()
            if not projects or (isinstance(projects, list) and "message" in projects[0]):
                return [{"status": "error", "message": "You don't have any GitHub Projects."}]
            if len(projects) == 1:
                project_id = projects[0]["id"]
            else:
                titles = [p["title"] for p in projects[:5]]
                return [{"status": "error",
                         "message": f"Which project board? You have: {', '.join(titles)}"}]
        project_id = _resolve_project_id(project_id.strip())
        count      = _safe_int(limit, 20)
        query = """
        query($projectId: ID!, $limit: Int!) {
          node(id: $projectId) {
            ... on ProjectV2 {
              items(first: $limit) {
                nodes {
                  id
                  fieldValues(first: 20) {
                    nodes {
                      ... on ProjectV2ItemFieldSingleSelectValue {
                        name
                        field { ... on ProjectV2FieldCommon { name } }
                      }
                      ... on ProjectV2ItemFieldDateValue {
                        date
                        field { ... on ProjectV2FieldCommon { name } }
                      }
                    }
                  }
                  content {
                    ... on Issue {
                      number title url state
                      assignees(first: 3) { nodes { login } }
                      labels(first: 5) { nodes { name } }
                    }
                  }
                }
              }
            }
          }
        }
        """
        data  = _graphql(query, {"projectId": project_id, "limit": count})
        items = data.get("node", {}).get("items", {}).get("nodes", [])
        if not items:
            return [{"message": "No items found in this project."}]
        result = []
        for item in items:
            content = item.get("content", {})
            if not content or not content.get("title"):
                continue
            status = start_date = end_date = ""
            for fv in item.get("fieldValues", {}).get("nodes", []):
                fname = fv.get("field", {}).get("name", "").lower()
                if fname == "status" and fv.get("name"):
                    status = fv["name"]
                elif "start" in fname and fv.get("date"):
                    start_date = fv["date"]
                elif ("end" in fname or "due" in fname) and fv.get("date"):
                    end_date = fv["date"]
            result.append({
                "item_id":    item["id"],
                "number":     content.get("number", ""),
                "title":      content.get("title", ""),
                "status":     status or "No status",
                "state":      content.get("state", ""),
                "url":        content.get("url", ""),
                "assignees":  [a["login"] for a in content.get("assignees", {}).get("nodes", [])],
                "labels":     [l["name"] for l in content.get("labels", {}).get("nodes", [])],
                "start_date": start_date,
                "end_date":   end_date,
            })
        return result if result else [{"message": "No issue items found in this project."}]
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def update_project_issue_by_title(
    project_id: str,
    issue_title: str,
    assignee: str = "",
    labels: str = "",
    start_date: str = "",
    end_date: str = "",
    move_to_column: str = "",
) -> dict:
    """
    Update a project board issue by its TITLE — no need to know item_id.
    Automatically looks up item_id from the board, then applies all updates.
    project_id: PVT_... node ID or project title (e.g. 'test project').
    issue_title: part of or exact issue title to match.
    assignee: GitHub username (optional).
    labels: comma-separated label names (optional).
    start_date: YYYY-MM-DD (optional).
    end_date: YYYY-MM-DD (optional).
    move_to_column: target column e.g. 'Ready', 'In progress', 'Done' (optional).
    """
    try:
        if not project_id or not project_id.strip():
            return {"status": "error", "message": "project_id is required."}
        if not issue_title or not issue_title.strip():
            return {"status": "error", "message": "issue_title is required."}

        project_id = _resolve_project_id(project_id.strip())

        # Step 1: Find the issue on the board by title
        board_items = list_project_issues(project_id)
        if not board_items or (len(board_items) == 1 and "message" in board_items[0]):
            return {"status": "error", "message": "No issues found on this project board."}

        search  = issue_title.strip().lower()
        matched = [
            item for item in board_items
            if isinstance(item, dict) and item.get("title")
            and search in item["title"].lower()
        ]
        if not matched:
            titles = [item.get("title", "") for item in board_items if isinstance(item, dict) and item.get("title")]
            return {
                "status":  "error",
                "message": (f"No issue matching '{issue_title}' found on the board. "
                            f"Available: {', '.join(titles[:5])}"),
            }

        item      = matched[0]
        item_id   = item["item_id"]
        issue_num = item.get("number", "")
        issue_url = item.get("url", "")
        results   = []

        # Step 2: Update issue assignee and labels via GitHub REST API
        if issue_url and (assignee or labels):
            try:
                url_parts    = issue_url.rstrip("/").split("/")
                issue_number = int(url_parts[-1])
                repo_name    = url_parts[-4] + "/" + url_parts[-3]
                g     = get_github_client()
                r     = g.get_repo(repo_name)
                issue = r.get_issue(issue_number)
                update_kwargs = {}
                if assignee and assignee.strip():
                    update_kwargs["assignees"] = [assignee.strip()]
                    results.append(f"Assigned to @{assignee.strip()}")
                if labels and labels.strip():
                    label_list = [l.strip() for l in labels.split(",") if l.strip()]
                    update_kwargs["labels"] = label_list
                    results.append(f"Labels: {', '.join(label_list)}")
                if update_kwargs:
                    issue.edit(**update_kwargs)
            except Exception as e:
                results.append(f"⚠️ Could not update issue fields: {str(e)[:80]}")

        # Step 3: Update date fields on the project board
        if start_date or end_date:
            date_result = update_project_issue_fields(
                project_id, item_id,
                start_date=start_date,
                end_date=end_date,
            )
            if date_result.get("status") == "success":
                results.extend(date_result.get("updated", []))
            else:
                results.append(f"⚠️ Date update: {date_result.get('message', '')}")

        # Step 4: Move to column if requested
        if move_to_column and move_to_column.strip():
            move_result = move_issue_to_column(project_id, item_id, move_to_column.strip())
            if move_result.get("status") == "success":
                results.append(f"Moved to '{move_to_column}'")
            else:
                results.append(f"⚠️ Move failed: {move_result.get('message', '')}")

        summary = ", ".join(results) if results else "No changes made."
        return {
            "status":    "success",
            "message":   f"Issue #{issue_num} updated: {summary}",
            "item_id":   item_id,
            "issue_num": issue_num,
            "updates":   results,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def create_project_issue(
    repo: str,
    project_id: str,
    title: str,
    body: str = "",
    labels: str = "",
    assignee: str = "",
    start_date: str = "",
    end_date: str = "",
) -> dict:
    """
    Full flow: Create an issue in a repo AND add it to a project board in the Backlog column.
    Also sets assignee, labels, start date and end date automatically.
    repo: full repo name OR short name. project_id: PVT_... or project title.
    title: required. body/labels/assignee/start_date/end_date: optional.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        if not project_id or not project_id.strip():
            return {"status": "error", "message": "Please provide the project board name or ID."}
        if not title or not title.strip():
            return {"status": "error", "message": "Please provide an issue title."}
        project_id = _resolve_project_id(project_id.strip())

        # Step 1: Create the issue
        issue_result = create_issue(
            repo=repo, title=title, body=body,
            labels=labels, assignee=assignee,
            start_date=start_date, end_date=end_date,
        )
        if issue_result.get("status") != "success":
            return issue_result

        issue_url    = issue_result["url"]
        issue_number = issue_result["number"]

        # Step 2: Add to project board
        add_result = add_issue_to_project(project_id, issue_url)
        if add_result.get("status") != "success":
            return {
                "status":  "success",
                "message": (
                    f"✅ Issue #{issue_number} created successfully!\n\n"
                    f"⚠️ Could not add to project board: {add_result.get('message')}\n\n"
                    "Please do NOT retry. You can manually add the issue to the board."
                ),
                "number":  issue_number,
                "url":     issue_url,
                "partial": True,
            }

        item_id = add_result["item_id"]

        # Step 3: Move to Backlog
        move_issue_to_column(project_id, item_id, "Backlog")

        # Step 4: Set date fields if provided
        date_result = None
        if start_date or end_date:
            date_result = update_project_issue_fields(
                project_id, item_id,
                start_date=start_date,
                end_date=end_date,
            )

        return {
            "status":     "success",
            "message":    f"Issue #{issue_number} created and added to project board in Backlog.",
            "number":     issue_number,
            "url":        issue_url,
            "item_id":    item_id,
            "repo":       repo,
            "project_id": project_id,
            "column":     "Backlog",
            "assignee":   assignee or "",
            "start_date": start_date or "",
            "end_date":   end_date or "",
            "dates_set":  date_result.get("message", "") if date_result else "",
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run()