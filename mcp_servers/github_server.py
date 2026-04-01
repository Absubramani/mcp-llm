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
    """Get raw access token string for GraphQL calls."""
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


def _normalize_repo(repo: str) -> str:
    """Fix common hallucinations and normalize repo name."""
    if not repo:
        return repo
    repo = repo.strip()
    # Fix double-s typo that LLM sometimes produces
    repo = repo.replace("Abssubramani/", "Absubramani/")
    repo = repo.replace("abssubramani/", "Absubramani/")
    return repo


def _resolve_short_repo(short_name: str) -> str:
    """
    Resolve a short repo name (no slash) to full 'owner/repo' by searching
    the authenticated user's repos. Returns '' if not found.
    """
    if not short_name or "/" in short_name:
        return short_name
    try:
        g = get_github_client()
        user = g.get_user()
        for r in user.get_repos(sort="updated"):
            if r.name.lower() == short_name.lower():
                return r.full_name
    except Exception:
        pass
    return ""


def _resolve_repo(repo: str) -> str:
    """
    Master repo resolver:
    1. Normalize spelling errors
    2. If no slash → try to find full name from user repos
    3. Return best guess
    """
    if not repo:
        return ""
    repo = _normalize_repo(repo)
    if "/" not in repo:
        resolved = _resolve_short_repo(repo)
        if resolved:
            return resolved
    return repo


def _resolve_project_id(id_or_title: str) -> str:
    """
    If input doesn't look like a node ID (PVT_... or PN_...), find it by title
    from the user's projects list.
    """
    if not id_or_title:
        return id_or_title
    if id_or_title.startswith("PVT_") or id_or_title.startswith("PN_"):
        return id_or_title
    try:
        projects = list_projects()
        for p in projects:
            if isinstance(p, dict) and "title" in p:
                if p["title"].lower() == id_or_title.lower():
                    return p.get("id", id_or_title)
    except Exception:
        pass
    return id_or_title


def _graphql(query: str, variables: dict = None) -> dict:
    """Execute a GitHub GraphQL query."""
    token = _get_access_token()
    resp = requests.post(
        "https://api.github.com/graphql",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"query": query, "variables": variables or {}},
        timeout=15,
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
        g     = get_github_client()
        user  = g.get_user()
        repos = list(user.get_repos(sort="updated"))[:int(limit or 10)]
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
            "auto_init":   auto_init.lower() == "true",
            "description": repo.description or "",
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
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
        repos = list(g.search_repositories(query=query))[:int(limit or 5)]
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


@mcp.tool()
def list_repo_files(repo: str, path: str = "", branch: str = "") -> list:
    """
    List files and directories in a GitHub repository.
    repo: full repo name like 'username/repo-name' or short name like 'my-repo'.
    path: internal path in the repo (empty for root).
    branch: branch name (optional).
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return [{"status": "error", "message": "Please provide the repository name (e.g. username/repo-name)."}]
        g        = get_github_client()
        r        = g.get_repo(repo)
        kwargs   = {}
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


@mcp.tool()
def read_file_from_repo(repo: str, file_path: str, branch: str = "") -> dict:
    """
    Read the content of a file from a GitHub repository.
    repo: full repo name like 'username/repo-name' or short name like 'my-repo'.
    file_path: full path to the file in the repo (e.g. 'README.md', 'agent/orchestrator.py').
    branch: branch name (optional, defaults to default branch).
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name (e.g. username/repo-name)."}
        if not file_path:
            return {"status": "error", "message": "Please provide the file path to read."}
        g      = get_github_client()
        r      = g.get_repo(repo)
        kwargs = {}
        if branch:
            kwargs["ref"] = branch
        content = r.get_contents(file_path, **kwargs)
        text    = content.decoded_content.decode("utf-8", errors="ignore")
        return {
            "file":    file_path,
            "repo":    repo,
            "branch":  branch or r.default_branch,
            "content": text[:3000],
            "size":    content.size,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def list_branches(repo: str) -> list:
    """
    List all branches in a repository.
    repo: full repo name like 'username/repo-name' or short name like 'my-repo'.
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
            {
                "name":    b.name,
                "default": b.name == r.default_branch,
            }
            for b in branches
        ]
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except GithubException as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def list_issues(repo: str, state: str = "open", limit: str = "") -> list:
    """
    List issues in a repository.
    repo: full repo name like 'username/repo-name' or short name like 'my-repo'.
    state: open, closed, or all.
    limit: number of results (max 100).
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return [{"status": "error", "message": "Please provide the repository name (e.g. username/repo-name)."}]
        g      = get_github_client()
        r      = g.get_repo(repo)
        issues = r.get_issues(state=state or "open")
        res    = []
        for i in list(issues)[:int(limit or 20)]:
            res.append({
                "number":     i.number,
                "title":      i.title,
                "state":      i.state,
                "author":     i.user.login if i.user else "",
                "created_at": i.created_at.isoformat(),
                "url":        i.html_url,
            })
        if not res:
            return [{"message": f"No {state or 'open'} issues found in {repo}."}]
        return res
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except GithubException as e:
        return [{"status": "error", "message": f"GitHub Error: {str(e)}"}]


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
    repo: full repo name like 'username/repo-name' or short name.
    title: issue title (required).
    body: issue description (optional).
    labels: comma separated label names (optional).
    assignee: GitHub username to assign the issue to (optional).
    start_date: start date in YYYY-MM-DD format (optional).
    end_date: end date / due date in YYYY-MM-DD format (optional).
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name (e.g. username/repo-name)."}
        if not title:
            return {"status": "error", "message": "Please provide an issue title."}
        g          = get_github_client()
        r          = g.get_repo(repo)
        label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else []
        assignees  = [assignee.strip()] if assignee and assignee.strip() else []

        full_body = body or ""
        if start_date or end_date:
            full_body = full_body.rstrip()
            if full_body:
                full_body += "\n\n"
            if start_date:
                full_body += f"**Start Date:** {start_date}\n"
            if end_date:
                full_body += f"**End Date:** {end_date}\n"

        create_kwargs = {
            "title":  title,
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


@mcp.tool()
def read_issue(repo: str, issue_number: str) -> dict:
    """
    Read full details of a specific issue.
    repo: full repo name like 'username/repo-name'.
    issue_number: the issue number.
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
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
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
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
        repo = _resolve_repo(repo)
        if not repo:
            return [{"status": "error", "message": "Please provide the repository name."}]
        g   = get_github_client()
        r   = g.get_repo(repo)
        prs = list(r.get_pulls(state=state or "open"))[:int(limit or 5)]
        if not prs:
            return [{"message": f"No {state or 'open'} pull requests found in {repo}.", "empty": True}]
        return [
            {
                "number":  pr.number,
                "title":   pr.title,
                "state":   pr.state,
                "author":  pr.user.login if pr.user else "",
                "head":    pr.head.ref,
                "base":    pr.base.ref,
                "created": pr.created_at.isoformat() if pr.created_at else "",
                "url":     pr.html_url,
            }
            for pr in prs
        ]
    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
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
    head: the branch containing your changes (must be an existing branch).
    base: the branch you want to merge into (default 'main').
    body: pull request description (optional).
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        if not title:
            return {"status": "error", "message": "Please provide a PR title."}
        if not head:
            return {"status": "error", "message": "Please provide the head branch (source branch with your changes)."}
        g  = get_github_client()
        r  = g.get_repo(repo)
        try:
            r.get_branch(head)
        except GithubException:
            branches = [b.name for b in list(r.get_branches())[:10]]
            return {
                "status":  "error",
                "message": f"Branch '{head}' does not exist in {repo}. Available branches: {', '.join(branches)}",
            }
        pr = r.create_pull(
            title = title,
            head  = head,
            base  = base or "main",
            body  = body or "",
        )
        return {
            "status":  "success",
            "message": f"Pull request #{pr.number} created successfully.",
            "number":  pr.number,
            "title":   pr.title,
            "url":     pr.html_url,
            "head":    head,
            "base":    base or "main",
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def create_branch(repo: str, branch_name: str, base_branch: str = "main") -> dict:
    """
    Create a new branch in a repository.
    repo: full repo name like 'username/repo-name'.
    branch_name: name of the new branch to create.
    base_branch: the existing branch to branch off from (default 'main').
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        g = get_github_client()
        r = g.get_repo(repo)
        try:
            source = r.get_branch(base_branch or "main")
        except GithubException:
            return {"status": "error", "message": f"Base branch '{base_branch}' does not exist in {repo}."}
        try:
            r.create_git_ref(ref=f"refs/heads/{branch_name}", sha=source.commit.sha)
            return {
                "status":  "success",
                "message": f"Branch '{branch_name}' created successfully from '{base_branch}'.",
                "branch":  branch_name,
                "repo":    repo,
            }
        except GithubException as e:
            if "Reference already exists" in str(e) or getattr(e, "status", 0) == 422:
                return {"status": "error", "message": f"Branch '{branch_name}' already exists."}
            return {"status": "error", "message": str(e)}
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except GithubException as e:
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# GITHUB PROJECTS v2 — GraphQL-based tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_projects(repo: str = "", limit: str = "5") -> list:
    """
    List GitHub Projects (v2) for the authenticated user or a specific repo.
    repo: optional 'owner/repo-name' to list repo-linked projects.
    limit: number of projects to return (default 10).
    If repo is empty, lists all projects owned by the authenticated user.
    """
    try:
        count = int(limit or 10)
        projects = []

        if repo:
            repo = _resolve_repo(repo)
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
            data = _graphql(query, {"owner": owner, "repo": repo_name, "limit": count})
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
            owner_login  = ""
            if p.get("owner"):
                owner_login = p["owner"].get("login", "")
            linked_repos = [r["nameWithOwner"] for r in p.get("repositories", {}).get("nodes", []) if r.get("nameWithOwner")]
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
def get_project_columns(project_id: str) -> list:
    """
    Get all status columns/options of a GitHub Project v2 board.
    project_id: the node ID of the project (from list_projects), or project title.
    Returns list of columns with their option IDs — needed for move_issue_to_column.
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
    project_id: the node ID of the project (from list_projects) or project title.
    issue_url: the full HTML URL of the issue e.g. https://github.com/owner/repo/issues/12
    Returns the project item ID needed for move_issue_to_column and update_project_issue_fields.
    """
    try:
        if not project_id:
            return {"status": "error", "message": "project_id is required."}
        if not issue_url:
            return {"status": "error", "message": "issue_url is required."}

        project_id = _resolve_project_id(project_id)

        parts     = issue_url.rstrip("/").split("/")
        number    = int(parts[-1])
        repo_name = parts[-3]
        owner     = parts[-4]

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
    project_id: the node ID of the project (from list_projects) or project title.
    item_id: the project item ID (from add_issue_to_project or list_project_issues).
    column_name: name of the target column e.g. 'Backlog', 'Ready', 'In progress', 'In review', 'Done'.
    """
    try:
        project_id = _resolve_project_id(project_id)
        columns    = get_project_columns(project_id)
        if not columns or (len(columns) == 1 and "message" in columns[0]):
            return {"status": "error", "message": "Could not retrieve project columns."}

        target = None
        for col in columns:
            if col.get("column_name", "").lower() == column_name.lower():
                target = col
                break

        if not target:
            available = [c["column_name"] for c in columns]
            return {
                "status":  "error",
                "message": f"Column '{column_name}' not found. Available: {', '.join(available)}",
            }

        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId,
            itemId: $itemId,
            fieldId: $fieldId,
            value: { singleSelectOptionId: $optionId }
          }) {
            projectV2Item { id }
          }
        }
        """
        _graphql(mutation, {
            "projectId": project_id,
            "itemId":    item_id,
            "fieldId":   target["status_field_id"],
            "optionId":  target["column_option_id"],
        })
        return {
            "status":  "success",
            "message": f"Issue moved to **{column_name}** successfully.",
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
    project_id: the node ID of the project (from list_projects) or project title.
    item_id: the project item ID (from add_issue_to_project or list_project_issues).
    start_date: start date in YYYY-MM-DD format (optional).
    end_date: end/due date in YYYY-MM-DD format (optional).
    """
    try:
        project_id = _resolve_project_id(project_id)
        query = """
        query($projectId: ID!) {
          node(id: $projectId) {
            ... on ProjectV2 {
              fields(first: 20) {
                nodes {
                  ... on ProjectV2Field {
                    id name dataType
                  }
                }
              }
            }
          }
        }
        """
        data   = _graphql(query, {"projectId": project_id})
        fields = data.get("node", {}).get("fields", {}).get("nodes", [])

        field_map = {
            f.get("name", "").lower(): f.get("id")
            for f in fields
            if f.get("dataType") == "DATE" and f.get("id")
        }

        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $date: Date!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId,
            itemId: $itemId,
            fieldId: $fieldId,
            value: { date: $date }
          }) {
            projectV2Item { id }
          }
        }
        """
        updated = []

        if start_date:
            start_field_id = (
                field_map.get("start date") or
                field_map.get("start") or
                field_map.get("startdate")
            )
            if start_field_id:
                _graphql(mutation, {
                    "projectId": project_id,
                    "itemId":    item_id,
                    "fieldId":   start_field_id,
                    "date":      start_date,
                })
                updated.append(f"Start Date: {start_date}")
            else:
                updated.append("⚠️ Start Date field not found in project — skipped")

        if end_date:
            end_field_id = (
                field_map.get("end date") or
                field_map.get("end") or
                field_map.get("due date") or
                field_map.get("enddate") or
                field_map.get("duedate")
            )
            if end_field_id:
                _graphql(mutation, {
                    "projectId": project_id,
                    "itemId":    item_id,
                    "fieldId":   end_field_id,
                    "date":      end_date,
                })
                updated.append(f"End Date: {end_date}")
            else:
                updated.append("⚠️ End Date field not found in project — skipped")

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
def list_project_issues(project_id: str, limit: str = "10") -> list:
    """
    List all issues/items in a GitHub project board.
    project_id: the Project Title (e.g. 'test project') or node ID (PVT_...).
    If project_id is empty, tries to find your only/default project board.
    """
    try:
        if not project_id:
            projects = list_projects()
            if not projects or (isinstance(projects, list) and "message" in projects[0]):
                return [{"status": "error", "message": "You don't have any GitHub Projects yet."}]
            if len(projects) == 1:
                project_id = projects[0]["id"]
            else:
                titles = [p["title"] for p in projects[:5]]
                return [{"status": "error", "message": f"Which project board should I check? You have {len(projects)} projects: {', '.join(titles)}"}]

        project_id = _resolve_project_id(project_id)
        count      = int(limit or 20)

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

            status     = ""
            start_date = ""
            end_date   = ""
            for fv in item.get("fieldValues", {}).get("nodes", []):
                field_name = fv.get("field", {}).get("name", "").lower()
                if field_name == "status" and fv.get("name"):
                    status = fv["name"]
                elif "start" in field_name and fv.get("date"):
                    start_date = fv["date"]
                elif ("end" in field_name or "due" in field_name) and fv.get("date"):
                    end_date = fv["date"]

            assignees = [a["login"] for a in content.get("assignees", {}).get("nodes", [])]
            labels    = [l["name"] for l in content.get("labels", {}).get("nodes", [])]

            result.append({
                "item_id":    item["id"],
                "number":     content.get("number", ""),
                "title":      content.get("title", ""),
                "status":     status or "No status",
                "state":      content.get("state", ""),
                "url":        content.get("url", ""),
                "assignees":  assignees,
                "labels":     labels,
                "start_date": start_date,
                "end_date":   end_date,
            })

        return result if result else [{"message": "No issue items found in this project."}]

    except RuntimeError as e:
        return [{"status": "error", "message": str(e)}]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


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
    repo: full repo name like 'owner/repo-name'.
    project_id: the node ID of the project (from list_projects) or project title.
    title: issue title (required).
    body: issue description (optional).
    labels: comma separated label names (optional).
    assignee: GitHub username to assign to (optional).
    start_date: start date YYYY-MM-DD (optional).
    end_date: end/due date YYYY-MM-DD (optional).
    """
    try:
        repo = _resolve_repo(repo)
        if not repo:
            return {"status": "error", "message": "Please provide the repository name."}
        if not project_id:
            return {"status": "error", "message": "Please provide the project board name or ID. (Call list_projects first if unsure)"}
        if not title:
            return {"status": "error", "message": "Please provide an issue title."}

        project_id = _resolve_project_id(project_id)

        issue_result = create_issue(
            repo=repo, title=title, body=body,
            labels=labels, assignee=assignee,
            start_date=start_date, end_date=end_date,
        )
        if issue_result.get("status") != "success":
            return issue_result

        issue_url    = issue_result["url"]
        issue_number = issue_result["number"]

        add_result = add_issue_to_project(project_id, issue_url)
        if add_result.get("status") != "success":
            return {
                "status":  "success",
                "message": (
                    f"✅ Issue #{issue_number} created successfully!\n\n"
                    f"⚠️ Could not add to project board: {add_result.get('message')}\n\n"
                    "Please do NOT retry creation. You can manually add the issue to the board."
                ),
                "number":  issue_number,
                "url":     issue_url,
                "partial": True,
            }

        item_id = add_result["item_id"]
        move_issue_to_column(project_id, item_id, "Backlog")

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