import pickle
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload, MediaFileUpload

# ================= CONFIG =================

BASE_DIR = Path(__file__).parent.parent
CLIENT_SECRET_FILE = BASE_DIR / "oauth-client.json"
TOKEN_FILE = BASE_DIR / "token.pickle"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://mail.google.com/",
]

# ================= AUTH =================

creds = None
if TOKEN_FILE.exists():
    with open(TOKEN_FILE, "rb") as token:
        creds = pickle.load(token)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
        creds = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "wb") as token:
        pickle.dump(creds, token)

drive_service = build("drive", "v3", credentials=creds)
mcp = FastMCP("Advanced Drive MCP")

# ================= CORE HELPERS =================

def find_item(name, parent_id=None, mime_type=None):
    """Find a file or folder by name, optionally within a parent."""
    query = f"name='{name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    if mime_type:
        query += f" and mimeType='{mime_type}'"

    results = drive_service.files().list(
        q=query,
        fields="files(id, name, mimeType, parents, size, modifiedTime)"
    ).execute()
    return results.get("files", [])


def resolve_path(path, create_missing=False):
    """
    Resolve a path like 'Projects/2024/Reports' to a folder ID.
    Supports nested paths of any depth.
    If create_missing=True, creates folders that don't exist.
    """
    if not path or path.strip("/") == "":
        return None  # root

    parts = [p for p in path.strip("/").split("/") if p]
    parent_id = None

    for part in parts:
        folders = find_item(
            part, parent_id,
            mime_type="application/vnd.google-apps.folder"
        )
        if folders:
            parent_id = folders[0]["id"]
        else:
            if not create_missing:
                return None
            metadata = {
                "name": part,
                "mimeType": "application/vnd.google-apps.folder"
            }
            if parent_id:
                metadata["parents"] = [parent_id]
            folder = drive_service.files().create(
                body=metadata, fields="id"
            ).execute()
            parent_id = folder["id"]

    return parent_id


def split_path(path):
    """Split a full path into (folder_path, file_name)."""
    parts = path.strip("/").split("/")
    return "/".join(parts[:-1]), parts[-1]


# ================= TOOLS =================

@mcp.tool()
def create_folder(path: str) -> dict:
    """
    Create a folder or nested folder structure.
    Examples: 'Projects', 'Projects/2024/Reports'
    Creates all missing parent folders automatically.
    """
    folder_id = resolve_path(path, create_missing=True)
    return {
        "status": "success",
        "message": f"Folder '{path}' created successfully.",
        "folder_id": folder_id
    }


@mcp.tool()
def list_files(path: str = "") -> list:
    """
    List all files and folders at a given path.
    Leave path empty to list root. Supports nested paths like 'Projects/2024'.
    """
    parent_id = resolve_path(path) if path else None

    if path and parent_id is None:
        return [{"status": "error", "message": f"Folder '{path}' not found."}]

    query = "trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = drive_service.files().list(
        q=query,
        fields="files(id, name, mimeType, size, modifiedTime)"
    ).execute()

    files = results.get("files", [])
    if not files:
        return [{"message": "No files found."}]

    return [
        {
            "name": f["name"],
            "id": f["id"],
            "type": "folder" if f["mimeType"] == "application/vnd.google-apps.folder" else "file",
            "size": f.get("size", "—"),
            "modified": f.get("modifiedTime", "—")
        }
        for f in files
    ]


@mcp.tool()
def create_text_file(path: str, content: str, overwrite: bool = False, append: bool = False) -> dict:
    """
    Create a text file at a given path with content.
    Path supports nested folders like 'Projects/2024/notes.txt'.
    Use overwrite=True to replace existing file.
    Use append=True to add content to existing file.
    """
    folder_path, file_name = split_path(path)

    if not file_name.endswith(".txt"):
        file_name += ".txt"

    parent_id = resolve_path(folder_path, create_missing=True) if folder_path else None
    existing = find_item(file_name, parent_id)

    if existing:
        file_id = existing[0]["id"]

        if not overwrite and not append:
            return {"status": "exists", "message": f"File '{file_name}' already exists. Use overwrite=True or append=True."}

        if append:
            # Read existing content and append
            existing_content = drive_service.files().get_media(fileId=file_id).execute()
            content = existing_content.decode("utf-8") + "\n" + content

        drive_service.files().delete(fileId=file_id).execute()

    media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/plain")
    metadata = {"name": file_name}
    if parent_id:
        metadata["parents"] = [parent_id]

    file = drive_service.files().create(
        body=metadata, media_body=media, fields="id"
    ).execute()

    return {
        "status": "success",
        "message": f"File '{path}' created successfully.",
        "file_id": file["id"]
    }


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the content of a text file.
    Supports nested paths like 'Projects/2024/notes.txt'.
    """
    folder_path, file_name = split_path(path)
    parent_id = resolve_path(folder_path) if folder_path else None
    files = find_item(file_name, parent_id)

    if not files:
        return f"Error: File '{path}' not found."

    file_id = files[0]["id"]
    content = drive_service.files().get_media(fileId=file_id).execute()
    return content.decode("utf-8")


@mcp.tool()
def delete_file(path: str) -> dict:
    """
    Delete a file or folder by path.
    Supports nested paths like 'Projects/2024/notes.txt' or 'Projects/2024'.
    """
    folder_path, name = split_path(path)
    parent_id = resolve_path(folder_path) if folder_path else None
    files = find_item(name, parent_id)

    if not files:
        return {"status": "error", "message": f"'{path}' not found."}

    drive_service.files().delete(fileId=files[0]["id"]).execute()
    return {"status": "success", "message": f"'{path}' deleted successfully."}


@mcp.tool()
def move_file(source_path: str, destination_folder: str) -> dict:
    """
    Move a file or folder to another folder.
    Both paths support nested structure.
    Example: move_file('Projects/old.txt', 'Archive/2024')
    """
    folder_path, file_name = split_path(source_path)
    src_parent = resolve_path(folder_path) if folder_path else None
    files = find_item(file_name, src_parent)

    if not files:
        return {"status": "error", "message": f"'{source_path}' not found."}

    file_id = files[0]["id"]
    old_parents = ",".join(files[0].get("parents", []))
    dest_parent = resolve_path(destination_folder, create_missing=True)

    drive_service.files().update(
        fileId=file_id,
        addParents=dest_parent,
        removeParents=old_parents,
        fields="id, parents"
    ).execute()

    return {"status": "success", "message": f"'{source_path}' moved to '{destination_folder}' successfully."}


@mcp.tool()
def rename_file(path: str, new_name: str) -> dict:
    """
    Rename a file or folder.
    Example: rename_file('Projects/old_name.txt', 'new_name.txt')
    """
    folder_path, file_name = split_path(path)
    parent_id = resolve_path(folder_path) if folder_path else None
    files = find_item(file_name, parent_id)

    if not files:
        return {"status": "error", "message": f"'{path}' not found."}

    drive_service.files().update(
        fileId=files[0]["id"],
        body={"name": new_name}
    ).execute()

    return {"status": "success", "message": f"'{file_name}' renamed to '{new_name}' successfully."}


@mcp.tool()
def copy_file(source_path: str, destination_folder: str, new_name: str = None) -> dict:
    """
    Copy a file to another folder, optionally with a new name.
    Example: copy_file('Projects/report.txt', 'Archive/2024', 'report_backup.txt')
    """
    folder_path, file_name = split_path(source_path)
    parent_id = resolve_path(folder_path) if folder_path else None
    files = find_item(file_name, parent_id)

    if not files:
        return {"status": "error", "message": f"'{source_path}' not found."}

    dest_parent = resolve_path(destination_folder, create_missing=True)
    body = {"name": new_name or file_name, "parents": [dest_parent]}

    copied = drive_service.files().copy(
        fileId=files[0]["id"], body=body, fields="id, name"
    ).execute()

    return {
        "status": "success",
        "message": f"'{source_path}' copied to '{destination_folder}/{copied['name']}' successfully.",
        "file_id": copied["id"]
    }


@mcp.tool()
def search_files(query: str, path: str = "") -> list:
    """
    Search for files by name keyword across Drive or within a folder.
    Example: search_files('report', 'Projects/2024')
    """
    parent_id = resolve_path(path) if path else None

    search_query = f"name contains '{query}' and trashed=false"
    if parent_id:
        search_query += f" and '{parent_id}' in parents"

    results = drive_service.files().list(
        q=search_query,
        fields="files(id, name, mimeType, modifiedTime)"
    ).execute()

    files = results.get("files", [])
    if not files:
        return [{"message": f"No files found matching '{query}'."}]

    return [
        {
            "name": f["name"],
            "id": f["id"],
            "type": "folder" if f["mimeType"] == "application/vnd.google-apps.folder" else "file",
            "modified": f.get("modifiedTime", "—")
        }
        for f in files
    ]


@mcp.tool()
def get_file_info(path: str) -> dict:
    """
    Get detailed info about a file or folder (size, type, modified date, ID).
    """
    folder_path, file_name = split_path(path)
    parent_id = resolve_path(folder_path) if folder_path else None
    files = find_item(file_name, parent_id)

    if not files:
        return {"status": "error", "message": f"'{path}' not found."}

    f = files[0]
    return {
        "name": f["name"],
        "id": f["id"],
        "type": "folder" if f["mimeType"] == "application/vnd.google-apps.folder" else "file",
        "mimeType": f["mimeType"],
        "size": f.get("size", "—"),
        "modified": f.get("modifiedTime", "—"),
        "parents": f.get("parents", [])
    }


if __name__ == "__main__":
    mcp.run()