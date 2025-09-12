from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from typing import Dict, Any, Optional
import asyncio
from loguru import logger
from api.github_client import GitHubClient
from config.settings import settings

router = APIRouter(prefix="/auth", tags=["authentication"])

# Store active GitHub clients (in production, use Redis or database)
active_github_clients: Dict[str, GitHubClient] = {}

@router.get("/github")
async def github_auth_start():
    """Start GitHub OAuth authorization"""
    try:
        github_client = GitHubClient()
        auth_url = github_client.get_authorization_url()
        
        # Store client for later use (in production, use proper session management)
        client_id = f"client_{len(active_github_clients)}"
        active_github_clients[client_id] = github_client
        
        # Return HTML page with authorization link
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GitHub Authorization</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 24px;
                    background-color: #24292e;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .btn:hover {{
                    background-color: #1a1e22;
                }}
                .info {{
                    background-color: #e3f2fd;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .code {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 3px;
                    font-family: monospace;
                    word-break: break-all;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔐 GitHub Authorization</h1>
                <div class="info">
                    <p><strong>Authorization Required</strong></p>
                    <p>To access GitHub data for crypto analytics, you need to authorize this application.</p>
                </div>
                
                <p>Click the button below to authorize with GitHub:</p>
                <a href="{auth_url}" class="btn">Authorize with GitHub</a>
                
                <div class="info">
                    <p><strong>After authorization:</strong></p>
                    <p>1. You'll be redirected back to this application</p>
                    <p>2. The authorization code will be automatically processed</p>
                    <p>3. GitHub data collection will begin</p>
                </div>
                
                <p><strong>Client ID:</strong> <span class="code">{client_id}</span></p>
                <p><em>Keep this Client ID for the callback process</em></p>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error starting GitHub authorization: {e}")
        raise HTTPException(status_code=500, detail="Failed to start GitHub authorization")

@router.get("/github/callback")
async def github_auth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    client_id: Optional[str] = None
):
    """Handle GitHub OAuth callback"""
    try:
        if error:
            logger.error(f"GitHub authorization error: {error}")
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
                    .error {{ background-color: #ffebee; padding: 20px; border-radius: 5px; color: #c62828; }}
                </style>
            </head>
            <body>
                <div class="error">
                    <h2>❌ Authorization Failed</h2>
                    <p>Error: {error}</p>
                    <p>Please try again or contact support.</p>
                </div>
            </body>
            </html>
            """)
        
        if not code:
            raise HTTPException(status_code=400, detail="Authorization code not provided")
        
        # Find the GitHub client (in production, use proper session management)
        github_client = None
        if client_id and client_id in active_github_clients:
            github_client = active_github_clients[client_id]
        else:
            # Create new client if not found
            github_client = GitHubClient()
        
        # Complete authorization
        success = await github_client.complete_authorization(code)
        
        if success:
            # Get user info
            user_info = await github_client.get_user_info()
            username = user_info.get("login", "Unknown")
            
            # Clean up
            if client_id in active_github_clients:
                del active_github_clients[client_id]
            
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Success</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
                    .success {{ background-color: #e8f5e8; padding: 20px; border-radius: 5px; color: #2e7d32; }}
                    .btn {{ display: inline-block; padding: 10px 20px; background-color: #4caf50; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }}
                </style>
            </head>
            <body>
                <div class="success">
                    <h2>✅ Authorization Successful!</h2>
                    <p>Welcome, <strong>{username}</strong>!</p>
                    <p>GitHub authorization completed successfully. The system can now collect GitHub metrics for crypto analytics.</p>
                    <a href="/" class="btn">Go to Dashboard</a>
                </div>
            </body>
            </html>
            """)
        else:
            raise HTTPException(status_code=400, detail="Failed to complete authorization")
            
    except Exception as e:
        logger.error(f"Error in GitHub callback: {e}")
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authorization Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
                .error {{ background-color: #ffebee; padding: 20px; border-radius: 5px; color: #c62828; }}
            </style>
        </head>
        <body>
            <div class="error">
                <h2>❌ Authorization Error</h2>
                <p>An error occurred during authorization: {str(e)}</p>
                <p>Please try again or contact support.</p>
            </div>
        </body>
        </html>
        """)

@router.get("/github/status")
async def github_auth_status():
    """Check GitHub authorization status"""
    try:
        # Check if we have a valid token
        github_client = GitHubClient()
        if github_client.access_token:
            is_valid = await github_client.check_token_validity()
            if is_valid:
                user_info = await github_client.get_user_info()
                return {
                    "authorized": True,
                    "user": user_info.get("login"),
                    "avatar_url": user_info.get("avatar_url"),
                    "name": user_info.get("name")
                }
        
        return {"authorized": False}
        
    except Exception as e:
        logger.error(f"Error checking GitHub status: {e}")
        return {"authorized": False, "error": str(e)}

@router.post("/github/revoke")
async def github_revoke_auth():
    """Revoke GitHub authorization"""
    try:
        # In a real implementation, you would revoke the token with GitHub
        # For now, we'll just clear the local token
        github_client = GitHubClient()
        github_client.access_token = None
        
        return {"message": "GitHub authorization revoked successfully"}
        
    except Exception as e:
        logger.error(f"Error revoking GitHub authorization: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke authorization")

@router.get("/github/test")
async def github_test_connection():
    """Test GitHub API connection"""
    try:
        async with GitHubClient() as github_client:
            # Test with a public repository
            test_repo = "microsoft/vscode"
            repo_info = await github_client.get_repository_info(test_repo)
            
            if repo_info:
                return {
                    "status": "success",
                    "message": "GitHub API connection successful",
                    "test_repo": {
                        "name": repo_info.get("name"),
                        "stars": repo_info.get("stargazers_count"),
                        "forks": repo_info.get("forks_count")
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to fetch repository information"
                }
                
    except Exception as e:
        logger.error(f"GitHub API test failed: {e}")
        return {
            "status": "error",
            "message": f"GitHub API test failed: {str(e)}"
        }
