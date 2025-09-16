import aiohttp
import asyncio
import webbrowser
import urllib.parse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from config.settings import settings

class GitHubClient:
    """Client for GitHub API integration with OAuth web authorization"""
    
    def __init__(self):
        self.client_id = settings.GITHUB_CLIENT_ID
        self.client_secret = settings.GITHUB_CLIENT_SECRET
        self.redirect_uri = settings.GITHUB_REDIRECT_URI
        self.access_token = None
        self.base_url = "https://api.github.com"
        self.auth_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"
        self.session = None
        self.rate_limit_delay = 2.0  # Increased delay between requests
    
    async def __aenter__(self):
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"token {self.access_token}"
        elif hasattr(settings, 'GITHUB_TOKEN') and settings.GITHUB_TOKEN:
            headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
        
        self.session = aiohttp.ClientSession(headers=headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Get GitHub OAuth authorization URL"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "repo,read:user,read:org,user:email",
            "response_type": "code"
        }
        
        if state:
            params["state"] = state
        
        return f"{self.auth_url}?{urllib.parse.urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri
        }
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(
                self.token_url, 
                json=data, 
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "access_token" in result:
                        self.access_token = result["access_token"]
                        logger.info("Successfully obtained GitHub access token")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to exchange code for token: {response.status} - {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return {}
    
    async def get_user_info(self) -> Dict[str, Any]:
        """Get authenticated user information"""
        if not self.access_token:
            logger.warning("No access token available")
            return {}
        
        return await self._make_request("user")
    
    async def check_token_validity(self) -> bool:
        """Check if current access token is valid"""
        if not self.access_token:
            return False
        
        try:
            user_info = await self.get_user_info()
            return bool(user_info.get("id"))
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return False
    
    def authorize_via_web(self) -> str:
        """Open web browser for GitHub authorization and return authorization URL"""
        auth_url = self.get_authorization_url()
        logger.info(f"Opening GitHub authorization URL: {auth_url}")
        
        try:
            webbrowser.open(auth_url)
            logger.info("Web browser opened for GitHub authorization")
        except Exception as e:
            logger.error(f"Failed to open web browser: {e}")
            logger.info(f"Please manually open this URL: {auth_url}")
        
        return auth_url
    
    async def complete_authorization(self, code: str) -> bool:
        """Complete the OAuth authorization process"""
        try:
            token_result = await self.exchange_code_for_token(code)
            if token_result.get("access_token"):
                logger.info("GitHub authorization completed successfully")
                return True
            else:
                logger.error("Failed to complete GitHub authorization")
                return False
        except Exception as e:
            logger.error(f"Error completing GitHub authorization: {e}")
            return False
    
    async def _make_request(
        self, 
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to GitHub API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/{endpoint}"
        
        if params is None:
            params = {}
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 403:  # Rate limit exceeded
                    logger.warning("GitHub API rate limit exceeded, waiting 600 seconds...")
                    await asyncio.sleep(600)  # Wait 10 minutes instead of 1 minute
                    return await self._make_request(endpoint, params)
                elif response.status == 404:
                    logger.warning(f"Repository not found: {endpoint}")
                    return {}
                else:
                    logger.error(f"GitHub API error: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"GitHub API request failed: {e}")
            return {}
    
    def _parse_repo_url(self, repo_url: str) -> tuple:
        """Parse GitHub repository URL to get owner and repo name"""
        if not repo_url:
            return None, None
        
        # Handle different URL formats
        if "github.com" in repo_url:
            parts = repo_url.split("github.com/")[-1].split("/")
            if len(parts) >= 2:
                return parts[0], parts[1].replace(".git", "")
        
        # Handle direct owner/repo format
        if "/" in repo_url and not repo_url.startswith("http"):
            parts = repo_url.split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
        
        return None, None
    
    async def get_repository_info(self, repo_url: str) -> Dict[str, Any]:
        """Get basic repository information"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return {}
        
        return await self._make_request(f"repos/{owner}/{repo}")
    
    async def get_repository_stats(self, repo_url: str) -> Dict[str, Any]:
        """Get comprehensive repository statistics"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return {}
        
        # Get basic repo info
        repo_info = await self.get_repository_info(repo_url)
        if not repo_info:
            return {}
        
        # Get additional stats
        contributors = await self.get_contributors(repo_url)
        commits = await self.get_commits(repo_url, days=30)
        issues = await self.get_issues(repo_url, state="all")
        pull_requests = await self.get_pull_requests(repo_url, state="all")
        languages = await self.get_languages(repo_url)
        releases = await self.get_releases(repo_url)
        
        return {
            "basic_info": repo_info,
            "contributors": contributors,
            "commits_30d": commits,
            "issues": issues,
            "pull_requests": pull_requests,
            "languages": languages,
            "releases": releases,
            "stats": {
                "stars": repo_info.get("stargazers_count", 0),
                "forks": repo_info.get("forks_count", 0),
                "watchers": repo_info.get("watchers_count", 0),
                "open_issues": repo_info.get("open_issues_count", 0),
                "size": repo_info.get("size", 0),
                "created_at": repo_info.get("created_at"),
                "updated_at": repo_info.get("updated_at"),
                "pushed_at": repo_info.get("pushed_at"),
                "language": repo_info.get("language"),
                "license": repo_info.get("license", {}).get("name") if repo_info.get("license") else None,
                "archived": repo_info.get("archived", False),
                "disabled": repo_info.get("disabled", False),
                "private": repo_info.get("private", False)
            }
        }
    
    async def get_contributors(self, repo_url: str) -> List[Dict]:
        """Get repository contributors"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        return await self._make_request(f"repos/{owner}/{repo}/contributors")
    
    async def get_commits(
        self, 
        repo_url: str, 
        days: int = 30,
        author: Optional[str] = None
    ) -> List[Dict]:
        """Get repository commits"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        params = {"since": since}
        
        if author:
            params["author"] = author
        
        return await self._make_request(f"repos/{owner}/{repo}/commits", params)
    
    async def get_commits_stats(self, repo_url: str, days: int = 30) -> Dict[str, Any]:
        """Get commit statistics"""
        commits = await self.get_commits(repo_url, days)
        
        if not commits:
            return {
                "total_commits": 0,
                "unique_contributors": 0,
                "commits_by_day": {},
                "commits_by_contributor": {},
                "avg_commits_per_day": 0
            }
        
        # Analyze commits
        contributors = set()
        commits_by_day = {}
        commits_by_contributor = {}
        
        for commit in commits:
            # Count unique contributors
            author = commit.get("commit", {}).get("author", {}).get("name")
            if author:
                contributors.add(author)
                commits_by_contributor[author] = commits_by_contributor.get(author, 0) + 1
            
            # Count commits by day
            date = commit.get("commit", {}).get("author", {}).get("date", "")[:10]
            if date:
                commits_by_day[date] = commits_by_day.get(date, 0) + 1
        
        return {
            "total_commits": len(commits),
            "unique_contributors": len(contributors),
            "commits_by_day": commits_by_day,
            "commits_by_contributor": commits_by_contributor,
            "avg_commits_per_day": len(commits) / days if days > 0 else 0
        }
    
    async def get_issues(
        self, 
        repo_url: str, 
        state: str = "open",
        labels: Optional[str] = None,
        per_page: int = 100
    ) -> List[Dict]:
        """Get repository issues"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        params = {
            "state": state,
            "per_page": per_page
        }
        
        if labels:
            params["labels"] = labels
        
        return await self._make_request(f"repos/{owner}/{repo}/issues", params)
    
    async def get_pull_requests(
        self, 
        repo_url: str, 
        state: str = "open",
        per_page: int = 100
    ) -> List[Dict]:
        """Get repository pull requests"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        params = {
            "state": state,
            "per_page": per_page
        }
        
        return await self._make_request(f"repos/{owner}/{repo}/pulls", params)
    
    async def get_languages(self, repo_url: str) -> Dict[str, int]:
        """Get repository languages"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return {}
        
        return await self._make_request(f"repos/{owner}/{repo}/languages")
    
    async def get_releases(self, repo_url: str, per_page: int = 100) -> List[Dict]:
        """Get repository releases"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        params = {"per_page": per_page}
        return await self._make_request(f"repos/{owner}/{repo}/releases", params)
    
    async def get_branches(self, repo_url: str) -> List[Dict]:
        """Get repository branches"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        return await self._make_request(f"repos/{owner}/{repo}/branches")
    
    async def get_tags(self, repo_url: str) -> List[Dict]:
        """Get repository tags"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        return await self._make_request(f"repos/{owner}/{repo}/tags")
    
    async def get_workflows(self, repo_url: str) -> List[Dict]:
        """Get repository workflows"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        return await self._make_request(f"repos/{owner}/{repo}/actions/workflows")
    
    async def get_workflow_runs(
        self, 
        repo_url: str, 
        workflow_id: Optional[str] = None,
        per_page: int = 100
    ) -> Dict[str, Any]:
        """Get workflow runs"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return {}
        
        endpoint = f"repos/{owner}/{repo}/actions/runs"
        if workflow_id:
            endpoint = f"repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
        
        params = {"per_page": per_page}
        return await self._make_request(endpoint, params)
    
    async def get_community_health(self, repo_url: str) -> Dict[str, Any]:
        """Get community health metrics"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return {}
        
        return await self._make_request(f"repos/{owner}/{repo}/community/profile")
    
    async def get_traffic_views(self, repo_url: str) -> Dict[str, Any]:
        """Get repository traffic views"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return {}
        
        return await self._make_request(f"repos/{owner}/{repo}/traffic/views")
    
    async def get_traffic_clones(self, repo_url: str) -> Dict[str, Any]:
        """Get repository traffic clones"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return {}
        
        return await self._make_request(f"repos/{owner}/{repo}/traffic/clones")
    
    async def get_traffic_popular_paths(self, repo_url: str) -> List[Dict]:
        """Get popular paths"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        return await self._make_request(f"repos/{owner}/{repo}/traffic/popular/paths")
    
    async def get_traffic_popular_referrers(self, repo_url: str) -> List[Dict]:
        """Get popular referrers"""
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            return []
        
        return await self._make_request(f"repos/{owner}/{repo}/traffic/popular/referrers")
    
    async def get_github_metrics(self, repo_url: str) -> Dict[str, Any]:
        """Get comprehensive GitHub metrics for analysis"""
        if not repo_url:
            return {}
        
        try:
            # Get basic repository stats
            repo_stats = await self.get_repository_stats(repo_url)
            if not repo_stats:
                return {}
            
            # Get commit statistics
            commits_stats = await self.get_commits_stats(repo_url, days=30)
            
            # Get issues and PRs
            open_issues = await self.get_issues(repo_url, state="open")
            closed_issues = await self.get_issues(repo_url, state="closed")
            open_prs = await self.get_pull_requests(repo_url, state="open")
            closed_prs = await self.get_pull_requests(repo_url, state="closed")
            
            # Get community health
            community_health = await self.get_community_health(repo_url)
            
            # Get traffic data (if available)
            traffic_views = await self.get_traffic_views(repo_url)
            traffic_clones = await self.get_traffic_clones(repo_url)
            
            # Calculate metrics
            basic_info = repo_stats.get("basic_info", {})
            contributors = repo_stats.get("contributors", [])
            
            # Calculate activity scores
            total_commits_30d = commits_stats.get("total_commits", 0)
            unique_contributors_30d = commits_stats.get("unique_contributors", 0)
            total_contributors = len(contributors)
            
            # Calculate PR merge rate
            total_prs = len(open_prs) + len(closed_prs)
            merged_prs = len([pr for pr in closed_prs if pr.get("merged_at")])
            pr_merge_rate = (merged_prs / total_prs * 100) if total_prs > 0 else 0
            
            # Calculate issue resolution rate
            total_issues = len(open_issues) + len(closed_issues)
            resolved_issues = len(closed_issues)
            issue_resolution_rate = (resolved_issues / total_issues * 100) if total_issues > 0 else 0
            
            # Calculate code quality score
            code_quality_score = 0
            if basic_info.get("license"):
                code_quality_score += 10
            if not basic_info.get("archived", False):
                code_quality_score += 10
            if total_commits_30d > 0:
                code_quality_score += 20
            if unique_contributors_30d > 1:
                code_quality_score += 20
            if pr_merge_rate > 50:
                code_quality_score += 20
            if issue_resolution_rate > 50:
                code_quality_score += 20
            
            return {
                "repository_info": {
                    "name": basic_info.get("name", ""),
                    "full_name": basic_info.get("full_name", ""),
                    "description": basic_info.get("description", ""),
                    "url": basic_info.get("html_url", ""),
                    "language": basic_info.get("language", ""),
                    "license": basic_info.get("license", {}).get("name") if basic_info.get("license") else None,
                    "created_at": basic_info.get("created_at"),
                    "updated_at": basic_info.get("updated_at"),
                    "pushed_at": basic_info.get("pushed_at")
                },
                "activity_metrics": {
                    "stars": basic_info.get("stargazers_count", 0),
                    "forks": basic_info.get("forks_count", 0),
                    "watchers": basic_info.get("watchers_count", 0),
                    "commits_24h": commits_stats.get("commits_by_day", {}).get(
                        datetime.utcnow().strftime("%Y-%m-%d"), 0
                    ),
                    "commits_7d": sum(commits_stats.get("commits_by_day", {}).values()),
                    "commits_30d": total_commits_30d,
                    "commits_90d": total_commits_30d * 3,  # Estimate
                    "active_contributors_30d": unique_contributors_30d,
                    "total_contributors": total_contributors,
                    "avg_commits_per_day": commits_stats.get("avg_commits_per_day", 0)
                },
                "development_metrics": {
                    "open_issues": len(open_issues),
                    "closed_issues_7d": len([i for i in closed_issues 
                                           if datetime.fromisoformat(i.get("closed_at", "").replace("Z", "+00:00")) 
                                           > datetime.utcnow() - timedelta(days=7)]),
                    "open_prs": len(open_prs),
                    "merged_prs_7d": len([pr for pr in closed_prs 
                                        if pr.get("merged_at") and 
                                        datetime.fromisoformat(pr.get("merged_at", "").replace("Z", "+00:00")) 
                                        > datetime.utcnow() - timedelta(days=7)]),
                    "closed_prs_7d": len([pr for pr in closed_prs 
                                        if datetime.fromisoformat(pr.get("closed_at", "").replace("Z", "+00:00")) 
                                        > datetime.utcnow() - timedelta(days=7)]),
                    "pr_merge_rate": pr_merge_rate,
                    "issue_resolution_time": 0,  # Would need more complex calculation
                    "bug_report_ratio": 0,  # Would need to analyze issue labels
                    "code_quality_score": code_quality_score,
                    "test_coverage": 0  # Would need to integrate with coverage tools
                },
                "community_metrics": {
                    "stars_change_7d": 0,  # Would need historical data
                    "watch_count": basic_info.get("subscribers_count", 0),
                    "primary_language": basic_info.get("language", ""),
                    "languages_distribution": repo_stats.get("languages", {}),
                    "external_contributors": max(0, total_contributors - 1),  # Estimate
                    "core_team_activity": 0  # Would need to identify core team
                },
                "traffic_metrics": {
                    "views_14d": traffic_views.get("count", 0) if traffic_views else 0,
                    "unique_visitors_14d": traffic_views.get("uniques", 0) if traffic_views else 0,
                    "clones_14d": traffic_clones.get("count", 0) if traffic_clones else 0,
                    "unique_cloners_14d": traffic_clones.get("uniques", 0) if traffic_clones else 0
                },
                "community_health": community_health,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting GitHub metrics for {repo_url}: {e}")
            return {}
    
    async def search_repositories(
        self, 
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 100
    ) -> Dict[str, Any]:
        """Search repositories"""
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page
        }
        
        return await self._make_request("search/repositories", params)
    
    async def get_user_repositories(
        self, 
        username: str,
        type: str = "all",
        sort: str = "updated",
        per_page: int = 100
    ) -> List[Dict]:
        """Get user repositories"""
        params = {
            "type": type,
            "sort": sort,
            "per_page": per_page
        }
        
        return await self._make_request(f"users/{username}/repos", params)
    
    async def get_organization_repositories(
        self, 
        org: str,
        type: str = "all",
        sort: str = "updated",
        per_page: int = 100
    ) -> List[Dict]:
        """Get organization repositories"""
        params = {
            "type": type,
            "sort": sort,
            "per_page": per_page
        }
        
        return await self._make_request(f"orgs/{org}/repos", params)
