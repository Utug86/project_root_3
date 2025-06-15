import logging
import threading
from typing import List, Optional, Dict, Any
from pathlib import Path
import http.client
import json
from datetime import datetime, timedelta

# === CONFIGURATION ===

SERPER_API_KEY = '92c04c9d14f70d29047baa2dae0330ff475e0adb'
SERPER_HOST = 'google.serper.dev'
SERPER_ENDPOINT = '/search'
SEARCH_QUOTA_MONTHLY = 2500  # Maximum allowed searches per month

# === LOGGING ===

logger = logging.getLogger("utils.search_utils")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [search_utils] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# === QUOTA MANAGER ===

class MonthlyQuotaManager:
    """
    Thread-safe manager for monthly API quota.
    Tracks API calls and persists usage to a file.
    """
    def __init__(self, quota_limit: int, usage_file: Optional[Path] = None) -> None:
        self.quota_limit = quota_limit
        self.usage_file = usage_file or Path("./serper_quota_usage.json")
        self.lock = threading.Lock()
        self._init_quota_file()

    def _init_quota_file(self) -> None:
        if not self.usage_file.exists():
            self._reset_quota()

        # Clean up quota if month changed
        try:
            with self.usage_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            month = data.get("month")
            now = datetime.utcnow()
            current_month = now.strftime("%Y-%m")
            if month != current_month:
                self._reset_quota()
        except Exception as e:
            logger.warning(f"Quota file error: {e}, resetting quota.")
            self._reset_quota()

    def _reset_quota(self) -> None:
        now = datetime.utcnow()
        data = {
            "month": now.strftime("%Y-%m"),
            "count": 0,
            "reset_at": (now.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        }
        try:
            with self.usage_file.open("w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to reset quota: {e}")

    def can_use(self, n: int = 1) -> bool:
        with self.lock:
            try:
                with self.usage_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data["count"] + n <= self.quota_limit
            except Exception as e:
                logger.error(f"Quota read error: {e}")
                return False

    def increment(self, n: int = 1) -> None:
        with self.lock:
            try:
                with self.usage_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                data["count"] += n
                with self.usage_file.open("w", encoding="utf-8") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.error(f"Quota increment error: {e}")

    def get_remaining(self) -> int:
        with self.lock:
            try:
                with self.usage_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return max(0, self.quota_limit - data["count"])
            except Exception as e:
                logger.error(f"Quota read error: {e}")
                return 0

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            try:
                with self.usage_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "month": data.get("month"),
                    "used": data.get("count"),
                    "remaining": max(0, self.quota_limit - data.get("count", 0)),
                    "reset_at": data.get("reset_at")
                }
            except Exception as e:
                logger.error(f"Quota status error: {e}")
                return {"error": str(e)}

# Instantiate the quota manager as a singleton for module use
quota_manager = MonthlyQuotaManager(SEARCH_QUOTA_MONTHLY)

# === SEARCH API ===

class SearchAPIError(Exception):
    """Raised for errors in the search API."""

def web_search(
    query: str,
    num_results: int = 8,
    location: Optional[str] = None,
    lang: str = "ru"
) -> List[str]:
    """
    Performs a Google search using Serper API with strict monthly quota control.

    Args:
        query (str): Search query.
        num_results (int, optional): Number of results to return (max 20 per API docs). Defaults to 8.
        location (str, optional): Location for localized search results.
        lang (str): Language code (default "ru").

    Returns:
        List[str]: List of formatted search results.

    Raises:
        SearchAPIError: If quota is exceeded or request fails.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string.")

    req_count = 1  # Each call counts as 1 API request
    if not quota_manager.can_use(req_count):
        logger.warning("Monthly search quota exceeded.")
        raise SearchAPIError("Monthly search quota exceeded.")

    # Prepare payload and headers
    payload = {
        "q": query,
        "hl": lang
    }
    if location:
        payload["location"] = location

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        conn = http.client.HTTPSConnection(SERPER_HOST, timeout=10)
        conn.request("POST", SERPER_ENDPOINT, body=json.dumps(payload), headers=headers)
        res = conn.getresponse()
        if res.status != 200:
            logger.error(f"Serper API HTTP error: {res.status} {res.reason}")
            raise SearchAPIError(f"Serper API HTTP error: {res.status} {res.reason}")
        raw_data = res.read()
        data = json.loads(raw_data.decode("utf-8"))
        results: List[str] = []

        # Serper API returns 'organic' field with results
        for entry in data.get("organic", [])[:num_results]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            snippet = entry.get("snippet", "")
            formatted = f"{title} ({link})\n{snippet}"
            results.append(formatted)

        quota_manager.increment(req_count)
        logger.info(f"web_search success: '{query}' | {len(results)} results | Quota used: {quota_manager.get_status()['used']}")
        return results
    except (http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.error(f"Serper API request error: {exc}")
        raise SearchAPIError(f"Serper API request error: {exc}") from exc
    except Exception as exc:
        logger.error(f"web_search unexpected error: {exc}")
        raise SearchAPIError(f"web_search unexpected error: {exc}") from exc
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_search_quota_status() -> Dict[str, Any]:
    """
    Returns current search quota status (used, remaining, reset time etc.).
    """
    return quota_manager.get_status()
