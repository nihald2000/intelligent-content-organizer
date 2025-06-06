import logging
import asyncio
import functools
from typing import Any, Callable, Dict, List, Optional
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def async_timer(func: Callable) -> Callable:
    """Decorator to time async function execution"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"{func.__name__} completed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} failed after {end_time - start_time:.3f}s: {str(e)}")
            raise
    return wrapper

def retry_async(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry async functions with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt} failed: {str(e)}")
                    logger.info(f"Retrying in {current_delay}s...")
                    
                    await asyncio.sleep(current_delay)
                    attempt += 1
                    current_delay *= backoff
            
        return wrapper
    return decorator

class MCPToolResponse:
    """Standardized response format for MCP tools"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None, 
                 metadata: Dict[str, Any] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        result = {
            "success": self.success,
            "timestamp": self.timestamp
        }
        
        if self.success:
            result["data"] = self.data
        else:
            result["error"] = self.error
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def success_response(cls, data: Any, metadata: Dict[str, Any] = None):
        """Create a success response"""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_response(cls, error: str, metadata: Dict[str, Any] = None):
        """Create an error response"""
        return cls(success=False, error=error, metadata=metadata)

def validate_required_params(params: Dict[str, Any], required: List[str]) -> Optional[str]:
    """Validate that required parameters are present"""
    missing = []
    for param in required:
        if param not in params or params[param] is None:
            missing.append(param)
    
    if missing:
        return f"Missing required parameters: {', '.join(missing)}"
    
    return None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem, Path(filename).suffix
        max_name_len = 255 - len(ext)
        filename = name[:max_name_len] + ext
    
    # Ensure not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename

def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    
    if add_ellipsis and max_length > 3:
        return text[:max_length - 3] + "..."
    else:
        return text[:max_length]

def extract_file_info(file_path: str) -> Dict[str, Any]:
    """Extract information about a file"""
    try:
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir()
        }
    except Exception as e:
        return {"error": str(e)}

async def batch_process(items: List[Any], processor: Callable, batch_size: int = 10, 
                       max_concurrent: int = 5) -> List[Any]:
    """Process items in batches with concurrency control"""
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item):
        async with semaphore:
            return await processor(item)
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = [process_item(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)
    
    return results

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Calculate estimated reading time in minutes"""
    word_count = len(text.split())
    return max(1, round(word_count / words_per_minute))

class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.completed_items = 0
        self.description = description
        self.start_time = time.time()
        self.errors = []
    
    def update(self, completed: int = 1, error: str = None):
        """Update progress"""
        self.completed_items += completed
        if error:
            self.errors.append(error)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.completed_items / self.total_items) * 100 if self.total_items > 0 else 0
        
        # Estimate remaining time
        if self.completed_items > 0:
            avg_time_per_item = elapsed_time / self.completed_items
            remaining_items = self.total_items - self.completed_items
            estimated_remaining_time = avg_time_per_item * remaining_items
        else:
            estimated_remaining_time = 0
        
        return {
            "description": self.description,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "progress_percent": round(progress_percent, 1),
            "elapsed_time_seconds": round(elapsed_time, 1),
            "estimated_remaining_seconds": round(estimated_remaining_time, 1),
            "errors_count": len(self.errors),
            "errors": self.errors[-5:] if self.errors else []  # Last 5 errors
        }
    
    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.completed_items >= self.total_items

def load_json_config(config_path: str, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load configuration from JSON file with fallback to defaults"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_path} not found, using defaults")
        return default_config or {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_path}: {str(e)}")
        return default_config or {}

def save_json_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to JSON file"""
    try:
        # Create directory if it doesn't exist
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {str(e)}")
        return False

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make a call"""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # Check if we can make a new call
        if len(self.calls) >= self.max_calls:
            # Wait until we can make a call
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.acquire()  # Recursive call after waiting
        
        # Record this call
        self.calls.append(now)

def escape_markdown(text: str) -> str:
    """Escape markdown special characters"""
    import re
    
    # Characters that need escaping in markdown
    markdown_chars = r'([*_`\[\]()#+\-!\\])'
    return re.sub(markdown_chars, r'\\\1', text)

def create_error_summary(errors: List[Exception]) -> str:
    """Create a summary of multiple errors"""
    if not errors:
        return "No errors"
    
    error_counts = {}
    for error in errors:
        error_type = type(error).__name__
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    summary_parts = []
    for error_type, count in error_counts.items():
        if count == 1:
            summary_parts.append(f"1 {error_type}")
        else:
            summary_parts.append(f"{count} {error_type}s")
    
    return f"Encountered {len(errors)} total errors: " + ", ".join(summary_parts)

async def safe_execute(func: Callable, *args, default_return=None, **kwargs):
    """Safely execute a function and return default on error"""
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default_return

def get_content_preview(content: str, max_length: int = 200) -> str:
    """Get a preview of content for display"""
    if not content:
        return "No content"
    
    # Clean up whitespace
    content = ' '.join(content.split())
    
    if len(content) <= max_length:
        return content
    
    # Try to break at sentence boundary
    preview = content[:max_length]
    last_sentence_end = max(preview.rfind('.'), preview.rfind('!'), preview.rfind('?'))
    
    if last_sentence_end > max_length * 0.7:  # If we found a good breaking point
        return preview[:last_sentence_end + 1]
    else:
        # Break at word boundary
        last_space = preview.rfind(' ')
        if last_space > max_length * 0.7:
            return preview[:last_space] + "..."
        else:
            return preview + "..."

class MemoryUsageTracker:
    """Track memory usage of operations"""
    
    def __init__(self):
        self.start_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def get_usage_delta(self) -> float:
        """Get memory usage change since initialization"""
        current_memory = self._get_memory_usage()
        return current_memory - self.start_memory
    
    def log_usage(self, operation_name: str):
        """Log current memory usage for an operation"""
        delta = self.get_usage_delta()
        logger.info(f"{operation_name} memory delta: {delta:.1f} MB")