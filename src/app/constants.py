from pathlib import Path

DEFAULT_SOURCE_FILE_ENCODING = "utf-8"

SERENA_FILE_ENCODING = "utf-8"

SERENA_LOG_FORMAT = "%(levelname)-5s %(asctime)-15s [%(threadName)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
SERENA_LOG_LEVEL = 20
TOOL_TIMEOUT = 240
TRACE_LSP_COMMUNICATION = True
LS_SPECIFIC_SETTINGS = {}
DEFAULT_MAX_TOOL_ANSWER_CHARS = 150000
DEFAULT_TOOL_TIMEOUT: float = 240
PROJECT_ROOT_PATH = "/home/user"  # /home/user/nextjs
IGNORED_PATHS: list = []
IGNORE_ALL_FILES_IN_GITIGNORE: bool = True

SERENA_MANAGED_DIR_NAME = ".serena"
_serena_in_home_managed_dir = Path.home() / ".serena"

SERENA_MANAGED_DIR_IN_HOME = str(_serena_in_home_managed_dir)
# IGNORED_BY_DEFAULT = [
#     ".serena",
# ]
