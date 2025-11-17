"""Base classes and utilities for tool management and execution."""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar

from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata
from sensai.util import logging
from sensai.util.string import dict_string

from app.constants import (
    DEFAULT_MAX_TOOL_ANSWER_CHARS,
    DEFAULT_SOURCE_FILE_ENCODING,
    DEFAULT_TOOL_TIMEOUT,
    IGNORE_ALL_FILES_IN_GITIGNORE,
    IGNORED_PATHS,
    LS_SPECIFIC_SETTINGS,
    PROJECT_ROOT_PATH,
    SERENA_FILE_ENCODING,
    SERENA_LOG_LEVEL,
    TOOL_TIMEOUT,
    TRACE_LSP_COMMUNICATION,
)
from app.solidlsp.ls_config import Language
from app.solidlsp.ls_exceptions import SolidLSPException
from app.solidlsp.ls_utils import FileUtils
from app.utils.class_decorators import singleton
from app.utils.file_system import GitignoreParser
from app.utils.inspection import iter_subclasses
from app.utils.ls_manager import LanguageServerFactory, LanguageServerManager
from app.utils.symbol import LanguageServerSymbolRetriever
from app.utils.task_executor import TaskExecutor
from app.utils.text_utils import MatchedConsecutiveLines

if TYPE_CHECKING:
    from app.utils.code_editor import CodeEditor

log = logging.getLogger(__name__)
T = TypeVar("T")
SUCCESS_RESULT = "OK"


class ApplyMethodProtocol(Protocol):
    """Protocol defining the signature for tool apply methods."""

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool with given arguments and return a string result."""
        ...


class Component(ABC):
    """Base component providing task execution and language server management."""

    def __init__(self, languages: list[Language] | None = None):
        """Initialize component with optional language support.

        Args:
            languages: List of programming languages this component supports.
        """
        self.languages = languages or []
        self.language_server_manager: LanguageServerManager | None = None
        self._ignored_patterns = self._initialize_ignored_patterns()
        self._task_executor = TaskExecutor("SerenaAgentTaskExecutor")

    def _initialize_ignored_patterns(self) -> list[str]:
        """Initialize and return the list of file patterns to ignore.

        Returns:
            Combined list of explicit and gitignore patterns.
        """
        patterns = list(IGNORED_PATHS)

        if patterns:
            log.info(f"Using {len(patterns)} ignored paths from explicit configuration")
            log.debug(f"Ignored paths: {patterns}")

        if IGNORE_ALL_FILES_IN_GITIGNORE:
            gitignore_parser = GitignoreParser(self.get_project_root())
            for spec in gitignore_parser.get_ignore_specs():
                log.debug(f"Adding {len(spec.patterns)} patterns from {spec.file_path}")
                patterns.extend(spec.patterns)

        return patterns

    def get_project_root(self) -> Path:
        """Get the project root directory path."""
        return PROJECT_ROOT_PATH

    # Task Execution Methods

    def issue_task(
        self,
        task: Callable[[], T],
        name: str | None = None,
        logged: bool = True,
        timeout: float | None = None,
    ) -> TaskExecutor.Task[T]:
        """Issue a task for asynchronous execution.

        Tasks are executed sequentially in the order they are issued.

        Args:
            task: The callable to execute.
            name: Optional task name for logging (defaults to function name).
            logged: Whether to log task management (errors always logged).
            timeout: Maximum wait time in seconds (None for indefinite).

        Returns:
            Task object for accessing the future result.
        """
        return self._task_executor.issue_task(
            task, name=name, logged=logged, timeout=timeout
        )

    def execute_task(
        self,
        task: Callable[[], T],
        name: str | None = None,
        logged: bool = True,
        timeout: float | None = None,
    ) -> T:
        """Execute a task synchronously and return its result.

        Args:
            task: The callable to execute.
            name: Optional task name for logging (defaults to function name).
            logged: Whether to log task management (errors always logged).
            timeout: Maximum wait time in seconds (None for indefinite).

        Returns:
            The result of the task execution.
        """
        return self._task_executor.execute_task(
            task, name=name, logged=logged, timeout=timeout
        )

    def get_current_tasks(self) -> list[TaskExecutor.TaskInfo]:
        """Get list of currently running or queued tasks.

        Returns:
            List of thread-safe TaskInfo objects in execution order.
        """
        return self._task_executor.get_current_tasks()

    def get_last_executed_task(self) -> TaskExecutor.TaskInfo | None:
        """Get information about the last executed task.

        Returns:
            TaskInfo for the last executed task, or None if no tasks executed.
        """
        return self._task_executor.get_last_executed_task()

    # Language Server Management

    def reset_language_server_manager(self) -> None:
        """Initialize or reset the language server manager for the current project."""
        ls_timeout = self._calculate_ls_timeout()

        self.create_language_server_manager(
            log_level=SERENA_LOG_LEVEL,
            ls_timeout=ls_timeout,
            trace_lsp_communication=TRACE_LSP_COMMUNICATION,
            ls_specific_settings=LS_SPECIFIC_SETTINGS,
        )

    def _calculate_ls_timeout(self) -> float | None:
        """Calculate appropriate language server timeout based on tool timeout.

        Returns:
            Language server timeout in seconds, or None for no timeout.

        Raises:
            ValueError: If tool timeout is less than 10 seconds.
        """
        tool_timeout = TOOL_TIMEOUT

        if tool_timeout is None or tool_timeout < 0:
            return None

        if tool_timeout < 10:
            raise ValueError(
                f"Tool timeout must be at least 10 seconds, got {tool_timeout}"
            )

        # LS timeout should be 5 seconds less than tool timeout
        return tool_timeout - 5

    def get_language_server_manager_or_raise(self) -> LanguageServerManager:
        """Get the language server manager or raise an error if not initialized.

        Returns:
            The initialized language server manager.

        Raises:
            Exception: If language server manager is not initialized.
        """
        if self.language_server_manager is None:
            raise Exception(
                "Language server manager not initialized. This indicates a problem "
                "during project activation. Please inspect Serena's logs to determine "
                "the issue. IMPORTANT: Wait for further instructions before continuing!"
            )
        return self.language_server_manager

    def create_language_server_manager(
        self,
        log_level: int = logging.INFO,
        ls_timeout: float | None = DEFAULT_TOOL_TIMEOUT - 5,
        trace_lsp_communication: bool = False,
        ls_specific_settings: dict[Language, Any] | None = None,
    ) -> LanguageServerManager:
        """Create and initialize the language server manager.

        Starts one language server per configured programming language.
        If a manager already exists, it will be stopped first.

        Args:
            log_level: Logging level for the language server.
            ls_timeout: Timeout for language server operations.
            trace_lsp_communication: Whether to trace LSP protocol messages.
            ls_specific_settings: Optional language-specific server configurations.

        Returns:
            The newly created and initialized language server manager.
        """
        if self.language_server_manager is not None:
            log.info("Stopping existing language server manager...")
            self.language_server_manager.stop_all()
            self.language_server_manager = None

        factory = LanguageServerFactory(
            project_root=self.get_project_root(),
            encoding=DEFAULT_SOURCE_FILE_ENCODING,
            ignored_patterns=self._ignored_patterns,
            ls_timeout=ls_timeout,
            ls_specific_settings=ls_specific_settings,
            log_level=log_level,
            trace_lsp_communication=trace_lsp_communication,
        )

        self.language_server_manager = LanguageServerManager.from_languages(
            self.languages, factory
        )
        return self.language_server_manager

    def create_language_server_symbol_retriever(self) -> LanguageServerSymbolRetriever:
        """Create a symbol retriever using the current language server manager.

        Returns:
            A new LanguageServerSymbolRetriever instance.
        """
        manager = self.get_language_server_manager_or_raise()
        return LanguageServerSymbolRetriever(manager)

    # File Operations

    def read_file(self, relative_path: str) -> str:
        """Read a file relative to the project root.

        Args:
            relative_path: Path relative to project root.

        Returns:
            The file contents as a string.
        """
        abs_path = Path(self.get_project_root()) / relative_path
        return FileUtils.read_file(str(abs_path), SERENA_FILE_ENCODING)

    def retrieve_content_around_line(
        self,
        relative_file_path: str,
        line: int,
        context_lines_before: int = 0,
        context_lines_after: int = 0,
    ) -> MatchedConsecutiveLines:
        """Retrieve file content around a specific line with context.

        Args:
            relative_file_path: Path relative to project root.
            line: Line number to center the content around.
            context_lines_before: Number of lines to include before target line.
            context_lines_after: Number of lines to include after target line.

        Returns:
            Container with the requested lines and context.
        """
        file_contents = self.read_file(relative_file_path)
        return MatchedConsecutiveLines.from_file_contents(
            file_contents,
            line=line,
            context_lines_before=context_lines_before,
            context_lines_after=context_lines_after,
            source_file_path=relative_file_path,
        )

    def create_code_editor(self) -> "CodeEditor":
        """Create a code editor instance with language server support.

        Returns:
            A new CodeEditor instance.
        """
        from app.utils.code_editor import LanguageServerCodeEditor

        return LanguageServerCodeEditor(self.create_language_server_symbol_retriever())


class Tool(Component):
    """Base class for all tools with apply method and metadata support."""

    def __init__(
        self,
        languages: list[Language] | None = None,
        ls_manager: LanguageServerManager | None = None,
    ):
        """Initialize tool with optional languages and language server manager.

        Args:
            languages: List of supported programming languages.
            ls_manager: Optional pre-existing language server manager.
        """
        super().__init__(languages=languages)
        if ls_manager is not None:
            self.language_server_manager = ls_manager

    @classmethod
    def get_name_from_cls(cls) -> str:
        """Derive tool name from class name in snake_case.

        Strips 'Tool' suffix and converts CamelCase to snake_case.

        Returns:
            The tool name as a string.
        """
        name = cls.__name__
        if name.endswith("Tool"):
            name = name[:-4]

        # Convert CamelCase to snake_case
        name = "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip(
            "_"
        )

        return name

    def get_name(self) -> str:
        """Get the name of this tool instance.

        Returns:
            The tool name.
        """
        return self.get_name_from_cls()

    @classmethod
    def get_tool_description(cls) -> str:
        """Get the tool description from the class docstring.

        Returns:
            The stripped docstring, or empty string if none exists.
        """
        docstring = cls.__doc__
        return docstring.strip() if docstring else ""

    def get_apply_fn(self) -> ApplyMethodProtocol:
        """Get the apply method for this tool.

        Returns:
            The apply method callable.

        Raises:
            RuntimeError: If apply method is not defined.
        """
        apply_fn = getattr(self, "apply", None)
        if apply_fn is None:
            raise RuntimeError(
                f"apply method not defined in {self}. Did you forget to implement it?"
            )
        return apply_fn

    @classmethod
    def _get_apply_method(cls) -> Callable:
        """Internal method to get the apply method from the class.

        Returns:
            The apply method.

        Raises:
            AttributeError: If apply method is not found.
        """
        # Try __dict__ first for dynamic changes
        if "apply" in cls.__dict__:
            return cls.__dict__["apply"]

        # Fall back to getattr for inherited methods
        apply_fn = getattr(cls, "apply", None)
        if apply_fn is None:
            raise AttributeError(
                f"apply method not defined in {cls}. Did you forget to implement it?"
            )
        return apply_fn

    @classmethod
    def get_apply_docstring_from_cls(cls) -> str:
        """Get the apply method's docstring from the class.

        Used for creating MCP tools without serialization issues.

        Returns:
            The stripped docstring.

        Raises:
            AttributeError: If apply method has no docstring.
        """
        apply_fn = cls._get_apply_method()
        docstring = apply_fn.__doc__

        if not docstring:
            raise AttributeError(
                f"apply method has no docstring in {cls}. Please add documentation."
            )

        return docstring.strip()

    def get_apply_docstring(self) -> str:
        """Get the apply method's docstring for this tool instance.

        Returns:
            The apply method docstring.
        """
        return self.get_apply_docstring_from_cls()

    @classmethod
    def get_apply_fn_metadata_from_cls(cls) -> FuncMetadata:
        """Get metadata for the apply method from the class.

        Used for creating MCP tools without serialization issues.

        Returns:
            Function metadata for the apply method.
        """
        apply_fn = cls._get_apply_method()
        return func_metadata(apply_fn, skip_names=["self", "cls"])

    def get_apply_fn_metadata(self) -> FuncMetadata:
        """Get metadata for the apply method of this tool instance.

        Returns:
            Function metadata for the apply method.
        """
        return self.get_apply_fn_metadata_from_cls()

    def _log_tool_application(self, frame: Any) -> None:
        """Log the tool application with its parameters.

        Args:
            frame: The stack frame containing local variables.
        """
        ignored_params = {"self", "log_call", "catch_exceptions", "args", "apply_fn"}
        params = {}

        for param, value in frame.f_locals.items():
            if param in ignored_params:
                continue
            if param == "kwargs":
                params.update(value)
            else:
                params[param] = value

        log.info(f"{self.get_name_from_cls()}: {dict_string(params)}")

    def _limit_length(self, result: str, max_answer_chars: int) -> str:
        """Limit the result string length to prevent oversized responses.

        Args:
            result: The result string to potentially truncate.
            max_answer_chars: Maximum allowed characters (-1 for default).

        Returns:
            Either the original result or an error message if too long.

        Raises:
            ValueError: If max_answer_chars is invalid.
        """
        if max_answer_chars == -1:
            max_answer_chars = DEFAULT_MAX_TOOL_ANSWER_CHARS

        if max_answer_chars <= 0:
            raise ValueError(
                f"max_answer_chars must be positive or -1 (default), got: {max_answer_chars}"
            )

        if (n_chars := len(result)) > max_answer_chars:
            return (
                f"The answer is too long ({n_chars} characters). "
                f"Please try a more specific query or increase max_answer_chars."
            )

        return result

    def _execute_apply_with_retry(self, apply_fn: Callable, kwargs: dict) -> str:
        """Execute the apply function with automatic retry on language server termination.

        Args:
            apply_fn: The apply function to execute.
            kwargs: Keyword arguments for the apply function.

        Returns:
            The result of the apply function.

        Raises:
            SolidLSPException: If the error is not recoverable.
        """
        try:
            return apply_fn(**kwargs)
        except SolidLSPException as e:
            if not e.is_language_server_terminated():
                raise

            affected_language = e.get_affected_language()
            if affected_language is None:
                log.error(
                    f"Language server terminated ({e}), but affected language unknown. "
                    "Not retrying."
                )
                raise

            log.error(
                f"Language server terminated while executing tool ({e}). "
                "Restarting and retrying..."
            )
            self.get_language_server_manager_or_raise().restart_language_server(
                affected_language
            )
            return apply_fn(**kwargs)

    def _save_language_server_caches(self) -> None:
        """Save all language server caches, logging any errors."""
        ls_manager = self.language_server_manager
        if ls_manager is None:
            return

        try:
            ls_manager.save_all_caches()
        except Exception as e:
            log.error(f"Error saving language server cache: {e}")

    def apply_ex(
        self, log_call: bool = True, catch_exceptions: bool = True, **kwargs
    ) -> str:
        """Apply the tool with logging and exception handling.

        Args:
            log_call: Whether to log the tool call and result.
            catch_exceptions: Whether to catch and return exceptions as strings.
            **kwargs: Arguments to pass to the apply method.

        Returns:
            The result of the tool application, or error message on failure.
        """

        def task() -> str:
            apply_fn = self.get_apply_fn()

            if log_call:
                self._log_tool_application(inspect.currentframe())

            result = self._execute_apply_with_retry(apply_fn, kwargs)

            if log_call:
                log.info(f"Result: {result}")

            self._save_language_server_caches()

            return result

        try:
            task_exec = self.issue_task(task, name=self.__class__.__name__)
            return task_exec.result(timeout=TOOL_TIMEOUT)
        except Exception as e:
            msg = f"Error: {e.__class__.__name__} - {e}"
            log.error(msg)
            return msg


@dataclass(kw_only=True)
class RegisteredTool:
    """Container for registered tool information."""

    tool_class: type[Tool]
    tool_name: str


@singleton
class ToolRegistry:
    """Singleton registry for managing all available tools."""

    def __init__(self) -> None:
        """Initialize the registry by discovering all Tool subclasses."""
        self._tool_dict: dict[str, RegisteredTool] = {}

        for cls in iter_subclasses(Tool):
            if not cls.__module__.startswith("app.tools"):
                continue

            name = cls.get_name_from_cls()
            if name in self._tool_dict:
                raise ValueError(
                    f"Duplicate tool name '{name}' found. Tool classes must have unique names."
                )

            self._tool_dict[name] = RegisteredTool(tool_class=cls, tool_name=name)

    def get_tool_class_by_name(self, tool_name: str) -> type[Tool]:
        """Get a tool class by its name.

        Args:
            tool_name: The name of the tool.

        Returns:
            The tool class.

        Raises:
            KeyError: If tool name is not registered.
        """
        return self._tool_dict[tool_name].tool_class

    def get_all_tool_classes(self) -> list[type[Tool]]:
        """Get all registered tool classes.

        Returns:
            List of all tool classes.
        """
        return [t.tool_class for t in self._tool_dict.values()]

    def get_tool_names(self) -> list[str]:
        """Get all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tool_dict.keys())

    def is_valid_tool_name(self, tool_name: str) -> bool:
        """Check if a tool name is registered.

        Args:
            tool_name: The name to check.

        Returns:
            True if the tool name is registered, False otherwise.
        """
        return tool_name in self._tool_dict


class AvailableTools:
    """Container for managing a collection of available tool instances."""

    def __init__(self, tools: list[Tool]):
        """Initialize with a list of tool instances.

        Args:
            tools: List of available tool instances.
        """
        self.tools = tools
        self.tool_names = [tool.get_name_from_cls() for tool in tools]
        self.tool_marker_names: set[str] = set()

    def __len__(self) -> int:
        """Get the number of available tools.

        Returns:
            The count of tools.
        """
        return len(self.tools)


class TSProject(Component):
    """TypeScript project manager with integrated language server and tool collection."""

    def __init__(self):
        """Initialize the TypeScript project with language server and all tools."""
        super().__init__(languages=[Language.TYPESCRIPT])

        log.debug("Initializing TSProject with language server manager")
        manager = self.create_language_server_manager()
        log.debug(
            f"Language server manager initialized: {self.language_server_manager}"
        )

        # Initialize all tools except TSProject itself
        self._all_tools = {
            tool_class: tool_class(languages=None, ls_manager=manager)
            for tool_class in ToolRegistry().get_all_tool_classes()
            if tool_class is not TSProject
        }

        self._exposed_tools = AvailableTools(list(self._all_tools.values()))
        log.debug(f"Exposed tools created: {len(self._exposed_tools)} tools available")
