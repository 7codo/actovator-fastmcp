"""Base classes and utilities for tool management and execution."""

import inspect
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Protocol, Self, TypeVar
import pathspec

from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata
from sensai.util import logging
from sensai.util.string import dict_string
import os
from app.constants import (
    DEFAULT_MAX_TOOL_ANSWER_CHARS,
    DEFAULT_SOURCE_FILE_ENCODING,
    DEFAULT_TOOL_TIMEOUT,
    IGNORE_ALL_FILES_IN_GITIGNORE,
    IGNORED_BY_DEFAULT,
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
from app.utils.file_system import GitignoreParser, match_path
from app.utils.inspection import iter_subclasses
from app.utils.ls_manager import LanguageServerFactory, LanguageServerManager
from app.utils.symbol import LanguageServerSymbolRetriever
from app.utils.task_executor import TaskExecutor
from app.utils.text_utils import MatchedConsecutiveLines, search_files

if TYPE_CHECKING:
    from app.utils.code_editor import CodeEditor

log = logging.getLogger(__name__)
T = TypeVar("T")
SUCCESS_RESULT = "OK"
TTool = TypeVar("TTool", bound="Tool")


class ApplyMethodProtocol(Protocol):
    """Protocol defining the signature for tool apply methods."""

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool with given arguments and return a string result."""
        ...


class TaskExecutionMixin:
    """Handles async task execution"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_executor = TaskExecutor("SerenaAgentTaskExecutor")

    def issue_task(
        self,
        task: Callable[[], T],
        name: str | None = None,
        logged: bool = True,
        timeout: float | None = None,
    ) -> TaskExecutor.Task[T]:
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
        return self._task_executor.execute_task(
            task, name=name, logged=logged, timeout=timeout
        )

    def get_current_tasks(self) -> list[TaskExecutor.TaskInfo]:
        return self._task_executor.get_current_tasks()

    def get_last_executed_task(self) -> TaskExecutor.TaskInfo | None:
        return self._task_executor.get_last_executed_task()


class PathIgnoreMixin:
    """Path ignore pattern matching"""

    def _initialize_ignored_patterns(self) -> list[str]:
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

    def get_ignore_spec(self) -> pathspec.PathSpec:
        return self._ignore_spec

    def _is_ignored_relative_path(
        self, relative_path: str | Path, ignore_non_source_files: bool = True
    ) -> bool:
        if str(relative_path) in [".", ""]:
            return False
        abs_path = os.path.join(self.get_project_root(), relative_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"File {abs_path} not found, the ignore check cannot be performed"
            )
        is_file = os.path.isfile(abs_path)
        if is_file and ignore_non_source_files:
            is_file_in_supported_language = False
            for language in self.languages:
                fn_matcher = language.get_source_fn_matcher()
                if fn_matcher.is_relevant_filename(abs_path):
                    is_file_in_supported_language = True
                    break
            if not is_file_in_supported_language:
                return True

        rel_path = Path(relative_path)
        if len(rel_path.parts) > 0 and rel_path.parts[0] == ".git":
            return True

        # Ignore paths listed in ignored_by_default
        for default_ignore in IGNORED_BY_DEFAULT:
            # If any path part matches the ignore pattern
            if default_ignore in rel_path.parts:
                return True

        return match_path(
            str(relative_path),
            self.get_ignore_spec(),
            root_path=self.get_project_root(),
        )

    def is_ignored_path(
        self, path: str | Path, ignore_non_source_files: bool = False
    ) -> bool:
        path = Path(path)
        if path.is_absolute():
            try:
                relative_path = path.relative_to(self.get_project_root())
            except ValueError:
                log.warning(
                    f"Path {path} is not relative to the project root {self.get_project_root()} and was therefore ignored"
                )
                return True
        else:
            relative_path = path

        return self._is_ignored_relative_path(
            str(relative_path), ignore_non_source_files=ignore_non_source_files
        )

    def is_path_in_project(self, path: str | Path) -> bool:
        path = Path(path)
        _proj_root = Path(self.get_project_root())
        if not path.is_absolute():
            path = _proj_root / path
        path = path.resolve()
        return path.is_relative_to(_proj_root)

    def validate_relative_path(
        self, relative_path: str, require_not_ignored: bool = False
    ) -> None:
        if not self.is_path_in_project(relative_path):
            raise ValueError(
                f"{relative_path=} points to path outside of the repository root; cannot access for safety reasons"
            )
        if require_not_ignored:
            if self.is_ignored_path(relative_path):
                raise ValueError(
                    f"Path {relative_path} is ignored; cannot access for safety reasons"
                )

    def relative_path_exists(self, relative_path: str) -> bool:
        abs_path = Path(self.get_project_root()) / relative_path
        return abs_path.exists()


class FileOperationsMixin:
    """File reading and path validation"""

    def read_file(self, relative_path: str) -> str:
        abs_path = Path(self.get_project_root()) / relative_path
        return FileUtils.read_file(str(abs_path), SERENA_FILE_ENCODING)

    def retrieve_content_around_line(
        self,
        relative_file_path: str,
        line: int,
        context_lines_before: int = 0,
        context_lines_after: int = 0,
    ) -> MatchedConsecutiveLines:
        file_contents = self.read_file(relative_file_path)
        return MatchedConsecutiveLines.from_file_contents(
            file_contents,
            line=line,
            context_lines_before=context_lines_before,
            context_lines_after=context_lines_after,
            source_file_path=relative_file_path,
        )

    def gather_source_files(self, relative_path: str = "") -> list[str]:
        rel_file_paths = []
        start_path = os.path.join(self.get_project_root(), relative_path)
        if not os.path.exists(start_path):
            raise FileNotFoundError(f"Relative path {start_path} not found.")
        if os.path.isfile(start_path):
            return [relative_path]
        else:
            for root, dirs, files in os.walk(start_path, followlinks=True):
                dirs[:] = [
                    d for d in dirs if not self.is_ignored_path(os.path.join(root, d))
                ]
                for file in files:
                    abs_file_path = os.path.join(root, file)
                    try:
                        if not self.is_ignored_path(
                            abs_file_path, ignore_non_source_files=True
                        ):
                            try:
                                rel_file_path = os.path.relpath(
                                    abs_file_path, start=self.get_project_root()
                                )
                            except Exception:
                                log.warning(
                                    "Ignoring path '%s' because it appears to be outside of the project root (%s)",
                                    abs_file_path,
                                    self.get_project_root(),
                                )
                                continue
                            rel_file_paths.append(rel_file_path)
                    except FileNotFoundError:
                        log.warning(
                            f"File {abs_file_path} not found (possibly due it being a symlink), skipping it in request_parsed_files",
                        )
            return rel_file_paths

    def search_source_files_for_pattern(
        self,
        pattern: str,
        relative_path: str = "",
        context_lines_before: int = 0,
        context_lines_after: int = 0,
        paths_include_glob: str | None = None,
        paths_exclude_glob: str | None = None,
    ) -> list[MatchedConsecutiveLines]:
        relative_file_paths = self.gather_source_files(relative_path=relative_path)
        return search_files(
            relative_file_paths,
            pattern,
            root_path=self.get_project_root(),
            file_reader=self.read_file,
            context_lines_before=context_lines_before,
            context_lines_after=context_lines_after,
            paths_include_glob=paths_include_glob,
            paths_exclude_glob=paths_exclude_glob,
        )


class LanguageServerMixin(PathIgnoreMixin):
    """Manages language server lifecycle"""

    def __init__(self, *args, languages: list[Language] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.languages = languages or []
        self.language_server_manager: LanguageServerManager | None = None
        self._ignored_patterns = self._initialize_ignored_patterns()
        processed_patterns = []
        for pattern in set(self._ignored_patterns):
            pattern = pattern.replace(os.path.sep, "/")
            processed_patterns.append(pattern)
        self._ignore_spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, processed_patterns
        )

    def get_project_root(self) -> Path:
        return PROJECT_ROOT_PATH

    def reset_language_server_manager(self) -> None:
        ls_timeout = self._calculate_ls_timeout()
        self.create_language_server_manager(
            log_level=SERENA_LOG_LEVEL,
            ls_timeout=ls_timeout,
            trace_lsp_communication=TRACE_LSP_COMMUNICATION,
            ls_specific_settings=LS_SPECIFIC_SETTINGS,
        )

    def _calculate_ls_timeout(self) -> float | None:
        tool_timeout = TOOL_TIMEOUT
        if tool_timeout is None or tool_timeout < 0:
            return None
        if tool_timeout < 10:
            raise ValueError(
                f"Tool timeout must be at least 10 seconds, got {tool_timeout}"
            )
        return tool_timeout - 5

    def get_language_server_manager_or_raise(self) -> LanguageServerManager:
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
        if self.language_server_manager is not None:
            log.info("Stopping existing language server manager...")
            self.language_server_manager.stop_all()
            self.language_server_manager = None
        log.info("self._ignored_patterns %s", self._ignored_patterns)
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
        manager = self.get_language_server_manager_or_raise()
        return LanguageServerSymbolRetriever(manager)

    def _save_language_server_caches(self) -> None:
        ls_manager = self.language_server_manager
        if ls_manager is None:
            return
        try:
            ls_manager.save_all_caches()
        except Exception as e:
            log.error(f"Error saving language server cache: {e}")


class Component(TaskExecutionMixin, LanguageServerMixin, FileOperationsMixin, ABC):
    """Base component providing task execution, language server management, and file operations."""

    def __init__(self, languages: list[Language] | None = None):
        super().__init__(languages=languages)
        self._all_tools: dict[type[Tool], Tool] = {}

    def create_code_editor(self) -> "CodeEditor":
        from app.utils.code_editor import LanguageServerCodeEditor

        return LanguageServerCodeEditor(self.create_language_server_symbol_retriever())

    def get_tool(self, tool_class: type[TTool]) -> TTool:
        if tool_class not in self._all_tools:
            raise KeyError(f"{tool_class} not found in tool pool")
        return self._all_tools[tool_class]  # type: ignore


class EditedFileContext(Component):
    """
    Context manager for file editing.

    Create the context, then use `set_updated_content` to set the new content, the original content
    being provided in `original_content`.
    When exiting the context without an exception, the updated content will be written back to the file.
    """

    def __init__(self, relative_path: str):
        self._abs_path = os.path.join(self.get_project_root(), relative_path)
        if not os.path.isfile(self._abs_path):
            raise FileNotFoundError(f"File {self._abs_path} does not exist.")
        with open(self._abs_path, encoding=SERENA_FILE_ENCODING) as f:
            self._original_content = f.read()
        self._updated_content: str | None = None

    def __enter__(self) -> Self:
        return self

    def get_original_content(self) -> str:
        return self._original_content

    def set_updated_content(self, content: str) -> None:
        self._updated_content = content

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._updated_content is not None and exc_type is None:
            with open(self._abs_path, "w", encoding=SERENA_FILE_ENCODING) as f:
                f.write(self._updated_content)
            log.info(f"Updated content written to {self._abs_path}")


class Tool(Component):
    """Base class for all tools with apply method and metadata support."""

    def __init__(
        self,
        languages: list[Language] | None = None,
        ls_manager: LanguageServerManager | None = None,
        component: "Component | None" = None,
    ):
        # Do not create _all_tools dictionary automatically to avoid RecursionError.
        super().__init__(languages=languages)
        if ls_manager is not None:
            self.language_server_manager = ls_manager

        self._owner: Component | None = component

    def get_tool(self, tool_class: type[TTool]) -> TTool:
        if self._owner is not None:  # we belong to a project
            return self._owner.get_tool(tool_class)
        raise RuntimeError(
            f"{self.__class__.__name__} was not created by a Component "
            "and cannot retrieve other tools."
        )

    @classmethod
    def get_name_from_cls(cls) -> str:
        name = cls.__name__
        if name.endswith("Tool"):
            name = name[:-4]
        name = "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip(
            "_"
        )
        return name

    def get_name(self) -> str:
        return self.get_name_from_cls()

    @classmethod
    def get_tool_description(cls) -> str:
        docstring = cls.__doc__
        return docstring.strip() if docstring else ""

    def get_apply_fn(self) -> ApplyMethodProtocol:
        apply_fn = getattr(self, "apply", None)
        if apply_fn is None:
            raise RuntimeError(
                f"apply method not defined in {self}. Did you forget to implement it?"
            )
        return apply_fn

    @classmethod
    def _get_apply_method(cls) -> Callable:
        if "apply" in cls.__dict__:
            return cls.__dict__["apply"]
        apply_fn = getattr(cls, "apply", None)
        if apply_fn is None:
            raise AttributeError(
                f"apply method not defined in {cls}. Did you forget to implement it?"
            )
        return apply_fn

    @classmethod
    def get_apply_docstring_from_cls(cls) -> str:
        apply_fn = cls._get_apply_method()
        docstring = apply_fn.__doc__
        if not docstring:
            raise AttributeError(
                f"apply method has no docstring in {cls}. Please add documentation."
            )
        return docstring.strip()

    def get_apply_docstring(self) -> str:
        return self.get_apply_docstring_from_cls()

    @classmethod
    def get_apply_fn_metadata_from_cls(cls) -> FuncMetadata:
        apply_fn = cls._get_apply_method()
        return func_metadata(apply_fn, skip_names=["self", "cls"])

    def get_apply_fn_metadata(self) -> FuncMetadata:
        return self.get_apply_fn_metadata_from_cls()

    def _log_tool_application(self, frame: Any) -> None:
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
        # Overridden to allow tool special handling if needed.
        super()._save_language_server_caches()

    def apply_ex(
        self, log_call: bool = True, catch_exceptions: bool = True, **kwargs
    ) -> str:
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
    tool_class: type[Tool]
    tool_name: str


@singleton
class ToolRegistry:
    """Singleton registry for managing all available tools."""

    def __init__(self) -> None:
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
        return self._tool_dict[tool_name].tool_class

    def get_all_tool_classes(self) -> list[type[Tool]]:
        return [t.tool_class for t in self._tool_dict.values()]

    def get_tool_names(self) -> list[str]:
        return list(self._tool_dict.keys())

    def is_valid_tool_name(self, tool_name: str) -> bool:
        return tool_name in self._tool_dict


class AvailableTools:
    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self.tool_names = [tool.get_name_from_cls() for tool in tools]
        self.tool_marker_names: set[str] = set()

    def __len__(self) -> int:
        return len(self.tools)


class TSProject(Component):
    def __init__(self):
        super().__init__(languages=[Language.TYPESCRIPT])
        manager = self.create_language_server_manager()

        # create every tool, passing *this* TSProject as the owner
        self._all_tools = {
            tool_class: tool_class(
                languages=None,
                ls_manager=manager,
                component=self,  # <-- here
            )
            for tool_class in ToolRegistry().get_all_tool_classes()
            if tool_class is not TSProject
        }
        self._exposed_tools = AvailableTools(list(self._all_tools.values()))
