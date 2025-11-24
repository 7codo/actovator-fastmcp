"""
Configuration objects for language servers
"""

import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from app.solidlsp import SolidLanguageServer


class FilenameMatcher:
    def __init__(self, *patterns: str) -> None:
        """
        :param patterns: fnmatch-compatible patterns
        """
        self.patterns = patterns

    def is_relevant_filename(self, fn: str) -> bool:
        for pattern in self.patterns:
            if fnmatch.fnmatch(fn, pattern):
                return True
        return False


class Language(str, Enum):
    """
    Possible languages with Multilspy.
    """

    TYPESCRIPT = "typescript"

    @classmethod
    def iter_all(cls, include_experimental: bool = False) -> Iterable[Self]:
        for lang in cls:
            if include_experimental or not lang.is_experimental():
                yield lang

    def is_experimental(self) -> bool:
        """
        Check if the language server is experimental or deprecated.
        """
        return self in {
            self.TYPESCRIPT_VTS,
            self.PYTHON_JEDI,
            self.CSHARP_OMNISHARP,
            self.RUBY_SOLARGRAPH,
            self.MARKDOWN,
        }

    def __str__(self) -> str:
        return self.value

    def get_source_fn_matcher(self) -> FilenameMatcher:
        match self:
            case self.PYTHON | self.PYTHON_JEDI:
                return FilenameMatcher("*.py", "*.pyi")
            case self.JAVA:
                return FilenameMatcher("*.java")
            case self.TYPESCRIPT | self.TYPESCRIPT_VTS:
                # see https://github.com/oraios/serena/issues/204
                path_patterns = []
                for prefix in ["c", "m", ""]:
                    for postfix in ["x", ""]:
                        for base_pattern in ["ts", "js"]:
                            path_patterns.append(f"*.{prefix}{base_pattern}{postfix}")
                return FilenameMatcher(*path_patterns)
            case self.CSHARP | self.CSHARP_OMNISHARP:
                return FilenameMatcher("*.cs")
            case self.RUST:
                return FilenameMatcher("*.rs")
            case self.GO:
                return FilenameMatcher("*.go")
            case self.RUBY:
                return FilenameMatcher("*.rb", "*.erb")
            case self.RUBY_SOLARGRAPH:
                return FilenameMatcher("*.rb")
            case self.CPP:
                return FilenameMatcher(
                    "*.cpp", "*.h", "*.hpp", "*.c", "*.hxx", "*.cc", "*.cxx"
                )
            case self.KOTLIN:
                return FilenameMatcher("*.kt", "*.kts")
            case self.DART:
                return FilenameMatcher("*.dart")
            case self.PHP:
                return FilenameMatcher("*.php")
            case self.R:
                return FilenameMatcher("*.R", "*.r", "*.Rmd", "*.Rnw")
            case self.PERL:
                return FilenameMatcher("*.pl", "*.pm", "*.t")
            case self.CLOJURE:
                return FilenameMatcher(
                    "*.clj", "*.cljs", "*.cljc", "*.edn"
                )  # codespell:ignore edn
            case self.ELIXIR:
                return FilenameMatcher("*.ex", "*.exs")
            case self.ELM:
                return FilenameMatcher("*.elm")
            case self.TERRAFORM:
                return FilenameMatcher("*.tf", "*.tfvars", "*.tfstate")
            case self.SWIFT:
                return FilenameMatcher("*.swift")
            case self.BASH:
                return FilenameMatcher("*.sh", "*.bash")
            case self.ZIG:
                return FilenameMatcher("*.zig", "*.zon")
            case self.LUA:
                return FilenameMatcher("*.lua")
            case self.NIX:
                return FilenameMatcher("*.nix")
            case self.ERLANG:
                return FilenameMatcher(
                    "*.erl", "*.hrl", "*.escript", "*.config", "*.app", "*.app.src"
                )
            case self.AL:
                return FilenameMatcher("*.al", "*.dal")
            case self.REGO:
                return FilenameMatcher("*.rego")
            case self.MARKDOWN:
                return FilenameMatcher("*.md", "*.markdown")
            case self.SCALA:
                return FilenameMatcher("*.scala", "*.sbt")
            case self.JULIA:
                return FilenameMatcher("*.jl")
            case self.FORTRAN:
                return FilenameMatcher(
                    "*.f90",
                    "*.F90",
                    "*.f95",
                    "*.F95",
                    "*.f03",
                    "*.F03",
                    "*.f08",
                    "*.F08",
                    "*.f",
                    "*.F",
                    "*.for",
                    "*.FOR",
                    "*.fpp",
                    "*.FPP",
                )
            case self.HASKELL:
                return FilenameMatcher("*.hs", "*.lhs")
            case _:
                raise ValueError(f"Unhandled language: {self}")

    def get_ls_class(self) -> type["SolidLanguageServer"]:
        match self:
            case self.TYPESCRIPT:
                from app.solidlsp.language_servers.typescript_language_server import (
                    TypeScriptLanguageServer,
                )

                return TypeScriptLanguageServer
            case _:
                raise ValueError(f"Unhandled language: {self}")

    @classmethod
    def from_ls_class(cls, ls_class: type["SolidLanguageServer"]) -> Self:
        """
        Get the Language enum value from a SolidLanguageServer class.

        :param ls_class: The SolidLanguageServer class to find the corresponding Language for
        :return: The Language enum value
        :raises ValueError: If the language server class is not supported
        """
        for enum_instance in cls:
            if enum_instance.get_ls_class() == ls_class:
                return enum_instance
        raise ValueError(f"Unhandled language server class: {ls_class}")


@dataclass
class LanguageServerConfig:
    """
    Configuration parameters
    """

    code_language: Language
    trace_lsp_communication: bool = False
    start_independent_lsp_process: bool = True
    ignored_paths: list[str] = field(default_factory=list)
    """Paths, dirs or glob-like patterns. The matching will follow the same logic as for .gitignore entries"""
    encoding: str = "utf-8"
    """File encoding to use when reading source files"""

    @classmethod
    def from_dict(cls, env: dict):
        """
        Create a MultilspyConfig instance from a dictionary
        """
        import inspect

        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )
