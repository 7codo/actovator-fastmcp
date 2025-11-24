"""
The Model Context Protocol (MCP) Server for the TypeScript-AI-Humanizer project
"""

import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import docstring_parser
from mcp.server.fastmcp import server
from mcp.server.fastmcp.server import FastMCP, Settings
from mcp.server.fastmcp.tools.base import Tool as MCPTool
from pydantic_settings import SettingsConfigDict
from sensai.util import logging

from app.tools.tools_base import TSProject, Tool
from app.constants import SERENA_LOG_FORMAT

log = logging.getLogger(__name__)


def configure_logging(*args, **kwargs) -> None:  # type: ignore
    # We only do something here if logging has not yet been configured.
    # Normally, logging is configured in the MCP server startup script.
    if not logging.is_enabled():
        logging.basicConfig(
            level=logging.INFO, stream=sys.stderr, format=SERENA_LOG_FORMAT
        )


# patch the logging configuration function in fastmcp, because it's hard-coded and broken
server.configure_logging = configure_logging  # type: ignore


@dataclass
class ProjectRequestContext:
    project: TSProject


class ProjectFactory:
    def __init__(self, project_path: str | None = None):
        """
        :param project_path: Absolute path to the TypeScript project directory.
            If None, the default from constants.py will be used.
        """
        self.project_path = project_path

    @staticmethod
    def make_mcp_tool(tool: Tool) -> MCPTool:
        """
        Create an MCP tool from a Tool instance.
        """
        func_name = tool.get_name()
        func_doc = tool.get_apply_docstring() or ""
        func_arg_metadata = tool.get_apply_fn_metadata()
        is_async = False
        parameters = func_arg_metadata.arg_model.model_json_schema()

        docstring = docstring_parser.parse(func_doc)

        # Mount the tool description from the docstring
        func_doc = (docstring.description or "").strip().strip(".")
        if func_doc:
            func_doc += "."
        if docstring.returns and (
            docstring_returns_descr := docstring.returns.description
        ):
            prefix = " " if func_doc else ""
            func_doc = f"{func_doc}{prefix}Returns {docstring_returns_descr.strip().strip('.')}."

        # Parse parameter descriptions
        docstring_params = {param.arg_name: param for param in docstring.params}
        parameters_properties: dict[str, dict[str, Any]] = parameters["properties"]
        for parameter, properties in parameters_properties.items():
            if (param_doc := docstring_params.get(parameter)) and param_doc.description:
                param_desc = f"{param_doc.description.strip().strip('.') + '.'}"
                properties["description"] = param_desc[0].upper() + param_desc[1:]

        def execute_fn(**kwargs) -> str:  # type: ignore
            return tool.apply_ex(log_call=True, catch_exceptions=True, **kwargs)

        return MCPTool(
            fn=execute_fn,
            name=func_name,
            description=func_doc,
            parameters=parameters,
            fn_metadata=func_arg_metadata,
            is_async=is_async,
            context_kwarg=None,
            annotations=None,
            title=None,
        )

    def _iter_tools(self, project: TSProject) -> Iterator[Tool]:
        yield from project._exposed_tools.tools

    def create_mcp_server(
        self,
    ) -> FastMCP:
        """
        Create an MCP server with the TSProject instance.
        """
        # Instantiate the TSProject (loads language server and tools)
        project = TSProject()
        # Override model_config to disable the use of `.env` files for reading settings
        Settings.model_config = SettingsConfigDict(env_prefix="FASTMCP_")
        mcp = FastMCP(
            lifespan=lambda srv: self.server_lifespan(srv, project), port=8001
        )
        return mcp

    @asynccontextmanager
    async def server_lifespan(
        self, mcp_server: FastMCP, project: TSProject
    ) -> AsyncIterator[None]:
        """Manage server startup and shutdown lifecycle."""
        # Register tools
        if mcp_server is not None:
            mcp_server._tool_manager._tools = {}
            for tool in self._iter_tools(project):
                mcp_tool = self.make_mcp_tool(tool)
                mcp_server._tool_manager._tools[tool.get_name()] = mcp_tool
            log.info(
                f"Starting MCP server with {len(mcp_server._tool_manager._tools)} tools: "
                f"{list(mcp_server._tool_manager._tools.keys())}"
            )
        log.info("MCP server lifetime setup complete")
        yield


def start_mcp_server():
    """Entrypoint to start the MCP server."""

    factory = ProjectFactory()
    mcp = factory.create_mcp_server()
    mcp.run()  # transport="streamable-http"


if __name__ == "__main__":
    # import pprint

    start_mcp_server()
    # project = TSProject()
    # print("tools", len(project._exposed_tools.tools))
    # tool = project._exposed_tools.tools[3]
    # print("tool name:", tool.get_name_from_cls())
    # result = tool.apply_ex(name_path="Alpha", relative_path="test.ts")
    # pprint.pprint(result, indent=3)

    # # FindSymbolTool
    # result = project._exposed_tools.tools[2].apply_ex(
    #     name_path="default", relative_path="next.config.ts"
    # )
    # print("result of FindSymbolTool", result)

    # for tool in project._exposed_tools.tools:
    #     print("tool", tool.get_name_from_cls())

# tool execute_shell_command (DONE)
# tool get_symbols_overview (DONE)
# tool find_symbol (DONE) zero based index

# tool find_referencing_symbols: Note: content_around_reference start from empty line and miss the end line

# tool replace_symbol_body: BUG: for method/function or class it replace it fully but in variables it replace the name and the value only
# // BEFORE
# const myVariable = 10;

# // AFTER
# const const test = 10;;


# tool insert_after_symbol (DONE)
# tool insert_before_symbol (DONE)
# tool rename_symbol (FAILD)
# tool read_file (DONE)
# tool create_text_file (DONE)
# tool list_dir (DONE)
# tool find_file (DONE)
# tool replace_regex (DONE)
# tool delete_lines (DONE)
# tool replace_lines  (DONE)
# tool insert_at_line (DONE) it push the below line
# tool search_for_pattern (DONE)
