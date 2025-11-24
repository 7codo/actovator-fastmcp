from logging import Logger
from pathlib import Path
from typing import Any, Literal
from app.run_mcp import ProjectFactory
import click
from sensai.util import logging


log = logging.getLogger(__name__)

# --------------------- Utilities -------------------------------------


class ProjectType(click.ParamType):
    """ParamType allowing either a project name or a path to a project directory."""

    name = "[PROJECT_NAME|PROJECT_PATH]"

    def convert(self, value: str, param: Any, ctx: Any) -> str:
        path = Path(value).resolve()
        if path.exists() and path.is_dir():
            return str(path)
        return value


PROJECT_TYPE = ProjectType()


class AutoRegisteringGroup(click.Group):
    """
    A click.Group subclass that automatically registers any click.Command
    attributes defined on the class into the group.

    After initialization, it inspects its own class for attributes that are
    instances of click.Command (typically created via @click.command) and
    calls self.add_command(cmd) on each. This lets you define your commands
    as static methods on the subclass for IDE-friendly organization without
    manual registration.
    """

    def __init__(self, name: str, help: str):
        super().__init__(name=name, help=help)
        # Scan class attributes for click.Command instances and register them.
        for attr in dir(self.__class__):
            cmd = getattr(self.__class__, attr)
            if isinstance(cmd, click.Command):
                self.add_command(cmd)


class TopLevelCommands(AutoRegisteringGroup):
    """Root CLI group containing the core Serena commands."""

    def __init__(self) -> None:
        super().__init__(
            name="serena",
            help="Serena CLI commands. You can run `<command> --help` for more info on each command.",
        )

    @staticmethod
    @click.command("start-mcp-server", help="Starts the Serena MCP server.")
    @click.option(
        "--transport",
        type=click.Choice(["stdio", "sse", "streamable-http"]),
        default="stdio",
        show_default=True,
        help="Transport protocol.",
    )
    @click.option("--host", type=str, default="0.0.0.0", show_default=True)
    @click.option("--port", type=int, default=8000, show_default=True)
    @click.option(
        "--log-level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        default=None,
        help="Override log level in config.",
    )
    @click.option(
        "--trace-lsp-communication",
        type=bool,
        is_flag=False,
        default=None,
        help="Whether to trace LSP communication.",
    )
    @click.option(
        "--tool-timeout",
        type=float,
        default=None,
        help="Override tool execution timeout in config.",
    )
    def start_mcp_server() -> None:
        # initialize logging, using INFO level initially (will later be adjusted by SerenaAgent according to the config)
        #   * memory log handler (for use by GUI/Dashboard)
        #   * stream handler for stderr (for direct console output, which will also be captured by clients like Claude Desktop)
        #   * file handler
        # (Note that stdout must never be used for logging, as it is used by the MCP server to communicate with the client.)
        Logger.root.setLevel(logging.INFO)
        factory = ProjectFactory()
        mcp = factory.create_mcp_server()
        mcp.run()


# Expose toplevel commands for the same reason
top_level = TopLevelCommands()
start_mcp_server = top_level.start_mcp_server

if __name__ == "__main__":
    # import pprint

    start_mcp_server()
    # project = TSProject()
    # print("tools", len(project._exposed_tools.tools))
