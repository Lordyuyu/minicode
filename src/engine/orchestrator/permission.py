from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.types import PermissionScope
from src.utils.logger import get_logger

logger = get_logger(__name__)

_READ_TOOLS: set[str] = {
    "read_file", "search_code", "grep", "glob", "list_files",
    "read_directory", "get_file_info", "fetch_url",
}

_WRITE_TOOLS: set[str] = _READ_TOOLS | {
    "write_file", "edit_file", "create_file", "delete_file",
    "rename_file", "apply_diff",
}

_TEST_TOOLS: set[str] = _WRITE_TOOLS | {
    "run_command", "run_test", "execute_script", "run_pytest",
}

_ALL_TOOLS: set[str] = _TEST_TOOLS | {
    "shell_exec", "install_package", "modify_config",
    "manage_service", "network_request",
}

_PERMISSION_MAP: dict[PermissionScope, set[str]] = {
    PermissionScope.READ_ONLY: _READ_TOOLS,
    PermissionScope.CODE_WRITE: _WRITE_TOOLS,
    PermissionScope.TEST_EXEC: _TEST_TOOLS,
    PermissionScope.FULL_ACCESS: _ALL_TOOLS,
}


class PermissionController:
    def __init__(self) -> None:
        self._whitelist: list[str] = []
        self._blacklist: list[str] = []

    def check_access(self, scope: PermissionScope, tool_name: str) -> bool:
        allowed = _PERMISSION_MAP.get(scope)
        if allowed is None:
            logger.warning("Unknown permission scope: {}", scope)
            return False
        return tool_name in allowed

    def check_path(self, path: str, scope: PermissionScope) -> bool:
        resolved = str(Path(path).resolve())
        for blocked in self._blacklist:
            if resolved.startswith(blocked):
                logger.warning("Path {} is blacklisted", resolved)
                return False
        if self._whitelist:
            for allowed in self._whitelist:
                if resolved.startswith(allowed):
                    return True
            logger.warning("Path {} is not in whitelist", resolved)
            return False
        return True

    def restrict_paths(self, whitelist: list[str], blacklist: list[str]) -> None:
        self._whitelist = [str(Path(p).resolve()) for p in whitelist]
        self._blacklist = [str(Path(p).resolve()) for p in blacklist]
        logger.info(
            "Path restrictions updated: whitelist={}, blacklist={}",
            self._whitelist,
            self._blacklist,
        )

    def validate_tool_call(
        self, tool_name: str, args: dict[str, Any], scope: PermissionScope
    ) -> tuple[bool, str]:
        if not self.check_access(scope, tool_name):
            msg = f"Tool '{tool_name}' is not allowed under scope {scope}"
            logger.warning(msg)
            return False, msg
        path = args.get("path") or args.get("file_path")
        if path is not None and not self.check_path(str(path), scope):
            msg = f"Path '{path}' is restricted under scope {scope}"
            logger.warning(msg)
            return False, msg
        return True, ""
