from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

TerminalService = Literal["mdds", "fpss"]

DEFAULT_TERMINAL_HOST = "127.0.0.1"
DEFAULT_TERMINAL_PORT = 25510
DEFAULT_TERMINAL_JAVA = "java"


@dataclass(frozen=True, slots=True)
class TerminalPing:
    active: bool
    status: str | None
    service: TerminalService
    url: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalStart:
    process: subprocess.Popen
    command: tuple[str, ...]


def ping_terminal(
    service: TerminalService = "mdds",
    *,
    host: str = DEFAULT_TERMINAL_HOST,
    port: int = DEFAULT_TERMINAL_PORT,
    timeout: float = 2.0,
) -> TerminalPing:
    url = f"http://{host}:{port}/v2/system/{service}/status"
    try:
        with urlopen(url, timeout=timeout) as response:
            status = response.read().decode("utf-8").strip().upper()
    except HTTPError as exc:
        return TerminalPing(False, None, service, url, f"HTTP {exc.code}: {exc.reason}")
    except URLError as exc:
        return TerminalPing(False, None, service, url, str(exc.reason))
    except TimeoutError as exc:
        return TerminalPing(False, None, service, url, str(exc))
    except OSError as exc:
        return TerminalPing(False, None, service, url, str(exc))

    return TerminalPing(status == "CONNECTED", status, service, url)


def is_terminal_active(
    service: TerminalService = "mdds",
    *,
    host: str = DEFAULT_TERMINAL_HOST,
    port: int = DEFAULT_TERMINAL_PORT,
    timeout: float = 2.0,
) -> bool:
    return ping_terminal(service, host=host, port=port, timeout=timeout).active


def start_terminal(
    jar_path: str | Path,
    *,
    java: str = DEFAULT_TERMINAL_JAVA,
    creds_file: str | Path | None = None,
    config_file: str | Path | None = None,
    log_directory: str | Path | None = None,
    cwd: str | Path | None = None,
    stdout=None,
    stderr=None,
) -> TerminalStart:
    jar = Path(jar_path).expanduser()
    if not jar.exists():
        raise FileNotFoundError(f"Theta Terminal jar not found: {jar}")

    command = [java, "-jar", str(jar)]
    if creds_file is not None:
        command.append(f"--creds-file={Path(creds_file).expanduser()}")
    if config_file is not None:
        command.append(f"--config={Path(config_file).expanduser()}")
    if log_directory is not None:
        command.append(f"--log-directory={Path(log_directory).expanduser()}")

    process = subprocess.Popen(
        command,
        cwd=str(Path(cwd).expanduser()) if cwd is not None else str(jar.parent),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    return TerminalStart(process=process, command=tuple(command))


def ensure_terminal_running(
    jar_path: str | Path,
    *,
    service: TerminalService = "mdds",
    host: str = DEFAULT_TERMINAL_HOST,
    port: int = DEFAULT_TERMINAL_PORT,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    request_timeout: float = 2.0,
    java: str = DEFAULT_TERMINAL_JAVA,
    creds_file: str | Path | None = None,
    config_file: str | Path | None = None,
    log_directory: str | Path | None = None,
    cwd: str | Path | None = None,
    stdout=None,
    stderr=None,
) -> TerminalStart | None:
    if is_terminal_active(service, host=host, port=port, timeout=request_timeout):
        return None

    started = start_terminal(
        jar_path,
        java=java,
        creds_file=creds_file,
        config_file=config_file,
        log_directory=log_directory,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if started.process.poll() is not None:
            raise RuntimeError(
                f"Theta Terminal exited before becoming active with code {started.process.returncode}"
            )
        if is_terminal_active(service, host=host, port=port, timeout=request_timeout):
            return started
        time.sleep(poll_interval)

    raise TimeoutError(f"Theta Terminal did not become active within {timeout} seconds")
