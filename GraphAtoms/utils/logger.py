import sys
from pathlib import Path

from loguru._logger import Core
from loguru._logger import Logger as _Logger


class LoggerBase:
    def __init__(
        self,
        workdir: Path | str | None = ".",
        logfile: Path | str | None = None,
        loglevel: int | str = "DEBUG",
        logstdout: bool = True,
        format: str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:^8}</level> | "
            "<level>{message}</level>"
        ),
    ) -> None:
        self._logger = _Logger(
            core=Core(),
            exception=None,
            depth=0,
            record=False,
            lazy=False,
            colors=False,
            raw=False,
            capture=True,
            patchers=[],
            extra={},
        )
        if logstdout:
            self._logger.add(sys.stderr, format=format, level=loglevel)
        if logfile is not None:
            logfile = Path(logfile)
            if workdir is not None:
                logfile = Path(workdir) / logfile.name
            self._logger.add(logfile, format=format, level=loglevel)


class Logger(LoggerBase):
    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warn(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)
