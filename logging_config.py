import logging
import os
from typing import Optional


# ------------------------- custom log levels ------------------------- #

# Define custom levels for different optimization events
INSTRUCTION_LEVEL = logging.INFO + 1
logging.addLevelName(INSTRUCTION_LEVEL, "INSTR")

CANDIDATE_LEVEL = logging.INFO + 2
logging.addLevelName(CANDIDATE_LEVEL, "CANDIDATE")

EVALUATION_LEVEL = logging.INFO + 3
logging.addLevelName(EVALUATION_LEVEL, "EVAL")

BEST_SCORE_LEVEL = logging.INFO + 4
logging.addLevelName(BEST_SCORE_LEVEL, "BEST")


def instr(self: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Log 'instruction/demo' messages at the custom INSTR level."""
    if self.isEnabledFor(INSTRUCTION_LEVEL):
        self._log(INSTRUCTION_LEVEL, msg, args, **kwargs)


def candidate(self: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Log candidate sampling messages."""
    if self.isEnabledFor(CANDIDATE_LEVEL):
        self._log(CANDIDATE_LEVEL, msg, args, **kwargs)


def evaluation(self: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Log evaluation result messages."""
    if self.isEnabledFor(EVALUATION_LEVEL):
        self._log(EVALUATION_LEVEL, msg, args, **kwargs)


def best_score(self: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Log best score updates."""
    if self.isEnabledFor(BEST_SCORE_LEVEL):
        self._log(BEST_SCORE_LEVEL, msg, args, **kwargs)


logging.Logger.instr = instr  # type: ignore[attr-defined]
logging.Logger.candidate = candidate  # type: ignore[attr-defined]
logging.Logger.evaluation = evaluation  # type: ignore[attr-defined]
logging.Logger.best_score = best_score  # type: ignore[attr-defined]


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors based on the log level for console output."""

    # Basic ANSI color codes with bold for emphasis
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m\033[97m",  # White on red background
        "INSTR": "\033[35m",  # Magenta for instructions/demos
        "CANDIDATE": "\033[94m",  # Bright blue for candidate sampling
        "EVAL": "\033[96m",  # Bright cyan for evaluation results
        "BEST": "\033[1m\033[92m",  # Bold bright green for best scores
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # First format the message normally
        message = super().format(record)

        # If the message spans multiple lines, indent continuation lines
        if "\n" in message:
            lines = message.splitlines()
            if len(lines) > 1:
                # Keep first line as-is, indent subsequent lines for readability
                lines = [lines[0]] + ["    " + line for line in lines[1:]]
                message = "\n".join(lines)

        # Then wrap the entire message in a color based on the level
        color = self.COLORS.get(record.levelname, "")
        if not color:
            return message

        return f"{color}{message}{self.RESET}"


def setup_logging(
    *,
    level: int = logging.INFO,
    log_path: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """
    Configure root logging with colored console output and optional file logging.

    This function is safe to call multiple times; it will reset existing handlers.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear any existing handlers configured by basicConfig or previous calls
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = ColorFormatter(fmt)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # Optional file handler without colors
    if log_path:
        # Ensure directory exists (only if there's a directory component)
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)
