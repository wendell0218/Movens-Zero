from dataclasses import dataclass
from pathlib import Path
import codecs
import os
import select
from collections import deque

MODES = {"1": "text", "2": "video", "3": "speech", "4": "music"}
LABELS = {"text": "Text", "video": "Video", "speech": "Speech", "music": "Music"}
SUFFIXES = {
    "video": {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"},
    "speech": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
    "music": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
}


@dataclass(frozen=True)
class TaskInput:
    mode: str
    content: str
    seconds: float = 6.0


def validate_task(mode, content, seconds=6.0):
    if mode not in LABELS:
        raise ValueError("Choose a modality (1-4)")
    content = content.strip()
    if not content:
        raise ValueError("Input must not be empty")
    if mode != "text":
        if len(content) >= 2 and content[0] == content[-1] and (content[0] in "\"'"):
            content = content[1:-1]
        path = Path(content).expanduser()
        if not path.is_file():
            raise ValueError(f"File does not exist or is not a regular file: {path}")
        if path.suffix.lower() not in SUFFIXES[mode]:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        if not os.access(path, os.R_OK):
            raise ValueError(f"File is not readable: {path}")
        content = str(path.resolve())
    return TaskInput(mode, content, seconds)


class KeyReader:
    def __init__(self, fd):
        self.fd = fd
        self.pending = deque()
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def read(self, timeout=0.2):
        if not self.pending:
            ready, _, _ = select.select([self.fd], [], [], timeout)
            if not ready:
                return None
            data = os.read(self.fd, 4096)
            if not data:
                return "\x03"
            self.pending.extend(self.decoder.decode(data))
        if not self.pending:
            return None
        key = self.pending.popleft()
        if key != "\x1b":
            return key
        if not self.pending:
            ready, _, _ = select.select([self.fd], [], [], 0.02)
            if ready:
                self.pending.extend(self.decoder.decode(os.read(self.fd, 4096)))
        if self.pending and self.pending[0] in ("[", "O"):
            key += self.pending.popleft()
            for _ in range(32):
                if not self.pending:
                    ready, _, _ = select.select([self.fd], [], [], 0.02)
                    if not ready:
                        break
                    self.pending.extend(self.decoder.decode(os.read(self.fd, 4096)))
                if not self.pending:
                    break
                char = self.pending.popleft()
                key += char
                if "@" <= char <= "~":
                    break
        return key


class TaskMenu:
    def __init__(self, emit=print, seconds=6.0):
        self.emit = emit
        self.seconds = seconds
        self.history = []
        self.editing = None
        self.buffer = ""
        self.escape = ""

    def show(self):
        self.emit(
            "\n[Task] 1 Text | 2 Video | 3 Speech | 4 Music | h History\nPress ] to run the selected task; Enter / ! returns to planner; o stops."
        )

    def remember(self, task):
        if task not in self.history:
            self.history.append(task)

    def begin(self, key):
        if key == "h":
            if not self.history:
                self.emit("[History] No inputs in this session.")
                return
            self.emit("\n[History] Enter an item number to reuse:")
            for i, task in enumerate(self.history, 1):
                self.emit(f"  {i}. {LABELS[task.mode]} · {task.content}")
            self.editing = "history"
        else:
            self.editing = MODES[key]
            label = (
                "motion description" if self.editing == "text" else "local file path"
            )
            self.emit(f"\n[{LABELS[self.editing]}] Enter a {label}, then press Enter.")
        self.buffer = ""
        self.escape = ""
        self.emit(
            "While editing: Ctrl+C stops; Esc cancels; Backspace edits (o is ordinary text)."
        )
        self.render()

    def render(self):
        self.emit("\r\x1b[2K> " + self.buffer, end="", flush=True)

    def cancel(self):
        self.editing = None
        self.buffer = ""
        self.escape = ""
        self.emit("\n[Input] Cancelled; the previous task is unchanged.")

    def feed(self, key):
        if key.startswith("\x1b") and len(key) > 1:
            return None
        if key == "\x1b":
            self.cancel()
            return None
        if key in ("\r", "\n"):
            self.emit("")
            try:
                if self.editing == "history":
                    index = int(self.buffer) - 1
                    if index < 0 or index >= len(self.history):
                        raise ValueError("History item does not exist")
                    old = self.history[index]
                    task = validate_task(old.mode, old.content, old.seconds)
                else:
                    task = validate_task(self.editing, self.buffer, self.seconds)
            except (ValueError, OSError) as exc:
                self.emit(f"\x07[Input error] {exc}; try again.")
                self.buffer = ""
                self.render()
                return None
            self.editing = None
            self.buffer = ""
            return task
        if key in ("\x7f", "\x08"):
            self.buffer = self.buffer[:-1]
        elif key == "\x15":
            self.buffer = ""
        elif key.isprintable() and len(self.buffer) < 4096:
            self.buffer += key
        self.render()
        return None
