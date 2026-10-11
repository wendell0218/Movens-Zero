"""Non-actuating, per-session task selection and line editing."""
from dataclasses import dataclass
from pathlib import Path
import codecs
import os
import select
from collections import deque

MODES = {'1': 'text', '2': 'video', '3': 'speech', '4': 'music'}
LABELS = {'text': '文本', 'video': '视频', 'speech': '语音', 'music': '音乐'}
SUFFIXES = {'video': {'.mp4','.mov','.mkv','.avi','.webm','.m4v'},
            'speech': {'.wav','.mp3','.flac','.ogg','.m4a'},
            'music': {'.wav','.mp3','.flac','.ogg','.m4a'}}

@dataclass(frozen=True)
class TaskInput:
    mode: str
    content: str
    seconds: float = 6.0


def validate_task(mode, content, seconds=6.0):
    if mode not in LABELS:
        raise ValueError('请选择 1–4 模态')
    content = content.strip()
    if not content:
        raise ValueError('输入不能为空')
    if mode != 'text':
        # Support a quoted path pasted from a shell, including spaces.
        if len(content) >= 2 and content[0] == content[-1] and content[0] in "\"'":
            content = content[1:-1]
        path = Path(content).expanduser()
        if not path.is_file():
            raise ValueError(f'文件不存在或不是普通文件：{path}')
        if path.suffix.lower() not in SUFFIXES[mode]:
            raise ValueError(f'文件类型不匹配：{path.suffix}')
        if not os.access(path, os.R_OK):
            raise ValueError(f'文件不可读：{path}')
        content = str(path.resolve())
    return TaskInput(mode, content, seconds)


class KeyReader:
    """Read pasted UTF-8 without TextIO buffering hiding already-read keys."""
    def __init__(self, fd):
        self.fd = fd
        self.pending = deque()
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')

    def read(self, timeout=0.2):
        if not self.pending:
            ready, _, _ = select.select([self.fd], [], [], timeout)
            if not ready:
                return None
            data = os.read(self.fd, 4096)
            if not data:
                return '\x03'
            self.pending.extend(self.decoder.decode(data))
        if not self.pending:
            return None
        key = self.pending.popleft()
        if key != '\x1b':
            return key
        # Consume ANSI sequences atomically: their trailing letters/brackets
        # must never leak into robot controls when cancelling an input.
        if not self.pending:
            ready, _, _ = select.select([self.fd], [], [], 0.02)
            if ready:
                self.pending.extend(self.decoder.decode(os.read(self.fd, 4096)))
        if self.pending and self.pending[0] in ('[', 'O'):
            key += self.pending.popleft()
            for _ in range(32):
                if not self.pending:
                    ready, _, _ = select.select([self.fd], [], [], 0.02)
                    if not ready:
                        break
                    self.pending.extend(self.decoder.decode(os.read(self.fd, 4096)))
                if not self.pending:
                    break
                char = self.pending.popleft(); key += char
                if '@' <= char <= '~':
                    break
        return key


class TaskMenu:
    def __init__(self, emit=print, seconds=6.0):
        self.emit = emit
        self.seconds = seconds
        self.history = []
        self.editing = None
        self.buffer = ''
        self.escape = ''

    def show(self):
        self.emit('\n[任务] 1 文本 | 2 video | 3 speech | 4 music | h 历史\n'
                  '选好输入后按 ] 执行；Enter / ! 返回 planner；o 停止。')

    def remember(self, task):
        if task not in self.history:
            self.history.append(task)

    def begin(self, key):
        if key == 'h':
            if not self.history:
                self.emit('[历史] 本次启动还没有输入记录。'); return
            self.emit('\n[历史] 输入编号后回车复用：')
            for i, task in enumerate(self.history, 1):
                self.emit(f'  {i}. {LABELS[task.mode]} · {task.content}')
            self.editing = 'history'
        else:
            self.editing = MODES[key]
            label = '动作描述' if self.editing == 'text' else '本机文件路径'
            self.emit(f'\n[{LABELS[self.editing]}] 输入{label}，回车确认。')
        self.buffer = ''; self.escape = ''
        self.emit('输入期间：Ctrl+C 停止；Esc 取消；退格修改（o 此时是普通字符）。')
        self.render()

    def render(self):
        self.emit('\r\033[2K> ' + self.buffer, end='', flush=True)

    def cancel(self):
        self.editing = None; self.buffer = ''; self.escape = ''
        self.emit('\n[输入] 已取消，原任务保持不变。')

    def feed(self, key):
        if key.startswith('\x1b') and len(key) > 1:
            return None
        if key == '\x1b':
            self.cancel(); return None
        if key in ('\r', '\n'):
            self.emit('')
            try:
                if self.editing == 'history':
                    index = int(self.buffer) - 1
                    if index < 0 or index >= len(self.history): raise ValueError('历史编号不存在')
                    old = self.history[index]
                    task = validate_task(old.mode, old.content, old.seconds)
                else:
                    task = validate_task(self.editing, self.buffer, self.seconds)
            except (ValueError, OSError) as exc:
                self.emit(f'\a[输入错误] {exc}；请重新输入。')
                self.buffer = ''; self.render(); return None
            self.editing = None; self.buffer = ''
            return task
        if key in ('\x7f', '\b'):
            self.buffer = self.buffer[:-1]
        elif key == '\x15':
            self.buffer = ''
        elif key.isprintable() and len(self.buffer) < 4096:
            self.buffer += key
        self.render()
        return None
