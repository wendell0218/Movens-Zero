"""UI tests use mocks/pipes only: no DDS, websocket or controller initialization."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from terminal_ui import TaskMenu, TaskInput, KeyReader, validate_task

class MenuTests(unittest.TestCase):
    def setUp(self):
        self.output=[]
        self.menu=TaskMenu(emit=lambda *a,**k:self.output.append(' '.join(map(str,a))))
    def enter(self, text):
        result=None
        for c in text+'\n': result=self.menu.feed(c)
        return result
    def test_text_history_and_duplicate(self):
        self.menu.begin('1'); task=self.enter('向左挥手 o ] !')
        self.assertEqual(task.content,'向左挥手 o ] !')
        self.menu.remember(task);self.menu.remember(task)
        self.assertEqual(len(self.menu.history),1)
        self.menu.begin('h');self.assertEqual(self.enter('1'),task)
    def test_all_media_and_missing_history(self):
        with tempfile.TemporaryDirectory() as d:
            for key,mode,suffix in [('2','video','.mp4'),('3','speech','.wav'),('4','music','.mp3')]:
                p=Path(d)/('含 空格'+suffix);p.write_bytes(b'fixture')
                self.menu.begin(key);task=self.enter('"'+str(p)+'"')
                self.assertEqual(task.mode,mode);self.menu.remember(task)
            p.unlink();self.menu.begin('h');self.assertIsNone(self.enter('3'))
            self.assertTrue(any('文件不存在' in s for s in self.output))
    def test_invalid_no_history_change(self):
        self.menu.begin('2');self.assertIsNone(self.enter('/does/not/exist.mp4'))
        self.assertEqual(self.menu.history,[]);self.assertEqual(self.menu.editing,'video')
    def test_cancel_and_edit(self):
        self.menu.begin('1')
        for c in 'abc\x7fd':self.menu.feed(c)
        self.assertEqual(self.menu.buffer,'abd')
        self.menu.feed('\x1b');self.assertIsNone(self.menu.editing)
    def test_paste_utf8_and_atomic_escape(self):
        r,w=os.pipe()
        try:
            os.write(w,'你好\x1b[A]'.encode());reader=KeyReader(r)
            self.assertEqual([reader.read() for _ in range(4)],['你','好','\x1b[A',']'])
            os.close(w);w=None;self.assertEqual(reader.read(),'\x03')
        finally:
            os.close(r)
            if w is not None:os.close(w)
    def test_cli_noarg_defers_selection(self):
        import client
        with patch('sys.argv',['client']), patch.object(client.sys.stdin,'isatty',return_value=True), \
             patch('builtins.input',side_effect=AssertionError('No initial input prompt')), \
             patch('run_psi0_sonic_online_jog.wait_psi_server'), \
             patch.object(client.os,'execv') as execute:
            client.main()
        argv=execute.call_args.args[1]
        self.assertIn('--interactive-task-menu',argv)
        self.assertIn('--max-seconds',argv)
    def test_history_revalidates_and_keeps_duration(self):
        task=TaskInput('text','wave',12.0);self.menu.remember(task)
        self.menu.begin('h');self.assertIsNone(self.enter('0'))
        self.assertEqual(self.enter('1'),task)
    def test_arrow_sequence_does_not_submit_or_cancel(self):
        self.menu.begin('1');self.menu.feed('a');self.menu.feed('\x1b[A')
        self.assertEqual(self.menu.buffer,'a');self.assertEqual(self.menu.editing,'text')

    def test_orchestrator_no_yes_and_task_gating(self):
        import run_psi0_sonic_online_jog as jog
        with patch('sys.argv',['jog','--visual-input-mode','text','--prompt','bootstrap',
                               '--max-seconds','6','--interactive-task-menu']):
            args=jog.parse_args()
        # First ] without input must not run a motion; second ] after selection may.
        keys=iter(['s','\n',']','1',*'A person stands still.','\n',']','o'])
        reader=MagicMock();reader.read.side_effect=lambda **kw:next(keys)
        proc=MagicMock();proc.poll.return_value=None
        flags=[];commands=[]
        with patch.object(jog,'parse_args',return_value=args), \
             patch.object(jog.sys.stdin,'isatty',return_value=True), \
             patch.object(jog,'raw_terminal',return_value=nullcontext()), \
             patch('terminal_ui.KeyReader',return_value=reader), \
             patch('builtins.input',side_effect=AssertionError('No startup confirmation allowed')), \
             patch.object(jog,'wait_psi_server'),patch.object(jog,'wait_live_state'), \
             patch.object(jog,'wait_log_pattern'),patch.object(jog,'wait_file'), \
             patch.object(jog,'start_process',return_value=proc),patch.object(jog,'stop_process'), \
             patch.object(jog,'clean_paths'),patch.object(jog.time,'sleep'), \
             patch.object(jog,'send_wbc_key',side_effect=lambda p,k:commands.append(k)), \
             patch.object(jog,'write_flag',side_effect=lambda p,v:flags.append(str(p))):
            self.assertEqual(jog.main(),0)
        self.assertEqual(flags.count(args.start_flag_path),1)
        self.assertIn('\n]',commands)

if __name__=='__main__':unittest.main(verbosity=2)
