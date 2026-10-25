"""Host integration without network: actual provider discovery, dispatch, TUI gates.

Only the agy subprocess producer and UI write_json egress are substituted.
The live script independently exercises the real agy producer.
"""
import importlib
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path.home()/'.hermes'/'hermes-agent'))
import hermes_bootstrap
from hermes_cli.plugins_discovery import suppress_plugin_discovery

# Avoid unrelated default-profile plugins during isolated regression execution.
with suppress_plugin_discovery():
    from run_agent import AIAgent
    from agent.subagent_lifecycle import bind_subagent_parent
    from agent.turn_api_call import _should_stream, perform_api_call
    from agent.notification_presentation import notification_turn
    from hermes_cli.plugins import get_plugin_manager
    from providers import get_provider_profile
    from tui_gateway import server


class HostPathTests(unittest.TestCase):
    def setUp(self):
        self.scope = suppress_plugin_discovery(); self.scope.__enter__()
        self.addCleanup(self.scope.__exit__,None,None,None)
        self.profile = get_provider_profile('antigravity-oauth')
        self.module = importlib.import_module(type(self.profile).__module__)
        self.bridge = importlib.import_module(type(self.profile).__module__+'.live_reasoning')
        self.client_module = importlib.import_module(type(self.profile).__module__+'.client')
        self.frames = []
        self.sid = 'adapter-regression'
        self.agent = AIAgent(provider='antigravity-oauth',model='gemini-4-argon',base_url='agy://local',
            api_mode='chat_completions',session_id='adapter-regression-store',platform='tui',
            enabled_toolsets=[],disabled_toolsets=['all'],quiet_mode=True,skip_context_files=True,
            skip_memory=True,skip_background_review=True,load_soul_identity=False,
            reasoning_callback=server._agent_cbs(self.sid)['reasoning_callback'],
            stream_delta_callback=lambda text: None)
        self.addCleanup(self.agent.client.close)
        server._sessions[self.sid] = {'agent':self.agent,'running':True,'show_reasoning':True,'profile_home':None}
        self.addCleanup(server._sessions.pop,self.sid,None)
        self.agent._current_turn_id = 'turn'
        self.agent._current_api_request_id = 'req'
        self.request = self.agent._build_api_kwargs([{'role':'user','content':'fixture'}],tools_for_api=[])
        self.assertEqual(self.request['_agy_session_id'],self.agent.session_id)
        self.assertTrue(_should_stream(self.agent))
        self.assertEqual(self.profile.auth_type,'external_process')
        self.spy = patch.object(server,'write_json',side_effect=lambda f:self.frames.append(f) or True)
        self.spy.start();self.addCleanup(self.spy.stop)

    def proc(self):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdin = io.StringIO()
        proc.stdout = io.StringIO('\n'.join(json.dumps(x) for x in [
            {'event':'init','init':{'conversation_id':'fixture'}},
            {'event':'step_update','step_update':{'thinking_delta':'private thinking'}},
            {'event':'text_delta','text_delta':'answer'},
            {'event':'result','result':{'status':'SUCCESS','response':'answer',
                'usage':{'input_tokens':3,'output_tokens':4,'total_tokens':7,'thinking_tokens':2}}},
        ])+'\n')
        proc.stderr = io.StringIO()
        return proc

    def perform(self):
        with patch.object(self.client_module,'is_authenticated',return_value=True), \
             patch.object(self.client_module.AntigravityClient,'_get_or_spawn_worker',return_value=self.proc()), \
             bind_subagent_parent(self.agent):
            return perform_api_call(self.agent,api_kwargs=self.request,_original_api_kwargs=dict(self.request),
                _llm_middleware_trace=[],_moa_prepared_request=None,_retry=NS(),thinking_spinner=None,
                retry_count=0,api_call_count=1,api_request_id='req',effective_task_id='task',turn_id='turn',interrupted=False).response

    def reasoning_frames(self):
        return [f for f in self.frames if f.get('params',{}).get('type') == 'reasoning.delta']

    def test_host_stream_consumer_delivers_before_owned_child_completion(self):
        import subprocess
        import time
        import os
        self.assertEqual(Path(self.module.__file__).parent,
            Path(os.environ['HERMES_HOME']) / 'plugins' / 'antigravity-oauth')
        executable = getattr(sys, '_base_executable', sys.executable)
        # The child cannot complete until the real host text callback receives
        # its delta. A collector-only path deadlocks/fails the bounded wait.
        program = '''import sys,json
sys.stdin.readline()
for event in [
 {'event':'init','conversation_id':'paced'},
 {'event':'step_update','step_update':{'thinking_delta':'private thinking'}},
 {'event':'step_update','step_update':{'text_delta':'answer\\n'}}]:
 print(json.dumps(event),flush=True)
if sys.stdin.readline().strip() != 'release': sys.exit(4)
print(json.dumps({'event':'result','result':{'status':'SUCCESS','response':'answer\\n',
 'usage':{'input_tokens':3,'output_tokens':4,'total_tokens':7}}}),flush=True)
sys.stdin.read()
'''
        real_popen = subprocess.Popen
        processes, delivered, received = [], [], threading.Event()
        def spawn(*args, **kwargs):
            proc = real_popen([executable, '-u', '-c', program], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                **self.client_module._own_process_group())
            processes.append(proc)
            return proc
        def on_text(text):
            delivered.append(text)
            if not received.is_set():
                received.set()
                processes[0].stdin.write('release\n')
                processes[0].stdin.flush()
        self.agent.stream_delta_callback = on_text
        with patch.object(self.client_module, 'is_authenticated', return_value=True), \
             patch.object(self.client_module, 'subprocess', NS(Popen=spawn, PIPE=subprocess.PIPE)), \
             bind_subagent_parent(self.agent):
            try:
                response = perform_api_call(self.agent, api_kwargs={**self.request,'timeout':3},
                    _original_api_kwargs=dict(self.request), _llm_middleware_trace=[],
                    _moa_prepared_request=None, _retry=NS(), thinking_spinner=None, retry_count=0,
                    api_call_count=1, api_request_id='req', effective_task_id='task',
                    turn_id='turn', interrupted=False).response
                self.assertTrue(received.is_set())
                self.assertEqual(''.join(delivered), 'answer\n')
                self.assertEqual(response.choices[0].message.content, 'answer\n')
                self.assertEqual(response.choices[0].message.reasoning_content, 'private thinking')
                self.assertEqual(len(self.reasoning_frames()), 1)
                self.assertEqual(response.usage.total_tokens, 7)
                self.assertEqual(len(processes), 1)
            finally:
                self.agent.client.cancel()
                self.agent.client.close()
                for proc in processes:
                    if proc.poll() is None: proc.kill()
                    proc.wait(timeout=3)
                    for pipe in (proc.stdin, proc.stdout, proc.stderr):
                        if pipe is not None: pipe.close()

    def test_host_abort_calls_plugin_cancel_and_owner_unwinds(self):
        import subprocess
        client = self.agent._create_request_openai_client(reason='cancel_fixture', api_kwargs=self.request)
        real_popen = subprocess.Popen
        executable = getattr(sys, '_base_executable', sys.executable)
        started, processes, errors = threading.Event(), [], []
        def spawn(*args, **kwargs):
            proc = real_popen([executable, '-u', '-c', 'import time; time.sleep(30)'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, **self.client_module._own_process_group())
            processes.append(proc)
            started.set()
            return proc
        def consume():
            try:
                client.chat.completions.create(**{**self.request,'timeout':10})
            except Exception as exc:
                errors.append(exc)
            finally:
                self.agent._close_request_openai_client(client, reason='cancel_fixture_owner')
        with patch.object(self.client_module, 'is_authenticated', return_value=True), \
             patch.object(self.client_module, 'subprocess', NS(Popen=spawn, PIPE=subprocess.PIPE)), \
             patch.object(type(client), 'cancel', autospec=True, side_effect=type(client).cancel) as cancel:
            owner = threading.Thread(target=consume, daemon=True)
            owner.start()
            try:
                self.assertTrue(started.wait(3))
                self.agent._abort_request_openai_client(client, reason='fixture_stale_abort')
                owner.join(3)
                self.assertFalse(owner.is_alive())
                cancel.assert_called_once_with(client)
                self.assertTrue(errors and isinstance(errors[0], InterruptedError), errors)
                self.assertEqual(len(processes), 1)
                self.assertTrue(all(proc.poll() is not None for proc in processes))
                self.assertEqual(client._worker_history, [])
                self.assertFalse(client._worker_lock.locked())
                replacement = self.agent._create_request_openai_client(reason='after_cancel', api_kwargs=self.request)
                self.assertIsNot(replacement, client)
                self.agent._close_request_openai_client(replacement, reason='fixture_done')
            finally:
                for proc in processes:
                    if proc.poll() is None: proc.kill()
                    proc.wait(timeout=3)
                owner.join(3)
                for proc in processes:
                    for pipe in (proc.stdin, proc.stdout, proc.stderr):
                        if pipe is not None: pipe.close()

    def test_host_native_tool_finish_and_interrupt_have_no_automatic_replay(self):
        tools = [{'type':'function','function':{'name':'terminal','parameters':{'type':'object'}}}]
        self.request = self.agent._build_api_kwargs([{'role':'user','content':'fixture'}], tools_for_api=tools)
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                proc = MagicMock()
                dead = threading.Event()
                proc.poll.side_effect = lambda: 1 if dead.is_set() else None
                proc.terminate.side_effect = dead.set
                proc.stdout = io.StringIO(json.dumps({'event':'step_update','step_update':{
                    'step_index':2,'step_type':'tool','tool_name':'run_command',
                    'tool_info':{'name':'run_command','parameters':{'CommandLine':'fixture-only'}}}})+'\n')
                proc.stderr = io.StringIO()
                observed = []
                def on_tool(name):
                    observed.append((name, dead.is_set()))
                    if interrupted: self.agent._interrupt_requested = True
                with patch.object(self.client_module, 'is_authenticated', return_value=True), \
                     patch.object(self.client_module, 'subprocess', NS(Popen=MagicMock(return_value=proc),
                         PIPE=self.client_module.subprocess.PIPE)) as producer, \
                     patch.object(self.agent, '_fire_tool_gen_started', side_effect=on_tool), \
                     bind_subagent_parent(self.agent):
                    def call():
                        return perform_api_call(self.agent, api_kwargs=self.request,
                            _original_api_kwargs=dict(self.request), _llm_middleware_trace=[],
                            _moa_prepared_request=None, _retry=NS(), thinking_spinner=None, retry_count=0,
                            api_call_count=1, api_request_id='req', effective_task_id='task',
                            turn_id='turn', interrupted=False).response
                    try:
                        if interrupted:
                            with self.assertRaises(InterruptedError): call()
                        else:
                            response = call()
                            self.assertEqual(response.choices[0].finish_reason, 'tool_calls')
                            calls = response.choices[0].message.tool_calls
                            self.assertEqual(len(calls), 1)
                            self.assertEqual(calls[0].id, 'agy_2_run_command')
                            self.assertEqual(json.loads(calls[0].function.arguments), {'command':'fixture-only'})
                            self.assertIsNotNone(response.usage)
                        self.assertEqual(observed, [('terminal', True)])
                        self.assertEqual(producer.Popen.call_count, 1, 'tool request was silently replayed')
                    finally:
                        self.agent._interrupt_requested = False

    def test_actual_nonstream_dispatch_and_no_post_response_duplicate(self):
        self.agent._disable_streaming = True
        self.assertFalse(_should_stream(self.agent))
        response = self.perform()
        self.assertEqual(len(self.reasoning_frames()),1)
        self.assertEqual(response.choices[0].message.reasoning_content,'private thinking')
        self.assertEqual(response.choices[0].message.content,'answer')
        from agent.chat_completion_helpers import _assistant_reasoning_text
        self.assertEqual(_assistant_reasoning_text(self.agent,response.choices[0].message),'private thinking')
        self.assertEqual(len(self.reasoning_frames()),1)
        self.assertEqual(response.usage.total_tokens,7)

    def test_hidden_and_muted_keep_response_but_no_event(self):
        server._sessions[self.sid]['show_reasoning'] = False
        self.assertEqual(self.perform().choices[0].message.reasoning_content,'private thinking')
        self.assertEqual(self.reasoning_frames(),[])
        server._sessions[self.sid]['show_reasoning'] = True
        with notification_turn(self.agent,muted=True,session_id=self.sid):
            self.assertEqual(self.perform().choices[0].message.reasoning_content,'private thinking')
        self.assertEqual(self.reasoning_frames(),[])

    def test_stream_true_uses_chunks_without_adapter_duplicate(self):
        self.agent._model_request_active.set()
        with patch.object(self.client_module,'is_authenticated',return_value=True), \
             patch.object(self.client_module.AntigravityClient,'_get_or_spawn_worker',return_value=self.proc()), \
             bind_subagent_parent(self.agent):
            def downstream(req):
                # Even a downstream middleware switching to stream=True cannot
                # cause the collector to emit; native consumer owns delivery.
                stream = self.agent.client.chat.completions.create(**{**req,'stream':True})
                chunks = list(stream)
                self.assertTrue(any(getattr(c.delta,'reasoning_content',None)
                    for chunk in chunks for c in chunk.choices))
            self.bridge.execution_middleware(request=self.request,next_call=downstream,
                provider='antigravity-oauth',session_id=self.agent.session_id,turn_id='turn',api_request_id='req',model=self.agent.model)
        self.assertEqual(self.reasoning_frames(),[])

    def test_provider_lifecycle_projection_idempotent_and_reinstalls_after_disposal(self):
        manager = get_plugin_manager()
        self.assertTrue(self.bridge._registrations[manager].active)
        before = len(manager._middleware['llm_execution'])
        self.profile.build_api_kwargs_extras(session_id='x')
        self.assertEqual(len(manager._middleware['llm_execution']),before)
        self.bridge._registrations[manager].dispose()
        self.profile.build_api_kwargs_extras(session_id='x')
        self.assertTrue(self.bridge._registrations[manager].active)
        self.assertEqual(len(manager._middleware['llm_execution']),before)
        self.assertNotIn('_agy_session_id',self.profile.build_api_kwargs_extras()[1])

    def test_launch_profile_lookup_and_closing_guard(self):
        self.agent._disable_streaming = True  # optional nonstream bridge guard
        live = self.bridge._live_session(self.agent.session_id,self.bridge._home())
        self.assertEqual(live[0],self.sid)
        server._sessions[self.sid]['_closing'] = True
        self.assertEqual(self.perform().choices[0].message.reasoning_content,'private thinking')
        self.assertEqual(self.reasoning_frames(),[])


    def test_oneshot_nonstream_collector_delivers_once(self):
        self.agent._model_request_active.set()
        client = self.agent.client
        client._worker_lock.acquire()
        try:
            with patch.object(self.client_module,'is_authenticated',return_value=True), \
                 patch.object(self.client_module.subprocess,'Popen',return_value=self.proc()), \
                 bind_subagent_parent(self.agent):
                response = self.bridge.execution_middleware(request=self.request,
                    next_call=lambda req:client.chat.completions.create(**req),
                    provider='antigravity-oauth',session_id=self.agent.session_id,
                    turn_id='turn',api_request_id='req',model=self.agent.model)
            self.assertEqual(response.choices[0].message.reasoning_content,'private thinking')
            self.assertEqual(len(self.reasoning_frames()),1)
        finally:
            client._worker_lock.release()


if __name__ == '__main__':
    unittest.main()
