"""Opt-in real agy probe through unmodified Hermes nonstream dispatch.

Run explicitly, not test discovery. Uses the existing account/bare Argon model.
Only UI egress (write_json) is captured in-process; no live gateway is restarted.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path.home() / '.hermes' / 'hermes-agent'))
import hermes_bootstrap  # activate the installed host dependencies before other Hermes imports
from hermes_cli.plugins_discovery import suppress_plugin_discovery


def main():
    # Suppress unrelated plugins/background jobs in this isolated probe process.
    # Provider discovery and its real PluginContext registration remain enabled.
    with suppress_plugin_discovery():
        from run_agent import AIAgent
        from agent.turn_api_call import _should_stream, perform_api_call
        from agent.subagent_lifecycle import bind_subagent_parent
        from providers import get_provider_profile
        from hermes_cli.plugins import get_plugin_manager
        from tui_gateway import server
        profile = get_provider_profile('antigravity-oauth')
        assert profile.auth_type == 'external_process'
        sid = 'agy-live-nonstream-probe'
        session_id = 'agy-live-nonstream-' + str(os.getpid())
        frames = []
        result_times = []
        collector_threads = set()
        started = time.monotonic()
        def observe(frame, event, arg):
            if (event == 'return' and frame.f_code.co_name == '_read_line'
                    and frame.f_code.co_filename.endswith('/antigravity-oauth/stream.py')
                    and isinstance(arg, str) and arg):
                try:
                    if json.loads(arg).get('event') == 'result':
                        result_times.append(time.monotonic()-started)
                except ValueError:
                    pass
            if event == 'call' and frame.f_code.co_name == 'collect_stream_completion':
                collector_threads.add(threading.current_thread().name)
        def capture(frame):
            # Do not print private reasoning bodies; retain only timing/counts.
            frames.append((time.monotonic()-started, frame))
            return True
        with tempfile.TemporaryDirectory(prefix='agy-main-path-') as cwd:
            agent = AIAgent(provider='antigravity-oauth', model='gemini-4-argon',
                base_url='agy://local', api_mode='chat_completions', session_id=session_id,
                enabled_toolsets=[], disabled_toolsets=['all'], quiet_mode=True,
                skip_context_files=True, skip_memory=True, skip_background_review=True,
                load_soul_identity=False, max_iterations=1, platform='tui', cwd=cwd,
                reasoning_callback=server._agent_cbs(sid)['reasoning_callback'],
                stream_delta_callback=lambda text: None)
            server._sessions[sid] = {'agent':agent, 'session_key':session_id,
                'running':True, 'show_reasoning':True, 'profile_home':None}
            agent._current_turn_id = session_id + ':turn'
            agent._current_api_request_id = agent._current_turn_id + ':api:1'
            request = agent._build_api_kwargs([{'role':'user','content':
                'Do not use any tools. Solve carefully: a bag has 4 red, 5 blue, and 6 green balls. Draw 6 without replacement. Given that each color appears, what is the exact probability that exactly two colors have equal counts? Enumerate the relevant count patterns and check the conditional denominator. Give a compact derivation and reduced fraction.'}], tools_for_api=[])
            request['timeout'] = 90
            assert request['_agy_session_id'] == session_id
            assert _should_stream(agent), 'Provider must opt into native streaming'
            agent._disable_streaming = True  # this probe specifically tests the legacy collector
            assert not _should_stream(agent)
            callbacks = get_plugin_manager()._middleware.get('llm_execution', [])
            assert any(c.__module__.endswith('.live_reasoning') for c in callbacks)
            started = time.monotonic()
            try:
                threading.setprofile(observe)
                with patch.object(server,'write_json',side_effect=capture), bind_subagent_parent(agent):
                    verdict = perform_api_call(agent, api_kwargs=request, _original_api_kwargs=dict(request),
                        _llm_middleware_trace=[], _moa_prepared_request=None, _retry=SimpleNamespace(),
                        thinking_spinner=None,retry_count=0,api_call_count=1,
                        api_request_id=agent._current_api_request_id,effective_task_id=session_id,
                        turn_id=agent._current_turn_id,interrupted=False)
                completed = time.monotonic()-started
                message = verdict.response.choices[0].message
                reasoning_events = [(t,f) for t,f in frames
                                    if f.get('params',{}).get('type') == 'reasoning.delta']
                receipt = {'default_should_stream':True,'probe_should_stream':False,'auth_type':profile.auth_type,
                    'model':agent.model,'middleware_registered':True,
                    'completion_seconds':completed,'reasoning_chars':len(message.reasoning_content or ''),
                    'answer_chars':len(message.content or ''),'finish_reason':verdict.response.choices[0].finish_reason,
                    'reasoning_event_count':len(reasoning_events),
                    'first_reasoning_event_seconds':reasoning_events[0][0] if reasoning_events else None,
                    'agy_result_consumed_seconds':result_times,
                    'collector_threads':sorted(collector_threads),
                    'event_keys':[list(f.keys()) for _,f in frames],
                    'usage':vars(verdict.response.usage),
                    'boundary':'real AIAgent build/dispatch + middleware + agy + DB + collector + installed TUI callback; write_json captured, no renderer'}
                rendered = json.dumps(receipt,default=lambda x:vars(x),indent=2)
                Path(__file__).resolve().parent.parent.joinpath('verification-live-reasoning-final.json').write_text(rendered + '\n')
                print(rendered)
                assert message.content and not message.tool_calls
                assert reasoning_events and reasoning_events[0][0] < completed
                assert result_times and reasoning_events[0][0] < result_times[0]
            finally:
                threading.setprofile(None)
                server._sessions.pop(sid,None)
                agent.client.close()
                print('cleanup: agent client closed, isolated session removed',file=sys.stderr)


if __name__ == '__main__':
    main()
