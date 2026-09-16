#!/usr/bin/env python3
"""Bounded single-GPU serving gateway; native Lucebox owns cache admission."""
import asyncio
import contextlib
import json
import logging
import os
import pathlib
import time
from collections import Counter, deque
from dataclasses import dataclass
from urllib.parse import urlsplit
from aiohttp import web, ClientSession, ClientTimeout, ClientError

ROOT = pathlib.Path(os.environ.get('LUCEBOX_ROOT', pathlib.Path(__file__).resolve().parent))
HOP = {'connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer','transfer-encoding','upgrade','host','content-length'}
CORS = {'Access-Control-Allow-Origin':'*'}

def normalize_session_id(body, headers):
    extra = body.get('extra_body')
    values = [body.get('session_id')]
    if isinstance(extra, dict):
        values.append(extra.get('session_id'))
    for key in ('session_id', 'x-session-affinity', 'x-session-id'):
        values.append(headers.get(key))
    supplied = [v for v in values if v is not None and v != '']
    if not supplied:
        return
    if any(not isinstance(v, str) or len(v.encode('utf-8')) > 256 or
           any(ord(c) < 32 or ord(c) == 127 for c in v) for v in supplied):
        raise ValueError('Session ID must be a string of at most 256 UTF-8 bytes without control characters')
    if len(set(supplied)) != 1:
        raise ValueError('Conflicting session identifiers')
    body['session_id'] = supplied[0]

def normalize_cache_metadata(body, headers):
    supplied = {field: headers.get('x-lucebox-cache-' + field)
                for field in ('owner', 'purpose', 'mode')}
    if not any(value is not None for value in supplied.values()):
        return
    extra = body.setdefault('extra_body', {})
    if not isinstance(extra, dict):
        raise ValueError('extra_body must be an object')
    meta = extra.setdefault('lucebox_cache', {})
    if not isinstance(meta, dict):
        raise ValueError('lucebox_cache must be an object')
    for field, value in supplied.items():
        if value is None:
            continue
        if field in meta and meta[field] != value:
            raise ValueError('Conflicting cache metadata: ' + field)
        if not isinstance(value, str) or len(value.encode('utf-8')) > 256 or any(ord(c)<32 or ord(c)==127 for c in value):
            raise ValueError('Invalid cache metadata: ' + field)
        if field == 'purpose' and value not in ('chat', 'compaction'):
            raise ValueError('Invalid cache purpose')
        if field == 'mode' and value not in ('auto', 'exact'):
            raise ValueError('Invalid cache mode')
        meta[field] = value

class TokenProgress:
    """Per-upstream-request committed progress; generic heartbeats never count."""
    def __init__(self):
        self.request_id = None
        self.tokens = 0

    def observe(self, line):
        if not line.startswith(b':lucebox-progress '):
            return False
        try:
            value = json.loads(line[len(b':lucebox-progress '):])
            request_id, tokens = value['id'], value['tokens']
            if not isinstance(request_id, str) or not request_id or type(tokens) is not int or tokens <= self.tokens:
                return False
            if self.request_id is not None and request_id != self.request_id:
                return False
            self.request_id, self.tokens = request_id, tokens
            return True
        except (ValueError, KeyError, TypeError):
            return False

class BackendFault(Exception):
    pass

class ClientGone(Exception):
    pass

@dataclass
class ForwardState:
    response: object = None
    streaming: bool = False

class Router:
    def __init__(self, config, root=ROOT):
        self.config, self.root = config, root
        self.prefix_cache_device = config.get('backend', {}).get('prefix_cache_device')
        if self.prefix_cache_device not in (None, 'cpu', 'gpu'):
            raise ValueError('backend.prefix_cache_device must be cpu or gpu')
        self.models, self.default = config['models'], config['default']
        for name, model in self.models.items():
            args = model.get('extra_args', [])
            if not isinstance(args, list) or any(not isinstance(arg, str) or '\0' in arg for arg in args):
                raise ValueError(f'{name}.extra_args must be a list of strings without NUL')
            canonical = model.get('canonical', name)
            if canonical not in self.models or self.models[canonical].get('canonical', canonical) != canonical:
                raise ValueError(f'{name}.canonical must name a non-alias model')
            if self.models[canonical]['path'] != model['path']:
                raise ValueError(f'{name}.canonical must use the same weights')
            if model.get('cache_default_policy', 'exact') not in ('exact', 'auto'):
                raise ValueError(f'{name}.cache_default_policy must be exact or auto')
            if model.get('hybrid_cache', 'off') not in ('off', 'shadow', 'auto'):
                raise ValueError(f'{name}.hybrid_cache must be off, shadow or auto')
            if type(model.get('hybrid_projections', False)) is not bool:
                raise ValueError(f'{name}.hybrid_projections must be boolean')
            window = model.get('compression_score_window', 0)
            if type(window) is not int or (window != 0 and not 512 <= window <= 65536):
                raise ValueError(f'{name}.compression_score_window must be 0 or 512..65536')
        self.url = config.get('backend_url','http://127.0.0.1:18216')
        self.limits = {'max_queue':4,'queue_timeout':120,'request_timeout':3600,
                       'prefill_timeout':600,'token_timeout':120,'cancel_grace':5,
                       'stop_grace':5,'load_timeout':180,'body_timeout':15}
        self.limits.update(config.get('serving',{}))
        self.active = self.process = self.log = self.client = None
        self.lock = asyncio.Lock()
        self.capacity = max(1, int(self.limits.get('max_concurrency', 1)))
        self.condition = asyncio.Condition()
        self.waiters = deque()
        self.inflight = 0
        self.recycle_pending = False
        self.pending = 0
        self.closing = False
        self.loading = False
        self.counts = Counter()
        self.starts = deque()
        self.warmup = None

    async def start(self, app):
        self.client = ClientSession(timeout=ClientTimeout(total=None,sock_connect=10),auto_decompress=False)
        self.warmup = asyncio.create_task(self.initialize())

    async def initialize(self):
        async with self.lock:
            try:
                await self.load(self.default)
            except Exception:
                logging.exception('Initial model load failed; bounded retry on next request')

    async def shutdown(self, app):
        self.closing = True
        if self.warmup and not self.warmup.done():
            self.warmup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.warmup

    async def cleanup(self, app):
        await self.stop()
        await self.client.close()

    async def stop(self):
        stopped_at = time.monotonic()
        process, self.process = self.process, None
        self.active = None
        if process and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(),self.limits['stop_grace'])
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        if process:
            logging.info('Backend stopped: elapsed=%.3fs exit=%s', time.monotonic()-stopped_at, process.returncode)
        if self.log:
            self.log.close()
            self.log = None

    async def load(self, name):
        name = self.models[name].get('canonical', name)
        if self.closing:
            raise BackendFault('Server is shutting down')
        if self.active == name and self.process and self.process.returncode is None:
            return
        # Bound repeated failed launches without penalizing healthy model switches.
        now = time.monotonic()
        while self.starts and now-self.starts[0] > 60:
            self.starts.popleft()
        if len(self.starts) >= 3:
            raise BackendFault('Backend restart cooldown; retry in 60 seconds')
        await self.stop()
        self.starts.append(now)
        self.loading = True
        try:
            conf = self.models[name]
            backend = self.config.get('backend', {})
            address = urlsplit(self.url)
            if address.scheme != 'http' or address.hostname not in ('127.0.0.1','localhost','::1'):
                self.loading = False
                raise BackendFault('Managed backend_url must use loopback HTTP')
            executable = pathlib.Path(backend.get('executable','server/build-hip/dflash_server'))
            if not executable.is_absolute():
                executable = self.root/executable
            prefix_slots = str(self.limits.get('prefix_cache_slots',4))
            concurrency = int(conf.get('max_concurrency', self.capacity))
            draft = conf.get('draft',self.config.get('draft'))
            args = [str(executable),conf['path']]
            if draft:
                args += ['--draft',draft]
                if conf.get('draft_block_size'):
                    args += ['--draft-block-size',str(conf['draft_block_size'])]
            args += ['--prefix-cache-slots',prefix_slots,'--max-ctx',str(conf['context']),
                    '--cache-type-k','q8_0','--cache-type-v','q8_0',
                    '--host',address.hostname,'--port',str(address.port or 80),'--model-name',name]
            if concurrency > 1:
                args += ['--max-concurrency',str(concurrency),
                         '--concurrent-prefix-cache-max-mib',str(self.limits.get('prefix_cache_max_mib',4096))]
                if conf.get('kv_pool_tokens'):
                    args += ['--kv-pool-tokens',str(conf['kv_pool_tokens'])]
                if self.limits.get('session_prefix_cache_max_tokens'):
                    args += ['--session-prefix-cache-max-tokens',
                             str(self.limits['session_prefix_cache_max_tokens'])]
            args.extend(backend.get('extra_args',[]))
            args.extend(conf.get('extra_args', []))
            env = os.environ.copy()
            env['DFLASH_COMPRESS_SCORE_WINDOW'] = str(conf.get('compression_score_window', 0))
            env['DFLASH_HYBRID_PROJECTIONS'] = '1' if conf.get('hybrid_projections', False) else '0'
            env['DFLASH_HYBRID_CACHE'] = conf.get('hybrid_cache', 'off')
            env['DFLASH_HYBRID_CALIBRATION'] = conf.get('hybrid_calibration', '')
            if self.prefix_cache_device is not None:
                env['DFLASH_PREFIX_CACHE_DEVICE'] = self.prefix_cache_device
            library_path = backend.get('library_path')
            if library_path:
                env['LD_LIBRARY_PATH'] = library_path+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            for key,var in [('bytes_per_token','LUCEBOX_RAM_BYTES_PER_TOKEN'),
                            ('work_bytes_per_token','LUCEBOX_RAM_WORK_BYTES_PER_TOKEN'),
                            ('snapshot_max_bytes','LUCEBOX_SNAPSHOT_MAX_BYTES'),
                            ('fixed_bytes','LUCEBOX_RAM_FIXED_BYTES'),('reserve_bytes','LUCEBOX_RAM_RESERVE_BYTES')]:
                if key in self.config.get('memory',{}):
                    env[var] = str(self.config['memory'][key])
            log_dir = pathlib.Path(os.environ.get('LUCEBOX_LOG_DIR',self.root))
            self.log = open(log_dir/'model-backend.log','ab',buffering=0)
            logging.info('Loading %s',name)
            self.process = await asyncio.create_subprocess_exec(*args,env=env,stdout=self.log,stderr=self.log)
            async with asyncio.timeout(self.limits['load_timeout']):
                while True:
                    if self.process.returncode is not None:
                        raise BackendFault(f'Backend exited ({self.process.returncode})')
                    try:
                        async with self.client.get(self.url+'/health',timeout=ClientTimeout(total=2)) as r:
                            if r.status == 200:
                                self.active = name
                                self.starts.clear()
                                self.counts['loads'] += 1
                                logging.info('Ready: %s',name)
                                return
                    except (ClientError,asyncio.TimeoutError):
                        pass
                    await asyncio.sleep(0.25)
        except BaseException:
            await self.stop()
            raise
        finally:
            self.loading = False

    def error(self, status, code, message):
        headers = dict(CORS)
        if status in (429,503):
            headers['Retry-After'] = '5'
        return web.json_response({'error':{'type':'server_error','code':code,'message':message}},status=status,headers=headers)

    async def health(self, request):
        if request.path == '/livez':
            return web.json_response({'status':'alive' if not self.closing else 'stopping'},status=503 if self.closing else 200)
        ready = bool(not self.closing and self.active and self.process and self.process.returncode is None)
        status = 'loading' if self.loading else 'unloaded'
        if ready:
            try:
                async with self.client.get(self.url+'/health',timeout=ClientTimeout(total=3)) as r:
                    ready = r.status == 200
                status = 'ok' if ready else 'unavailable'
            except (ClientError,asyncio.TimeoutError):
                ready, status = False, 'unresponsive'
        return web.json_response({'status':status,'active_model':self.active,'busy':bool(self.inflight or self.lock.locked()),'active_requests':self.inflight,'max_concurrency':self.capacity,
                                  'pending_requests':self.pending,'counters':dict(self.counts)},status=200 if ready else 503,headers=CORS)

    async def client_watch(self, request):
        while True:
            if request.transport is None or request.transport.is_closing():
                raise ClientGone()
            await asyncio.sleep(0.1)

    async def settle_cancelled_backend(self):
        # Closing the upstream socket activates Lucebox's existing cancellation.
        # Do not give the worker to the next request until idle, or recycle it.
        deadline = time.monotonic()+self.limits['cancel_grace']
        while time.monotonic() < deadline:
            try:
                async with self.client.get(self.url+'/status/json',timeout=ClientTimeout(total=0.5)) as r:
                    if r.status == 200 and (await r.json()).get('phase') == 'idle':
                        return
            except (ClientError,asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.1)
        self.counts['backend_recycles'] += 1
        await self.stop()

    async def forward(self, request, body, state):
        headers = {k:v for k,v in request.headers.items() if k.lower() not in HOP}
        async with self.client.request(request.method,self.url+request.raw_path,data=body,headers=headers,allow_redirects=False) as upstream:
            state.streaming = upstream.headers.get('Content-Type','').startswith('text/event-stream')
            if not state.streaming:
                # Hold nonstream headers until the complete body is available.
                # A backend crash can then become a real HTTP 502, not a broken 200.
                payload = bytearray()
                async for chunk in upstream.content.iter_chunked(65536):
                    payload.extend(chunk)
                    if len(payload)>8*1024*1024:
                        raise BackendFault('Backend response exceeds 8 MiB')
                headers = {k:v for k,v in upstream.headers.items() if k.lower() not in HOP}
                headers.update(CORS)
                if upstream.status in (429,503): headers['Retry-After']='5'
                if upstream.status>=400: self.counts['upstream_errors']+=1
                return web.Response(status=upstream.status,headers=headers,body=payload)
            out = state.response = web.StreamResponse(status=upstream.status,headers={k:v for k,v in upstream.headers.items() if k.lower() not in HOP})
            out.headers.update(CORS)
            if upstream.status in (429,503):
                out.headers['Retry-After'] = '5'
            await out.prepare(request)
            if upstream.status >= 400:
                self.counts['upstream_errors'] += 1
            first = True
            progress = time.monotonic()
            pending = b''
            committed = TokenProgress()
            require_finish = request.path.endswith('/chat/completions') and upstream.status < 400
            terminal = False
            while True:
                if state.streaming:
                    limit = self.limits['prefill_timeout'] if first else self.limits['token_timeout']
                    timeout = max(0.001,limit-(time.monotonic()-progress))
                else:
                    timeout = self.limits['request_timeout']
                chunk = await asyncio.wait_for(upstream.content.readany(),timeout)
                if not chunk:
                    if require_finish and not terminal:
                        raise BackendFault('Backend stream ended before finish_reason')
                    break
                if state.streaming:
                    pending += chunk
                    if len(pending) > 2*1024*1024:
                        raise BackendFault('Backend SSE frame exceeds limit')
                    while b'\n' in pending:
                        line,pending = pending.split(b'\n',1)
                        if committed.observe(line):
                            first = False
                            progress = time.monotonic()
                        if not line.startswith(b'data:'):
                            continue # Heartbeats must not hide a stalled worker.
                        if line[5:].strip() == b'[DONE]':
                            if require_finish and not terminal:
                                raise BackendFault('Backend sent DONE before finish_reason')
                            continue
                        try:
                            event = json.loads(line[5:])
                        except (ValueError,TypeError):
                            continue
                        if not isinstance(event,dict):
                            raise BackendFault('Invalid backend SSE event')
                        choices = event.get('choices',[])
                        if not isinstance(choices,list):
                            raise BackendFault('Invalid backend SSE choices')
                        terminal |= bool(event.get('error')) or any(isinstance(c,dict) and c.get('finish_reason') is not None for c in choices)
                        substantive = False
                        for choice in choices:
                            delta = choice.get('delta',{}) if isinstance(choice,dict) else {}
                            if isinstance(delta,dict):
                                substantive |= any(delta.get(k) for k in ('content','reasoning_content','reasoning','tool_calls'))
                        substantive |= bool(event.get('delta',{})) # Anthropic content deltas
                        substantive |= event.get('type','') in ('response.output_text.delta','response.reasoning_text.delta','response.function_call_arguments.delta')
                        if substantive:
                            first = False
                            progress = time.monotonic()
                await out.write(chunk)
            await out.write_eof()
            return out

    async def fail_forward(self, state, status, code, message):
        out = state.response
        if out is None or not out.prepared:
            return self.error(status,code,message)
        if state.streaming:
            payload = json.dumps({'error':{'type':'server_error','code':code,'message':message}})
            with contextlib.suppress(ConnectionError,RuntimeError):
                await out.write(('data: '+payload+'\n\ndata: [DONE]\n\n').encode())
                await out.write_eof()
        else:
            # A started nonstream response cannot have its HTTP status rewritten.
            # Close it as incomplete instead of reporting a successful completion.
            out.force_close()

        return out

    async def run_request(self, request, data):
        model = self.models.get(data.get('model'), {})
        if model.get('hybrid_cache', 'off') != 'off' and request.path != '/cache/release':
            extra = data.setdefault('extra_body', {})
            if not isinstance(extra, dict):return self.error(400,'invalid_request','extra_body must be an object')
            meta = extra.setdefault('lucebox_cache', {})
            if not isinstance(meta, dict):return self.error(400,'invalid_request','lucebox_cache must be an object')
            meta.setdefault('mode', model.get('cache_default_policy', 'exact'))
            deadline = int((time.time()+self.limits['request_timeout'])*1000)
            supplied = meta.get('deadline_unix_ms', deadline)
            if type(supplied) is not int or supplied < 0:
                return self.error(400,'invalid_request','Invalid request deadline')
            meta['deadline_unix_ms'] = min(supplied or deadline, deadline)
        state = ForwardState()
        operation = asyncio.create_task(self.forward(request,json.dumps(data).encode(),state))
        watch = asyncio.create_task(self.client_watch(request))
        try:
            async with asyncio.timeout(self.limits['request_timeout']):
                done,_ = await asyncio.wait([operation,watch],return_when=asyncio.FIRST_COMPLETED)
                if watch in done:
                    await watch
                return await operation
        except (ClientGone,ConnectionResetError,BrokenPipeError):
            self.counts['cancelled'] += 1
            operation.cancel()
            with contextlib.suppress(asyncio.CancelledError,Exception):
                await operation
            await self.recover_request(cancelled=True)
            return state.response or web.Response(status=499)
        except (asyncio.TimeoutError,ClientError,BackendFault) as e:
            self.counts['failed'] += 1
            logging.error('Inference failed: %s: %s; backend_returncode=%s', type(e).__name__, e, self.process.returncode if self.process else None)
            operation.cancel()
            with contextlib.suppress(asyncio.CancelledError,Exception):
                await operation
            timed_out = isinstance(e,asyncio.TimeoutError)
            result = await self.fail_forward(state,504 if timed_out else 502,
                'inference_timeout' if timed_out else 'backend_failure',
                'Inference made no progress within its deadline; retry the request.' if timed_out else 'Backend connection failed or stream was incomplete; retry the request.')
            await self.recover_request()
            return result
        except asyncio.CancelledError:
            operation.cancel()
            with contextlib.suppress(asyncio.CancelledError,Exception):
                await operation
            await self.recover_request(cancelled=True)
            raise
        finally:
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError,Exception):
                await watch

    async def acquire_slot(self, name):
        if self.warmup:
            await asyncio.shield(self.warmup)
        ticket = object()
        async with self.condition:
            self.waiters.append(ticket)
            try:
                while True:
                    limit = int(self.models[name].get('max_concurrency',self.capacity))
                    if (self.waiters[0] is ticket and not self.recycle_pending and
                        self.inflight < limit and
                        (self.inflight == 0 or self.active == name)):
                        if self.inflight == 0:
                            await self.load(name)
                        self.inflight += 1
                        return
                    await self.condition.wait()
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    async def release_slot(self):
        async with self.condition:
            self.inflight -= 1
            if self.inflight == 0 and self.recycle_pending:
                try:
                    await self.stop()
                finally:
                    self.recycle_pending = False
            self.condition.notify_all()

    async def recover_request(self, cancelled=False):
        if self.capacity == 1:
            if cancelled:
                await self.settle_cancelled_backend()
            else:
                self.counts['backend_recycles'] += 1
                await self.stop()
        elif not cancelled:
            # A failing stream must not terminate peers sharing the model.
            # Stop admitting work and recycle once the active cohort drains.
            self.recycle_pending = True
            self.counts['backend_recycles'] += 1

    async def handle(self, request):
        if request.method == 'OPTIONS':
            return web.Response(status=204,headers={**CORS,'Access-Control-Allow-Headers':'*','Access-Control-Allow-Methods':'GET,POST,OPTIONS'})
        if request.method == 'GET' and request.path == '/v1/models':
            return web.json_response({'object':'list','data':[{'id':n,'object':'model','owned_by':'local','context_length':c['context'],'max_context_length':c['context']} for n,c in self.models.items()]},headers=CORS)
        if request.path in ('/health','/readyz','/livez'):
            return await self.health(request)
        if request.method != 'POST':
            state = ForwardState()
            try:
                async with asyncio.timeout(None if request.path=='/status/events' else 10):
                    return await self.forward(request,None,state)
            except (ClientError,asyncio.TimeoutError,BackendFault):
                return await self.fail_forward(state,503,'backend_unavailable','Backend unavailable')
        if self.closing:
            return self.error(503,'shutting_down','Server is shutting down')
        if self.pending >= self.limits['max_queue']+self.capacity:
            self.counts['queue_rejected'] += 1
            return self.error(429,'server_busy','Inference queue is full; retry later')
        self.pending += 1
        acquired = False
        try:
            try:
                async with asyncio.timeout(self.limits['body_timeout']):
                    data = await request.json()
            except web.HTTPRequestEntityTooLarge:
                return self.error(413,'request_too_large','Request body exceeds 8 MiB')
            except (ValueError,asyncio.TimeoutError):
                return self.error(400,'invalid_request','Expected a JSON object within the upload deadline')
            if not isinstance(data,dict):
                return self.error(400,'invalid_request','Expected a JSON object')
            try:
                normalize_session_id(data, request.headers)
                normalize_cache_metadata(data, request.headers)
            except ValueError as exc:
                return self.error(400, 'invalid_session_id', str(exc))
            name = data.get('model',self.default)
            if name == 'dflash':
                name = self.default
            if not isinstance(name,str) or name not in self.models:
                return self.error(404,'model_not_found','Unknown model. Use /v1/models.')
            name = self.models[name].get('canonical', name)
            data['model'] = name
            try:
                if self.capacity > 1:
                    await asyncio.wait_for(self.acquire_slot(name),self.limits['queue_timeout'])
                else:
                    await asyncio.wait_for(self.lock.acquire(),self.limits['queue_timeout'])
                acquired = True
            except asyncio.TimeoutError:
                self.counts['queue_rejected'] += 1
                return self.error(429,'queue_timeout','Inference queue wait expired; retry later')
            except (BackendFault,ClientError,OSError):
                logging.exception('Concurrent model load failed')
                return self.error(503,'backend_unavailable','Model could not be loaded; retry later')
            if request.transport is None or request.transport.is_closing():
                return web.Response(status=499)
            try:
                if self.capacity == 1:
                    await self.load(name)
            except (BackendFault,asyncio.TimeoutError,ClientError,OSError):
                logging.exception('Model load failed')
                return self.error(503,'backend_unavailable','Model could not be loaded; retry later')
            self.counts['requests'] += 1
            return await self.run_request(request,data)
        finally:
            if acquired:
                if self.capacity > 1:
                    await self.release_slot()
                else:
                    self.lock.release()
            self.pending -= 1

def create_app(config, router=None):
    router = router or Router(config)
    app = web.Application(client_max_size=8*1024*1024)
    app.on_startup.append(router.start)
    app.on_shutdown.append(router.shutdown)
    app.on_cleanup.append(router.cleanup)
    app.router.add_route('*','/{path:.*}',router.handle)
    return app

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config_path = pathlib.Path(os.environ.get('LUCEBOX_CONFIG',ROOT/'models.json'))
    config = json.loads(config_path.read_text())
    web.run_app(create_app(config),host=config.get('listen_host','127.0.0.1'),port=config.get('listen_port',8216),handler_cancellation=True,shutdown_timeout=10)
