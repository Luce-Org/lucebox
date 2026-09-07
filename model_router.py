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
        self.models, self.default = config['models'], config['default']
        self.url = config.get('backend_url','http://127.0.0.1:18216')
        self.limits = {'max_queue':4,'queue_timeout':120,'request_timeout':3600,
                       'prefill_timeout':600,'token_timeout':120,'cancel_grace':5,
                       'stop_grace':5,'load_timeout':180,'body_timeout':15}
        self.limits.update(config.get('serving',{}))
        self.active = self.process = self.log = self.client = None
        self.lock = asyncio.Lock()
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
        if self.log:
            self.log.close()
            self.log = None

    async def load(self, name):
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
            args = [str(executable),conf['path'],
                    '--draft',self.config['draft'],'--draft-block-size','16',
                    '--prefix-cache-slots','2','--max-ctx',str(conf['context']),
                    '--cache-type-k','q8_0','--cache-type-v','q8_0',
                    '--host',address.hostname,'--port',str(address.port or 80),'--model-name',name]
            args.extend(backend.get('extra_args',[]))
            env = os.environ.copy()
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
        return web.json_response({'status':status,'active_model':self.active,'busy':self.lock.locked(),
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
            while True:
                if state.streaming:
                    limit = self.limits['prefill_timeout'] if first else self.limits['token_timeout']
                    timeout = max(0.001,limit-(time.monotonic()-progress))
                else:
                    timeout = self.limits['request_timeout']
                chunk = await asyncio.wait_for(upstream.content.readany(),timeout)
                if not chunk:
                    break
                if state.streaming:
                    pending += chunk
                    if len(pending) > 2*1024*1024:
                        raise BackendFault('Backend SSE frame exceeds limit')
                    while b'\n' in pending:
                        line,pending = pending.split(b'\n',1)
                        if not line.startswith(b'data:'):
                            continue # Heartbeats must not hide a stalled worker.
                        try:
                            event = json.loads(line[5:])
                        except (ValueError,TypeError):
                            continue
                        if not isinstance(event,dict):
                            raise BackendFault('Invalid backend SSE event')
                        choices = event.get('choices',[])
                        if not isinstance(choices,list):
                            raise BackendFault('Invalid backend SSE choices')
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
            await self.settle_cancelled_backend()
            return state.response or web.Response(status=499)
        except (asyncio.TimeoutError,ClientError,BackendFault) as e:
            self.counts['failed'] += 1
            operation.cancel()
            with contextlib.suppress(asyncio.CancelledError,Exception):
                await operation
            timed_out = isinstance(e,asyncio.TimeoutError)
            result = await self.fail_forward(state,504 if timed_out else 502,
                'inference_timeout' if timed_out else 'backend_failure',
                'Inference made no progress within its deadline; retry the request.' if timed_out else 'Backend connection failed; retry the request.')
            self.counts['backend_recycles'] += 1
            await self.stop()
            return result
        except asyncio.CancelledError:
            operation.cancel()
            with contextlib.suppress(asyncio.CancelledError,Exception):
                await operation
            await self.settle_cancelled_backend()
            raise
        finally:
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError,Exception):
                await watch

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
        if self.pending >= self.limits['max_queue']+1:
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
            name = data.get('model',self.default)
            if name == 'dflash':
                name = self.default
            if not isinstance(name,str) or name not in self.models:
                return self.error(404,'model_not_found','Unknown model. Use /v1/models.')
            data['model'] = name
            try:
                await asyncio.wait_for(self.lock.acquire(),self.limits['queue_timeout'])
                acquired = True
            except asyncio.TimeoutError:
                self.counts['queue_rejected'] += 1
                return self.error(429,'queue_timeout','Inference queue wait expired; retry later')
            if request.transport is None or request.transport.is_closing():
                return web.Response(status=499)
            try:
                await self.load(name)
            except (BackendFault,asyncio.TimeoutError,ClientError,OSError):
                logging.exception('Model load failed')
                return self.error(503,'backend_unavailable','Model could not be loaded; retry later')
            self.counts['requests'] += 1
            return await self.run_request(request,data)
        finally:
            if acquired:
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
