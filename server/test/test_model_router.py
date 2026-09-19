import asyncio
import contextlib
import unittest
import pathlib
import tempfile
from unittest.mock import AsyncMock, patch
from aiohttp import web, ClientSession
from aiohttp.test_utils import TestServer, TestClient
from model_router import Router, create_app

class Process:
    returncode = None

class FakeRouter(Router):
    async def load(self,name):
        self.active=name
        self.process=Process()
        self.counts['loads']+=1
    async def stop(self):
        self.process=self.active=None
        self.counts['stops']+=1
    async def settle_cancelled_backend(self):
        self.counts['cancel_settled']+=1
        await self.stop()

class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.entered=asyncio.Event()
        self.release=asyncio.Event()
        self.backend_calls=0
        app=web.Application()
        app.router.add_route('*','/{path:.*}',self.backend)
        self.backend_server=TestServer(app)
        await self.backend_server.start_server()
        config={'default':'test','models':{'test':{'path':'unused','context':131072}},'draft':'unused',
                'backend_url':str(self.backend_server.make_url('')).rstrip('/'),
                'serving':{'max_queue':1,'queue_timeout':0.25,'prefill_timeout':0.15,'token_timeout':0.1,'request_timeout':2}}
        self.router=FakeRouter(config)
        self.client=TestClient(TestServer(create_app(config,self.router)))
        await self.client.start_server()
        await self.router.warmup
    async def asyncTearDown(self):
        self.release.set()
        await self.client.close()
        await self.backend_server.close()
    async def backend(self,request):
        if request.path=='/health':return web.json_response({'status':'ok'})
        if request.path=='/status/json':return web.json_response({'phase':'idle'})
        self.backend_calls+=1
        data=await request.json()
        mode=data.get('mode','ok')
        if mode=='hold':
            self.entered.set()
            await self.release.wait()
        if mode=='reject':return web.json_response({'error':{'code':'insufficient_memory'}},status=503)
        if mode=='crash':
            request.transport.abort()
            return web.Response()
        if mode in ('stall','stream'):
            out=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await out.prepare(request)
            self.entered.set()
            try:
                for i in range(100):
                    await out.write(b': heartbeat\n\n' if mode=='stall' else b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
                    await asyncio.sleep(0.02)
            except ConnectionResetError:pass
            return out
        return web.json_response({'choices':[{'message':{'content':'OK'}}]})
    async def post(self,mode='ok'):
        return await self.client.post('/v1/chat/completions',json={'model':'test','mode':mode})
    async def test_normal_and_resource_error_preserved(self):
        r=await self.post();self.assertEqual(r.status,200);await r.read()
        r=await self.post('reject');self.assertEqual(r.status,503)
        self.assertEqual((await r.json())['error']['code'],'insufficient_memory')
        self.assertEqual(r.headers['Retry-After'],'5')
        self.assertEqual(self.router.counts['stops'],0)
        r=await self.post();self.assertEqual(r.status,200);await r.read()
    async def test_queue_is_bounded_and_expires(self):
        first=asyncio.create_task(self.post('hold'))
        await self.entered.wait()
        second=asyncio.create_task(self.post())
        while self.router.pending<2:await asyncio.sleep(0.005)
        third=await self.post();self.assertEqual(third.status,429)
        self.assertEqual((await third.json())['error']['code'],'server_busy')
        expired=await second;self.assertEqual(expired.status,429)
        self.assertEqual((await expired.json())['error']['code'],'queue_timeout')
        self.release.set();r=await first;await r.read()
        self.assertEqual(self.backend_calls,1)
        self.assertEqual(self.router.pending,0)
    async def test_heartbeat_does_not_mask_stall(self):
        r=await self.post('stall');self.assertEqual(r.status,200)
        body=await r.text()
        self.assertIn('inference_timeout',body)
        self.assertIn('[DONE]',body)
        for _ in range(20):
            if self.router.counts['stops']:break
            await asyncio.sleep(0.01)
        self.assertEqual(self.router.counts['stops'],1)
        r=await self.post();self.assertEqual(r.status,200);await r.read()
    async def test_backend_crash_returns_502_and_next_request_works(self):
        r=await self.post('crash');self.assertEqual(r.status,502)
        self.assertEqual((await r.json())['error']['code'],'backend_failure')
        r=await self.post();self.assertEqual(r.status,200);await r.read()
    async def test_disconnect_releases_worker(self):
        r=await self.post('stream')
        await r.content.read(20)
        r.close()
        for _ in range(50):
            if self.router.pending==0:break
            await asyncio.sleep(0.01)
        self.assertEqual(self.router.pending,0)
        self.assertEqual(self.router.counts['cancel_settled'],1)
        r=await self.post();self.assertEqual(r.status,200);await r.read()
    async def test_cancelled_queued_request_never_runs(self):
        first=asyncio.create_task(self.post('hold'));await self.entered.wait()
        async with ClientSession() as client:
            queued=asyncio.create_task(client.post(self.client.make_url('/v1/chat/completions'),json={'model':'test'}))
            while self.router.pending<2:await asyncio.sleep(0.005)
            queued.cancel()
            with contextlib.suppress(asyncio.CancelledError):await queued
        self.release.set();r=await first;await r.read()
        await asyncio.sleep(0.1)
        self.assertEqual(self.backend_calls,1)
        self.assertEqual(self.router.pending,0)
    async def test_managed_backend_uses_operator_configuration(self):
        config=dict(self.router.config)
        config['backend']={'executable':'build/custom-server','library_path':'/example/rocm/lib','extra_args':['--threads','2']}
        with tempfile.TemporaryDirectory() as temp:
            managed=Router(config,root=pathlib.Path(temp))
            managed.client=self.router.client
            try:
                with patch('model_router.asyncio.create_subprocess_exec',new=AsyncMock(return_value=Process())) as spawn:
                    await managed.load('test')
                    args=spawn.call_args.args
                    self.assertEqual(args[0],str(pathlib.Path(temp)/'build/custom-server'))
                    self.assertEqual(args[args.index('--port')+1],str(self.backend_server.port))
                    self.assertEqual(args[-2:],('--threads','2'))
                    self.assertTrue(spawn.call_args.kwargs['env']['LD_LIBRARY_PATH'].startswith('/example/rocm/lib'))
                    self.assertFalse(managed.loading)
            finally:
                if managed.log:managed.log.close()

    async def test_health_and_model_metadata(self):
        r=await self.client.get('/livez');self.assertEqual(r.status,200);await r.read()
        r=await self.client.get('/readyz');self.assertEqual((await r.json())['status'],'ok')
        r=await self.client.get('/v1/models');self.assertEqual((await r.json())['data'][0]['context_length'],131072)

    async def test_external_backend_services_and_url(self):
        config={'default':'external','models':{'external':{
                    'path':'external','context':80000,
                    'upstream_url':str(self.backend_server.make_url('')).rstrip('/'),
                    'external_services':['target.service','proxy.service'],
                    'external_ready_urls':[str(self.backend_server.make_url('/health'))]}},
                'serving':{'load_timeout':2}}
        router=Router(config)
        router.client=self.router.client
        router.control_external_services=AsyncMock()
        await router.load('external')
        self.assertEqual(router.active,'external')
        self.assertEqual(router.active_url,config['models']['external']['upstream_url'])
        router.control_external_services.assert_awaited_once_with(
            'start',['target.service','proxy.service'])
        await router.stop()
        action,services=router.control_external_services.await_args_list[-1].args
        self.assertEqual(action,'stop')
        self.assertEqual(list(services),['proxy.service','target.service'])

if __name__=='__main__':unittest.main(verbosity=2)
