"""Bounded local HTTP regressions; no model or GPU required."""
import asyncio, importlib.util, json, pathlib, unittest
from aiohttp import web, ClientSession
from aiohttp.test_utils import TestServer
spec=importlib.util.spec_from_file_location('router',pathlib.Path(__file__).parents[2]/'model_router.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.IsolatedAsyncioTestCase):
 def test_monotonic_request_progress(self):
  p=m.TokenProgress()
  def line(i,n):return b':lucebox-progress '+json.dumps({'id':i,'tokens':n}).encode()
  self.assertTrue(p.observe(line('a',64)))
  for x in [line('a',64),line('a',63),line('b',128),line('a',True),b':heartbeat',b':lucebox-progress null']:
   self.assertFalse(p.observe(x))
  self.assertTrue(p.observe(line('a',128)))
 async def run_stream(self,kind):
  async def backend(request):
   response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
   try:
    await response.write(b'data: {"choices":[{"delta":{"content":"start"},"finish_reason":null}]}\n\n')
    for i in range(1,9):
     await asyncio.sleep(.02)
     frame=(b':lucebox-progress '+json.dumps({'id':'r','tokens':i*64}).encode()) if kind=='progress' else b':heartbeat'
     if kind in ('progress','heartbeat'):await response.write(frame+b'\n\n')
     else:break
    if kind!='truncated':await response.write(b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n')
    await response.write(b'data: [DONE]\n\n')
   except (ConnectionError,RuntimeError):pass
   return response
  app=web.Application();app.router.add_post('/v1/chat/completions',backend)
  upstream=TestServer(app);await upstream.start_server()
  router=m.Router({'default':'m','models':{'m':{'path':'weights'}},'backend_url':str(upstream.make_url('')).rstrip('/')})
  router.limits['token_timeout']=.08
  errors=[]
  async def forward(request):
   state=m.ForwardState()
   try:return await router.forward(request,b'{}',state)
   except (asyncio.TimeoutError,m.BackendFault) as e:
    errors.append(e);return await router.fail_forward(state,502,'backend_failure',str(e) or 'timeout')
  proxy_app=web.Application();proxy_app.router.add_post('/v1/chat/completions',forward)
  proxy=TestServer(proxy_app)
  try:
   async with ClientSession() as client:
    router.client=client;await proxy.start_server()
    async with client.post(proxy.make_url('/v1/chat/completions')) as response:text=await response.text()
  finally:await proxy.close();await upstream.close()
  return errors,text
 async def test_buffered_tool_progress_survives_timeout(self):
  errors,text=await self.run_stream('progress');self.assertEqual(errors,[]);self.assertIn('tool_calls',text)
 async def test_generic_heartbeats_do_not_hide_stall(self):
  errors,text=await self.run_stream('heartbeat');self.assertIsInstance(errors[0],asyncio.TimeoutError)
 async def test_truncated_stream_is_reported(self):
  errors,text=await self.run_stream('truncated');self.assertIsInstance(errors[0],m.BackendFault);self.assertIn('before finish_reason',text)
if __name__=='__main__':unittest.main()
