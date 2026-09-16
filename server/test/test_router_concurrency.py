import asyncio, unittest
from unittest.mock import AsyncMock
from model_router import Router

class ConcurrentRouterTests(unittest.IsolatedAsyncioTestCase):
    def make_router(self):
        r=Router({'models':{'a':{'context':131072,'path':'weights'},'b':{'context':131072,'path':'weights'}},'default':'a','serving':{'max_concurrency':3}})
        async def load(name):
            self.assertEqual(r.inflight,0)
            r.active=name
        r.load=AsyncMock(side_effect=load);r.stop=AsyncMock()
        return r
    async def test_three_slots_then_queue(self):
        r=self.make_router()
        await asyncio.gather(*(r.acquire_slot('a') for _ in range(3)))
        self.assertEqual(r.inflight,3);self.assertEqual(r.load.await_count,1)
        t=asyncio.create_task(r.acquire_slot('a'));await asyncio.sleep(.01)
        self.assertFalse(t.done())
        await r.release_slot();await asyncio.wait_for(t,1)
        self.assertEqual(r.inflight,3)
        for _ in range(3):await r.release_slot()
    async def test_switch_waits_for_all_peers_and_preserves_fifo(self):
        r=self.make_router();await r.acquire_slot('a');await r.acquire_slot('a')
        b=asyncio.create_task(r.acquire_slot('b'));await asyncio.sleep(.01)
        a=asyncio.create_task(r.acquire_slot('a'));await asyncio.sleep(.01)
        await r.release_slot();self.assertFalse(b.done())
        await r.release_slot();await asyncio.wait_for(b,1)
        self.assertEqual(r.active,'b');self.assertFalse(a.done())
        await r.release_slot();await asyncio.wait_for(a,1);await r.release_slot()
    async def test_cancel_waiter_leaves_no_deadlock(self):
        r=self.make_router();await r.acquire_slot('a')
        b=asyncio.create_task(r.acquire_slot('b'));await asyncio.sleep(.01);b.cancel()
        with self.assertRaises(asyncio.CancelledError):await b
        await asyncio.wait_for(r.acquire_slot('a'),1)
        self.assertEqual(len(r.waiters),0)
        await r.release_slot();await r.release_slot()
    async def test_failed_stream_does_not_kill_peer(self):
        r=self.make_router();await r.acquire_slot('a');await r.acquire_slot('a')
        await r.recover_request();r.stop.assert_not_awaited()
        await r.release_slot();r.stop.assert_not_awaited()
        await r.release_slot();r.stop.assert_awaited_once()
        self.assertFalse(r.recycle_pending)
    async def test_disconnect_keeps_peer_alive(self):
        r=self.make_router();await r.acquire_slot('a');await r.acquire_slot('a')
        await r.recover_request(cancelled=True);r.stop.assert_not_awaited()
        await r.release_slot();self.assertEqual(r.inflight,1);await r.release_slot()
