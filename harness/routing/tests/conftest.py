import pytest

from mocks import MockBackends, serve


@pytest.fixture
def mock():
    m = MockBackends()
    with serve(m.app()) as base:
        m.base = base
        yield m
