# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the Podman client adapter.

These tests exercise adapter logic without a running Podman daemon by injecting a
mocked client. The adapter is built with ``object.__new__`` so ``__init__`` (which
imports the optional ``podman`` package) is skipped.
"""

from unittest.mock import MagicMock

from kubeflow.trainer.backends.container.adapters.podman import PodmanClientAdapter


def _make_adapter_with_mock_client() -> tuple[PodmanClientAdapter, MagicMock]:
    """Build a PodmanClientAdapter with a mocked client, bypassing __init__."""
    adapter = object.__new__(PodmanClientAdapter)
    adapter.client = MagicMock()
    adapter.client.containers.list.return_value = []
    adapter._runtime_type = "podman"
    return adapter, adapter.client


def test_list_containers_default_filters_none():
    """list_containers() with the default filters=None must not raise."""
    adapter, client = _make_adapter_with_mock_client()

    result = adapter.list_containers()

    assert result == []
    client.containers.list.assert_called_once_with(all=True, filters=None)


def test_list_containers_explicit_none():
    """Passing filters=None explicitly must be equivalent to the default."""
    adapter, client = _make_adapter_with_mock_client()

    result = adapter.list_containers(filters=None)

    assert result == []
    client.containers.list.assert_called_once_with(all=True, filters=None)


def test_list_containers_does_not_mutate_caller_filters():
    """The podman-py single-value workaround must not mutate the caller's dict."""
    adapter, client = _make_adapter_with_mock_client()
    filters = {"label": ["key=value"]}

    adapter.list_containers(filters=filters)

    # The caller's dict is unchanged; the flattening happens on a local copy.
    assert filters == {"label": ["key=value"]}
    client.containers.list.assert_called_once_with(all=True, filters={"label": "key=value"})
