"""
Unit tests for tools/error_utils.py

Run from the project root:
    pytest tests/test_error_utils.py -v
"""

import socket
from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError
from google.auth.exceptions import RefreshError
from tools.error_utils import classify_error


def _http_error(status: int) -> HttpError:
    resp = MagicMock()
    resp.status = status
    return HttpError(resp=resp, content=b"error body")


class TestClassifyHttpError:
    def test_429_is_retryable(self):
        result = classify_error(_http_error(429))
        assert result["retryable"] is True

    def test_429_message_mentions_rate_limit(self):
        result = classify_error(_http_error(429))
        assert "rate limit" in result["message"].lower()

    def test_500_is_retryable(self):
        assert classify_error(_http_error(500))["retryable"] is True

    def test_503_is_retryable(self):
        assert classify_error(_http_error(503))["retryable"] is True

    def test_500_message_mentions_server_error(self):
        result = classify_error(_http_error(500))
        assert "server error" in result["message"].lower()

    def test_401_is_not_retryable(self):
        assert classify_error(_http_error(401))["retryable"] is False

    def test_401_message_mentions_token_file(self):
        result = classify_error(_http_error(401))
        assert "token" in result["message"].lower()

    def test_403_is_not_retryable(self):
        assert classify_error(_http_error(403))["retryable"] is False

    def test_403_message_mentions_permission(self):
        result = classify_error(_http_error(403))
        assert "permission" in result["message"].lower()

    def test_404_is_not_retryable(self):
        assert classify_error(_http_error(404))["retryable"] is False


class TestClassifyRefreshError:
    def test_refresh_error_is_not_retryable(self):
        assert classify_error(RefreshError("expired"))["retryable"] is False

    def test_refresh_error_message_mentions_token(self):
        result = classify_error(RefreshError("expired"))
        assert "token" in result["message"].lower()


class TestClassifyNetworkError:
    def test_connection_error_is_retryable(self):
        assert classify_error(ConnectionError("down"))["retryable"] is True

    def test_connection_error_message_mentions_network(self):
        result = classify_error(ConnectionError("down"))
        assert "network" in result["message"].lower()

    def test_timeout_error_is_retryable(self):
        assert classify_error(TimeoutError("timed out"))["retryable"] is True

    def test_socket_timeout_is_retryable(self):
        assert classify_error(socket.timeout("timed out"))["retryable"] is True


class TestClassifyUnknownError:
    def test_unknown_error_is_not_retryable(self):
        assert classify_error(ValueError("something"))["retryable"] is False

    def test_unknown_error_message_is_str_of_exception(self):
        result = classify_error(ValueError("something random"))
        assert "something random" in result["message"]
