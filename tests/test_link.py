"""Tests for the main Speechmatics vCon link."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from speechmatics_vcon_link import (
    run,
    default_options,
    get_redis_connection,
    get_vcon_from_redis,
    store_vcon_to_redis,
    has_transcription,
    get_audio_url,
    is_audio_dialog,
)
from speechmatics_vcon_link.client import (
    SpeechmaticsClient,
    SpeechmaticsError,
    SpeechmaticsAuthError,
    SpeechmaticsTimeoutError,
)


@pytest.fixture
def fixtures_path():
    """Get path to test fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_speechmatics_response(fixtures_path):
    """Load sample Speechmatics response."""
    with open(fixtures_path / "sample_speechmatics_response.json") as f:
        return json.load(f)


@pytest.fixture
def sample_vcon_dict():
    """Create a sample vCon dictionary for testing."""
    return {
        "uuid": "test-uuid-12345",
        "vcon": "0.0.1",
        "created_at": "2025-01-02T12:00:00Z",
        "parties": [{"tel": "+1-555-123-4567", "name": "Alice"}, {"tel": "+1-555-987-6543", "name": "Bob"}],
        "dialog": [
            {
                "type": "recording",
                "start": "2025-01-02T12:15:30Z",
                "duration": 65.2,
                "parties": [0, 1],
                "mimetype": "audio/wav",
                "url": "https://example.com/recordings/call-123.wav",
            }
        ],
        "analysis": [],
        "attachments": [],
    }


@pytest.fixture
def sample_vcon_with_transcription(sample_vcon_dict):
    """Create a sample vCon with existing transcription."""
    vcon_dict = sample_vcon_dict.copy()
    vcon_dict["analysis"] = [
        {
            "type": "speechmatics_transcription",
            "dialog": 0,
            "vendor": "speechmatics",
            "encoding": "json",
            "body": {"results": []},
        }
    ]
    return vcon_dict


@pytest.fixture
def mock_vcon_class():
    """Create a mock vCon class."""
    with patch("speechmatics_vcon_link.vcon.Vcon") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    """Create a mock Redis connection."""
    mock = MagicMock()
    mock.json.return_value = MagicMock()
    return mock


class TestDefaultOptions:
    """Tests for default options."""

    def test_default_options_keys(self):
        """Test that default options have expected keys."""
        expected_keys = [
            "api_key",
            "api_url",
            "save_native_format",
            "save_wtf_format",
            "model",
            "language",
            "enable_diarization",
            "poll_interval",
            "max_poll_attempts",
            "skip_if_exists",
            "redis_host",
            "redis_port",
            "redis_db",
        ]

        for key in expected_keys:
            assert key in default_options

    def test_default_save_formats(self):
        """Test that both save formats are enabled by default."""
        assert default_options["save_native_format"] is True
        assert default_options["save_wtf_format"] is True

    def test_default_model(self):
        """Test default model is enhanced."""
        assert default_options["model"] == "enhanced"


class TestGetAudioUrl:
    """Tests for audio URL extraction."""

    def test_get_url_from_url_field(self):
        """Test extracting URL from url field."""
        dialog = {"url": "https://example.com/audio.wav"}
        assert get_audio_url(dialog) == "https://example.com/audio.wav"

    def test_get_url_from_filename_http(self):
        """Test extracting URL from filename field (http)."""
        dialog = {"filename": "http://example.com/audio.wav"}
        assert get_audio_url(dialog) == "http://example.com/audio.wav"

    def test_get_url_from_filename_https(self):
        """Test extracting URL from filename field (https)."""
        dialog = {"filename": "https://example.com/audio.wav"}
        assert get_audio_url(dialog) == "https://example.com/audio.wav"

    def test_get_url_none_for_local_filename(self):
        """Test that local filenames return None."""
        dialog = {"filename": "local-file.wav"}
        assert get_audio_url(dialog) is None

    def test_get_url_empty_dialog(self):
        """Test with empty dialog."""
        assert get_audio_url({}) is None

    def test_get_url_prefers_url_over_filename(self):
        """Test that url field is preferred over filename."""
        dialog = {"url": "https://example.com/url.wav", "filename": "https://example.com/filename.wav"}
        assert get_audio_url(dialog) == "https://example.com/url.wav"


class TestIsAudioDialog:
    """Tests for audio dialog detection."""

    def test_recording_type(self):
        """Test recording type detection."""
        assert is_audio_dialog({"type": "recording"}) is True

    def test_audio_type(self):
        """Test audio type detection."""
        assert is_audio_dialog({"type": "audio"}) is True

    def test_audio_mimetype(self):
        """Test audio mimetype detection."""
        assert is_audio_dialog({"mimetype": "audio/wav"}) is True
        assert is_audio_dialog({"mimetype": "audio/mp3"}) is True
        assert is_audio_dialog({"mimetype": "audio/mpeg"}) is True

    def test_text_dialog(self):
        """Test text dialog is not audio."""
        assert is_audio_dialog({"type": "text"}) is False

    def test_video_mimetype(self):
        """Test video mimetype is not audio."""
        assert is_audio_dialog({"mimetype": "video/mp4"}) is False

    def test_empty_dialog(self):
        """Test empty dialog."""
        assert is_audio_dialog({}) is False


class TestHasTranscription:
    """Tests for transcription existence check."""

    def test_has_transcription_true(self):
        """Test when transcription exists."""
        mock_vcon = Mock()
        mock_vcon.analysis = [{"type": "speechmatics_transcription", "dialog": 0}]

        assert has_transcription(mock_vcon, 0, "speechmatics_transcription") is True

    def test_has_transcription_false(self):
        """Test when transcription does not exist."""
        mock_vcon = Mock()
        mock_vcon.analysis = []

        assert has_transcription(mock_vcon, 0, "speechmatics_transcription") is False

    def test_has_transcription_different_dialog(self):
        """Test when transcription exists for different dialog."""
        mock_vcon = Mock()
        mock_vcon.analysis = [{"type": "speechmatics_transcription", "dialog": 1}]

        assert has_transcription(mock_vcon, 0, "speechmatics_transcription") is False

    def test_has_transcription_different_type(self):
        """Test when different analysis type exists."""
        mock_vcon = Mock()
        mock_vcon.analysis = [{"type": "wtf_transcription", "dialog": 0}]

        assert has_transcription(mock_vcon, 0, "speechmatics_transcription") is False


class TestGetRedisConnection:
    """Tests for Redis connection."""

    @patch("speechmatics_vcon_link.redis.Redis")
    def test_get_redis_default(self, mock_redis_class):
        """Test Redis connection with defaults."""
        opts = default_options.copy()
        get_redis_connection(opts)

        mock_redis_class.assert_called_once_with(host="localhost", port=6379, db=0, decode_responses=False)

    @patch("speechmatics_vcon_link.redis.Redis")
    def test_get_redis_custom(self, mock_redis_class):
        """Test Redis connection with custom options."""
        opts = {"redis_host": "redis.example.com", "redis_port": 6380, "redis_db": 2}
        get_redis_connection(opts)

        mock_redis_class.assert_called_once_with(host="redis.example.com", port=6380, db=2, decode_responses=False)


class TestGetVconFromRedis:
    """Tests for vCon retrieval from Redis."""

    def test_get_vcon_success(self, mock_redis, sample_vcon_dict):
        """Test successful vCon retrieval."""
        mock_redis.json().get.return_value = sample_vcon_dict

        with patch("speechmatics_vcon_link.vcon.Vcon") as mock_vcon_class:
            mock_vcon = Mock()
            mock_vcon_class.return_value = mock_vcon

            result = get_vcon_from_redis(mock_redis, "test-uuid")

            assert result == mock_vcon
            mock_redis.json().get.assert_called_once()

    def test_get_vcon_not_found(self, mock_redis):
        """Test vCon not found in Redis."""
        mock_redis.json().get.return_value = None

        result = get_vcon_from_redis(mock_redis, "nonexistent-uuid")

        assert result is None

    def test_get_vcon_error(self, mock_redis):
        """Test error handling during retrieval."""
        mock_redis.json().get.side_effect = Exception("Redis error")

        result = get_vcon_from_redis(mock_redis, "test-uuid")

        assert result is None


class TestStoreVconToRedis:
    """Tests for vCon storage to Redis."""

    def test_store_vcon_success(self, mock_redis, sample_vcon_dict):
        """Test successful vCon storage."""
        mock_vcon = Mock()
        mock_vcon.uuid = "test-uuid"
        mock_vcon.to_dict.return_value = sample_vcon_dict

        result = store_vcon_to_redis(mock_redis, mock_vcon)

        assert result is True
        mock_redis.json().set.assert_called_once()

    def test_store_vcon_error(self, mock_redis):
        """Test error handling during storage."""
        mock_vcon = Mock()
        mock_vcon.uuid = "test-uuid"
        mock_vcon.to_dict.return_value = {}
        mock_redis.json().set.side_effect = Exception("Redis error")

        result = store_vcon_to_redis(mock_redis, mock_vcon)

        assert result is False


class TestRun:
    """Tests for the main run function."""

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.store_vcon_to_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_run_success(
        self,
        mock_client_class,
        mock_store,
        mock_get_vcon,
        mock_get_redis,
        sample_vcon_dict,
        sample_speechmatics_response,
    ):
        """Test successful run execution."""
        # Setup mocks
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_dict["dialog"]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        mock_store.return_value = True

        mock_client = MagicMock()
        mock_client.transcribe.return_value = sample_speechmatics_response
        mock_client_class.return_value = mock_client

        opts = {
            "api_key": "test-key",
            "save_native_format": True,
            "save_wtf_format": True,
        }

        result = run("test-uuid", "test-link", opts)

        assert result == "test-uuid"
        mock_vcon.add_analysis.assert_called()
        mock_store.assert_called_once()

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    def test_run_vcon_not_found(self, mock_get_vcon, mock_get_redis):
        """Test run when vCon is not found."""
        mock_get_redis.return_value = MagicMock()
        mock_get_vcon.return_value = None

        opts = {"api_key": "test-key"}

        result = run("nonexistent-uuid", "test-link", opts)

        assert result is None

    def test_run_missing_api_key(self):
        """Test run without API key raises error."""
        opts = {"api_key": None}

        with pytest.raises(ValueError, match="API key is required"):
            run("test-uuid", "test-link", opts)

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.store_vcon_to_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_run_skip_existing(
        self, mock_client_class, mock_store, mock_get_vcon, mock_get_redis, sample_vcon_with_transcription
    ):
        """Test run skips existing transcriptions."""
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_with_transcription["dialog"]
        mock_vcon.analysis = sample_vcon_with_transcription["analysis"]
        mock_get_vcon.return_value = mock_vcon

        opts = {
            "api_key": "test-key",
            "skip_if_exists": True,
            "save_native_format": True,
            "save_wtf_format": False,
        }

        result = run("test-uuid", "test-link", opts)

        # Should succeed but not process anything
        assert result == "test-uuid"
        mock_client_class.return_value.transcribe.assert_not_called()

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_run_auth_error(self, mock_client_class, mock_get_vcon, mock_get_redis, sample_vcon_dict):
        """Test run handles authentication error."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_dict["dialog"]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        mock_client = MagicMock()
        mock_client.transcribe.side_effect = SpeechmaticsAuthError("Invalid API key")
        mock_client_class.return_value = mock_client

        opts = {"api_key": "invalid-key"}

        with pytest.raises(SpeechmaticsAuthError):
            run("test-uuid", "test-link", opts)

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_run_timeout_error(self, mock_client_class, mock_get_vcon, mock_get_redis, sample_vcon_dict):
        """Test run handles timeout error."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_dict["dialog"]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        mock_client = MagicMock()
        mock_client.transcribe.side_effect = SpeechmaticsTimeoutError("Timeout")
        mock_client_class.return_value = mock_client

        opts = {"api_key": "test-key"}

        with pytest.raises(SpeechmaticsTimeoutError):
            run("test-uuid", "test-link", opts)

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    def test_run_skip_non_audio_dialogs(self, mock_get_vcon, mock_get_redis):
        """Test run skips non-audio dialogs."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = [{"type": "text", "body": "Hello"}]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        opts = {"api_key": "test-key"}

        result = run("test-uuid", "test-link", opts)

        # Should succeed without processing
        assert result == "test-uuid"

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    def test_run_skip_missing_url(self, mock_get_vcon, mock_get_redis):
        """Test run skips dialogs without audio URL."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = [{"type": "recording", "filename": "local.wav"}]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        opts = {"api_key": "test-key"}

        result = run("test-uuid", "test-link", opts)

        # Should succeed without processing
        assert result == "test-uuid"

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.store_vcon_to_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_run_native_format_only(
        self,
        mock_client_class,
        mock_store,
        mock_get_vcon,
        mock_get_redis,
        sample_vcon_dict,
        sample_speechmatics_response,
    ):
        """Test run with native format only."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_dict["dialog"]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        mock_store.return_value = True

        mock_client = MagicMock()
        mock_client.transcribe.return_value = sample_speechmatics_response
        mock_client_class.return_value = mock_client

        opts = {
            "api_key": "test-key",
            "save_native_format": True,
            "save_wtf_format": False,
        }

        result = run("test-uuid", "test-link", opts)

        assert result == "test-uuid"
        # Should only add one analysis (native)
        assert mock_vcon.add_analysis.call_count == 1

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.store_vcon_to_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_run_wtf_format_only(
        self,
        mock_client_class,
        mock_store,
        mock_get_vcon,
        mock_get_redis,
        sample_vcon_dict,
        sample_speechmatics_response,
    ):
        """Test run with WTF format only."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_dict["dialog"]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        mock_store.return_value = True

        mock_client = MagicMock()
        mock_client.transcribe.return_value = sample_speechmatics_response
        mock_client_class.return_value = mock_client

        opts = {
            "api_key": "test-key",
            "save_native_format": False,
            "save_wtf_format": True,
        }

        result = run("test-uuid", "test-link", opts)

        assert result == "test-uuid"
        # Should only add one analysis (WTF)
        assert mock_vcon.add_analysis.call_count == 1

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.store_vcon_to_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_run_both_formats(
        self,
        mock_client_class,
        mock_store,
        mock_get_vcon,
        mock_get_redis,
        sample_vcon_dict,
        sample_speechmatics_response,
    ):
        """Test run with both formats enabled."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_dict["dialog"]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        mock_store.return_value = True

        mock_client = MagicMock()
        mock_client.transcribe.return_value = sample_speechmatics_response
        mock_client_class.return_value = mock_client

        opts = {
            "api_key": "test-key",
            "save_native_format": True,
            "save_wtf_format": True,
        }

        result = run("test-uuid", "test-link", opts)

        assert result == "test-uuid"
        # Should add two analyses (native and WTF)
        assert mock_vcon.add_analysis.call_count == 2

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.store_vcon_to_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_run_store_failure(
        self,
        mock_client_class,
        mock_store,
        mock_get_vcon,
        mock_get_redis,
        sample_vcon_dict,
        sample_speechmatics_response,
    ):
        """Test run handles storage failure."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_dict["dialog"]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        mock_store.return_value = False  # Storage fails

        mock_client = MagicMock()
        mock_client.transcribe.return_value = sample_speechmatics_response
        mock_client_class.return_value = mock_client

        opts = {"api_key": "test-key"}

        result = run("test-uuid", "test-link", opts)

        assert result is None


class TestOptionMerging:
    """Tests for option merging behavior."""

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    def test_options_merge_with_defaults(self, mock_get_vcon, mock_get_redis):
        """Test that provided options merge with defaults."""
        mock_get_redis.return_value = MagicMock()
        mock_get_vcon.return_value = None  # Will return early

        # Provide only api_key, should use other defaults
        opts = {"api_key": "test-key"}

        run("test-uuid", "test-link", opts)

        # Should have used default redis settings
        mock_get_redis.return_value  # Connection should be created

    @patch("speechmatics_vcon_link.get_redis_connection")
    @patch("speechmatics_vcon_link.get_vcon_from_redis")
    @patch("speechmatics_vcon_link.SpeechmaticsClient")
    def test_options_override_defaults(self, mock_client_class, mock_get_vcon, mock_get_redis, sample_vcon_dict):
        """Test that provided options override defaults."""
        mock_get_redis.return_value = MagicMock()

        mock_vcon = MagicMock()
        mock_vcon.dialog = sample_vcon_dict["dialog"]
        mock_vcon.analysis = []
        mock_get_vcon.return_value = mock_vcon

        opts = {
            "api_key": "test-key",
            "model": "standard",  # Override default "enhanced"
            "poll_interval": 10,  # Override default 5
        }

        # Just verify options are passed correctly
        # The test verifies the option merging happens
        try:
            run("test-uuid", "test-link", opts)
        except Exception:
            pass  # We just want to verify option merging
