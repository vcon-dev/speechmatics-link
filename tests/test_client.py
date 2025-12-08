"""Tests for Speechmatics API client."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from speechmatics_vcon_link.client import (
    SpeechmaticsClient,
    TranscriptionConfig,
    JobStatus,
    SpeechmaticsError,
    SpeechmaticsAuthError,
    SpeechmaticsJobError,
    SpeechmaticsTimeoutError,
)


@pytest.fixture
def api_key():
    """Test API key."""
    return "test-api-key-12345"


@pytest.fixture
def client(api_key):
    """Create a test client."""
    return SpeechmaticsClient(api_key=api_key)


@pytest.fixture
def mock_response():
    """Create a mock response factory."""
    def _mock_response(status_code=200, json_data=None, raise_for_status=None):
        mock = Mock()
        mock.status_code = status_code
        mock.json.return_value = json_data or {}
        if raise_for_status:
            mock.raise_for_status.side_effect = raise_for_status
        else:
            mock.raise_for_status.return_value = None
        return mock
    return _mock_response


class TestSpeechmaticsClientInit:
    """Tests for client initialization."""
    
    def test_init_with_api_key(self, api_key):
        """Test client initialization with valid API key."""
        client = SpeechmaticsClient(api_key=api_key)
        assert client.api_key == api_key
        assert client.api_url == SpeechmaticsClient.DEFAULT_API_URL
        assert client.timeout == 300
    
    def test_init_with_custom_url(self, api_key):
        """Test client initialization with custom URL."""
        custom_url = "https://custom.api.com/v2"
        client = SpeechmaticsClient(api_key=api_key, api_url=custom_url)
        assert client.api_url == custom_url
    
    def test_init_strips_trailing_slash(self, api_key):
        """Test that trailing slash is stripped from URL."""
        client = SpeechmaticsClient(api_key=api_key, api_url="https://api.com/v2/")
        assert client.api_url == "https://api.com/v2"
    
    def test_init_without_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="API key is required"):
            SpeechmaticsClient(api_key=None)
        
        with pytest.raises(ValueError, match="API key is required"):
            SpeechmaticsClient(api_key="")
    
    def test_init_with_custom_timeout(self, api_key):
        """Test client initialization with custom timeout."""
        client = SpeechmaticsClient(api_key=api_key, timeout=60)
        assert client.timeout == 60


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TranscriptionConfig()
        assert config.language is None
        assert config.operating_point == "enhanced"
        assert config.enable_diarization is False
        assert config.diarization_max_speakers is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TranscriptionConfig(
            language="en",
            operating_point="standard",
            enable_diarization=True,
            diarization_max_speakers=5
        )
        assert config.language == "en"
        assert config.operating_point == "standard"
        assert config.enable_diarization is True
        assert config.diarization_max_speakers == 5


class TestBuildConfig:
    """Tests for config building."""
    
    def test_build_config_default(self, client):
        """Test building config with defaults."""
        config = TranscriptionConfig()
        result = client._build_config(config)
        
        assert result["type"] == "transcription"
        assert result["transcription_config"]["operating_point"] == "enhanced"
        assert result["transcription_config"]["language"] == "auto"
    
    def test_build_config_with_language(self, client):
        """Test building config with specific language."""
        config = TranscriptionConfig(language="es")
        result = client._build_config(config)
        
        assert result["transcription_config"]["language"] == "es"
    
    def test_build_config_with_diarization(self, client):
        """Test building config with diarization enabled."""
        config = TranscriptionConfig(enable_diarization=True, diarization_max_speakers=3)
        result = client._build_config(config)
        
        assert result["transcription_config"]["diarization"] == "speaker"
        assert result["transcription_config"]["speaker_diarization_config"]["max_speakers"] == 3


class TestSubmitJob:
    """Tests for job submission."""
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_submit_job_success(self, mock_session_class, api_key, mock_response):
        """Test successful job submission."""
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response(
            status_code=200,
            json_data={"id": "job-123"}
        )
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        job_id = client.submit_job("https://example.com/audio.wav")
        
        assert job_id == "job-123"
        mock_session.post.assert_called_once()
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_submit_job_with_config(self, mock_session_class, api_key, mock_response):
        """Test job submission with custom config."""
        import json
        
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response(
            status_code=200,
            json_data={"id": "job-456"}
        )
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        config = TranscriptionConfig(language="fr", enable_diarization=True)
        job_id = client.submit_job("https://example.com/audio.wav", config)
        
        assert job_id == "job-456"
        call_args = mock_session.post.call_args
        # Config is sent as multipart file, extract and parse it
        files = call_args.kwargs.get("files", {})
        config_tuple = files.get("config")
        assert config_tuple is not None
        config_json = json.loads(config_tuple[1])
        assert config_json["transcription_config"]["language"] == "fr"
        assert config_json["transcription_config"]["diarization"] == "speaker"
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_submit_job_auth_error(self, mock_session_class, api_key, mock_response):
        """Test job submission with auth error."""
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response(status_code=401)
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        with pytest.raises(SpeechmaticsAuthError, match="Invalid API key"):
            client.submit_job("https://example.com/audio.wav")
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_submit_job_no_job_id_error(self, mock_session_class, api_key, mock_response):
        """Test job submission with no job ID returned."""
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response(
            status_code=200,
            json_data={}
        )
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        with pytest.raises(SpeechmaticsError, match="No job ID returned"):
            client.submit_job("https://example.com/audio.wav")


class TestGetJobStatus:
    """Tests for job status retrieval."""
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_get_job_status_success(self, mock_session_class, api_key, mock_response):
        """Test successful status retrieval."""
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response(
            status_code=200,
            json_data={"job": {"status": "running"}}
        )
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        status = client.get_job_status("job-123")
        
        assert status["job"]["status"] == "running"
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_get_job_status_not_found(self, mock_session_class, api_key, mock_response):
        """Test status retrieval for non-existent job."""
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response(status_code=404)
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        with pytest.raises(SpeechmaticsJobError, match="Job not found"):
            client.get_job_status("job-999")


class TestGetTranscript:
    """Tests for transcript retrieval."""
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_get_transcript_success(self, mock_session_class, api_key, mock_response):
        """Test successful transcript retrieval."""
        mock_session = MagicMock()
        transcript_data = {"results": [{"type": "word", "alternatives": [{"content": "Hello"}]}]}
        mock_session.get.return_value = mock_response(
            status_code=200,
            json_data=transcript_data
        )
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        result = client.get_transcript("job-123")
        
        assert result == transcript_data
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_get_transcript_not_found(self, mock_session_class, api_key, mock_response):
        """Test transcript retrieval for non-existent job."""
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response(status_code=404)
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        with pytest.raises(SpeechmaticsJobError, match="not found"):
            client.get_transcript("job-999")


class TestWaitForCompletion:
    """Tests for waiting for job completion."""
    
    @patch.object(SpeechmaticsClient, 'get_job_status')
    @patch.object(SpeechmaticsClient, 'get_transcript')
    def test_wait_for_completion_immediate(self, mock_get_transcript, mock_get_status, client):
        """Test immediate job completion."""
        mock_get_status.return_value = {"job": {"status": "done"}}
        mock_get_transcript.return_value = {"results": []}
        
        result = client.wait_for_completion("job-123", poll_interval=0.01)
        
        assert result == {"results": []}
        mock_get_status.assert_called_once_with("job-123")
    
    @patch.object(SpeechmaticsClient, 'get_job_status')
    @patch.object(SpeechmaticsClient, 'get_transcript')
    def test_wait_for_completion_after_polling(self, mock_get_transcript, mock_get_status, client):
        """Test job completion after polling."""
        mock_get_status.side_effect = [
            {"job": {"status": "running"}},
            {"job": {"status": "running"}},
            {"job": {"status": "done"}},
        ]
        mock_get_transcript.return_value = {"results": []}
        
        result = client.wait_for_completion("job-123", poll_interval=0.01)
        
        assert result == {"results": []}
        assert mock_get_status.call_count == 3
    
    @patch.object(SpeechmaticsClient, 'get_job_status')
    def test_wait_for_completion_rejected(self, mock_get_status, client):
        """Test handling of rejected job."""
        mock_get_status.return_value = {"job": {"status": "rejected", "errors": ["Bad audio"]}}
        
        with pytest.raises(SpeechmaticsJobError, match="Job was rejected"):
            client.wait_for_completion("job-123", poll_interval=0.01)
    
    @patch.object(SpeechmaticsClient, 'get_job_status')
    def test_wait_for_completion_timeout(self, mock_get_status, client):
        """Test timeout waiting for completion."""
        mock_get_status.return_value = {"job": {"status": "running"}}
        
        with pytest.raises(SpeechmaticsTimeoutError, match="did not complete"):
            client.wait_for_completion("job-123", poll_interval=0.01, max_attempts=2)


class TestTranscribe:
    """Tests for the convenience transcribe method."""
    
    @patch.object(SpeechmaticsClient, 'submit_job')
    @patch.object(SpeechmaticsClient, 'wait_for_completion')
    def test_transcribe_success(self, mock_wait, mock_submit, client):
        """Test successful transcription."""
        mock_submit.return_value = "job-123"
        mock_wait.return_value = {"results": []}
        
        result = client.transcribe("https://example.com/audio.wav")
        
        assert result == {"results": []}
        mock_submit.assert_called_once()
        mock_wait.assert_called_once_with("job-123", 5, 120)


class TestDeleteJob:
    """Tests for job deletion."""
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_delete_job_success(self, mock_session_class, api_key, mock_response):
        """Test successful job deletion."""
        mock_session = MagicMock()
        mock_session.delete.return_value = mock_response(status_code=200)
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        result = client.delete_job("job-123")
        
        assert result is True
    
    @patch('speechmatics_vcon_link.client.requests.Session')
    def test_delete_job_failure(self, mock_session_class, api_key, mock_response):
        """Test failed job deletion."""
        mock_session = MagicMock()
        mock_session.delete.return_value = mock_response(
            status_code=404,
            raise_for_status=requests.HTTPError("Not found")
        )
        mock_session_class.return_value = mock_session
        
        client = SpeechmaticsClient(api_key=api_key)
        result = client.delete_job("job-999")
        
        assert result is False


class TestJobStatus:
    """Tests for JobStatus enum."""
    
    def test_job_status_values(self):
        """Test JobStatus enum values."""
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.DONE.value == "done"
        assert JobStatus.REJECTED.value == "rejected"
        assert JobStatus.DELETED.value == "deleted"

