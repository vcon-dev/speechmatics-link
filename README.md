# Speechmatics vCon Link

A vCon server link for transcribing audio dialogs using the [Speechmatics](https://www.speechmatics.com/) API. This link processes vCon objects and stores transcription results in both Speechmatics native format and the standardized [World Transcription Format (WTF)](https://datatracker.ietf.org/doc/draft-howe-vcon-wtf-extension/).

## Features

- Batch transcription of audio dialogs via Speechmatics API
- Dual output formats:
  - Speechmatics native JSON format (preserves all provider-specific data)
  - WTF format (standardized transcription schema for interoperability)
- Speaker diarization support
- Automatic language detection
- Idempotent processing (skip already transcribed dialogs)
- Retry logic with exponential backoff
- Comprehensive logging

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/yourusername/speechmatics-link.git
```

Install from a specific version/tag:

```bash
pip install git+https://github.com/yourusername/speechmatics-link.git@v0.1.0
```

Install for development:

```bash
git clone https://github.com/yourusername/speechmatics-link.git
cd speechmatics-link
pip install -e ".[dev]"
```

## Configuration

### vCon Server Configuration

Add to your vcon-server `config.yml`:

```yaml
links:
  speechmatics:
    module: speechmatics_vcon_link
    pip_name: git+https://github.com/yourusername/speechmatics-link.git@main
    options:
      api_key: ${SPEECHMATICS_API_KEY}
      save_native_format: true
      save_wtf_format: true
      model: "enhanced"
      enable_diarization: false
      skip_if_exists: true

chains:
  transcription_chain:
    links:
      - speechmatics
    ingress_lists:
      - incoming_calls
    storages:
      - postgres
    enabled: 1
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_key` | string | `$SPEECHMATICS_API_KEY` | Speechmatics API key (required) |
| `api_url` | string | `https://asr.api.speechmatics.com/v2` | Speechmatics API base URL |
| `save_native_format` | bool | `true` | Store transcription in Speechmatics native format |
| `save_wtf_format` | bool | `true` | Store transcription in WTF format |
| `model` | string | `"enhanced"` | Transcription model: "standard" or "enhanced" |
| `language` | string | `null` | Language code (null for auto-detect) |
| `enable_diarization` | bool | `false` | Enable speaker diarization |
| `diarization_max_speakers` | int | `null` | Maximum speakers for diarization |
| `poll_interval` | int | `5` | Seconds between job status checks |
| `max_poll_attempts` | int | `120` | Maximum polling attempts before timeout |
| `skip_if_exists` | bool | `true` | Skip dialogs with existing transcription |
| `redis_host` | string | `"localhost"` | Redis host |
| `redis_port` | int | `6379` | Redis port |
| `redis_db` | int | `0` | Redis database number |

### Environment Variables

Set your Speechmatics API key:

```bash
export SPEECHMATICS_API_KEY=your-api-key-here
```

## Output Formats

### Speechmatics Native Format

Stored as vCon analysis with type `speechmatics_transcription`. Contains the complete Speechmatics API response including:

- Word-level timing and confidence
- Punctuation
- Speaker labels (if diarization enabled)
- Job metadata

### WTF Format

Stored as vCon analysis with type `wtf_transcription`. Follows the WTF schema defined in [draft-howe-vcon-wtf-extension-01](https://datatracker.ietf.org/doc/draft-howe-vcon-wtf-extension/):

```json
{
  "transcript": {
    "text": "Complete transcription text...",
    "language": "en-US",
    "duration": 65.2,
    "confidence": 0.95
  },
  "segments": [
    {
      "id": 0,
      "start": 0.5,
      "end": 4.8,
      "text": "Hello, this is Alice from customer service.",
      "confidence": 0.97,
      "speaker": "S1",
      "words": [0, 1, 2, 3, 4, 5, 6]
    }
  ],
  "words": [
    {
      "id": 0,
      "start": 0.5,
      "end": 0.8,
      "text": "Hello",
      "confidence": 0.98,
      "speaker": "S1",
      "is_punctuation": false
    }
  ],
  "speakers": {
    "S1": {
      "id": "S1",
      "label": "Speaker S1",
      "segments": [0, 1],
      "total_time": 4.3,
      "confidence": 0.97
    }
  },
  "metadata": {
    "created_at": "2025-01-02T12:15:30Z",
    "processed_at": "2025-01-02T12:16:35Z",
    "provider": "speechmatics",
    "model": "enhanced",
    "processing_time": 12.5,
    "audio": {
      "duration": 65.2,
      "format": "wav"
    }
  },
  "quality": {
    "audio_quality": "high",
    "average_confidence": 0.95,
    "low_confidence_words": 0,
    "multiple_speakers": true
  },
  "extensions": {
    "speechmatics": {
      "job": { ... },
      "format": "2.9"
    }
  }
}
```

## Usage Examples

### Basic Usage

The link automatically processes audio dialogs in vCons:

```python
# The run() function is called by vcon-server
from speechmatics_vcon_link import run

result = run(
    vcon_uuid="your-vcon-uuid",
    link_name="speechmatics",
    opts={
        "api_key": "your-api-key",
        "save_native_format": True,
        "save_wtf_format": True,
    }
)
```

### Using the Client Directly

```python
from speechmatics_vcon_link.client import SpeechmaticsClient, TranscriptionConfig

client = SpeechmaticsClient(api_key="your-api-key")

# Configure transcription
config = TranscriptionConfig(
    language="en",
    operating_point="enhanced",
    enable_diarization=True,
)

# Transcribe audio
result = client.transcribe(
    audio_url="https://example.com/audio.wav",
    config=config,
)

print(result)
```

### Converting to WTF Format

```python
from speechmatics_vcon_link.converter import convert_to_wtf

# Convert Speechmatics response to WTF format
wtf_result = convert_to_wtf(
    speechmatics_response,
    created_at="2025-01-02T12:00:00Z",
    processing_time=12.5,
)

print(wtf_result["transcript"]["text"])
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=speechmatics_vcon_link --cov-report=html

# Run specific test file
pytest tests/test_converter.py -v
```

### Code Formatting

```bash
black speechmatics_vcon_link/
```

### Project Structure

```
speechmatics-link/
  speechmatics_vcon_link/
    __init__.py       # Main link implementation
    client.py         # Speechmatics API client
    converter.py      # WTF format converter
  tests/
    test_link.py      # Link tests
    test_client.py    # Client tests
    test_converter.py # Converter tests
    fixtures/         # Test data
  pyproject.toml      # Package config
  README.md           # This file
```

## Troubleshooting

### API Key Issues

- Ensure `SPEECHMATICS_API_KEY` is set or `api_key` is provided in options
- Verify your API key has batch transcription permissions
- Check API key hasn't expired

### Audio URL Issues

- Audio must be accessible via HTTP/HTTPS URL
- Supported formats: WAV, MP3, FLAC, OGG, WebM
- URL must be publicly accessible or use signed URLs

### Timeout Errors

- Increase `max_poll_attempts` for longer audio files
- Check Speechmatics service status
- Verify audio file isn't corrupted

### Redis Connection Issues

- Verify Redis is running and accessible
- Check `redis_host`, `redis_port`, `redis_db` settings
- Ensure Redis JSON module is available

### Module Not Found

- Verify package is installed: `pip list | grep speechmatics`
- Check module name in config matches: `speechmatics_vcon_link`
- Restart vcon-server after installation

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [Speechmatics API Documentation](https://docs.speechmatics.com/)
- [vCon Specification](https://datatracker.ietf.org/wg/vcon/documents/)
- [WTF Extension Draft](https://datatracker.ietf.org/doc/draft-howe-vcon-wtf-extension/)
- [vCon Server Documentation](https://github.com/vcon-dev/vcon-server)

