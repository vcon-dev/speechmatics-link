"""Integration tests for Speechmatics vCon link.

These tests make real API calls to Speechmatics and require:
1. A valid SPEECHMATICS_API_KEY environment variable
2. Network access
3. The --run-integration flag to pytest

Run with:
    pytest tests/test_integration.py --run-integration -v

Note: These tests will consume API credits.
"""

import os
import json
import pytest
from pathlib import Path

from speechmatics_vcon_link.client import (
    SpeechmaticsClient,
    TranscriptionConfig,
    SpeechmaticsError,
)
from speechmatics_vcon_link.converter import convert_to_wtf


# Public sample audio files for testing
# Using GitHub-hosted audio files that Speechmatics can access
SAMPLE_AUDIO_URLS = {
    "english_short": "https://github.com/Picovoice/porcupine/raw/master/resources/audio_samples/multiple_keywords.wav",
    "english_medium": "https://github.com/mozilla/DeepSpeech/raw/master/data/smoke_test/LDC93S1.wav",
}


@pytest.fixture(scope="module")
def api_key():
    """Get API key from environment."""
    key = os.environ.get("SPEECHMATICS_API_KEY")
    if not key:
        pytest.skip("SPEECHMATICS_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="module")
def client(api_key):
    """Create a Speechmatics client."""
    return SpeechmaticsClient(api_key=api_key)


@pytest.fixture(scope="module")
def sample_audio_url():
    """Get a sample audio URL for testing."""
    return SAMPLE_AUDIO_URLS["english_short"]


@pytest.mark.integration
class TestSpeechmaticsClientIntegration:
    """Integration tests for the Speechmatics client."""

    def test_submit_and_get_transcript(self, client, sample_audio_url):
        """Test submitting a job and retrieving the transcript."""
        # Submit job
        job_id = client.submit_job(sample_audio_url)
        assert job_id is not None
        assert isinstance(job_id, str)
        print(f"\nSubmitted job: {job_id}")

        # Wait for completion and get transcript
        transcript = client.wait_for_completion(job_id, poll_interval=3, max_attempts=60)

        assert transcript is not None
        assert "results" in transcript
        assert len(transcript["results"]) > 0
        print(f"Got transcript with {len(transcript['results'])} results")

        # Verify we got actual words
        words = [r for r in transcript["results"] if r.get("type") == "word"]
        assert len(words) > 0
        print(f"Transcript contains {len(words)} words")

    def test_transcribe_convenience_method(self, client, sample_audio_url):
        """Test the convenience transcribe method."""
        transcript = client.transcribe(audio_url=sample_audio_url, poll_interval=3, max_attempts=60)

        assert transcript is not None
        assert "results" in transcript

        # Extract text
        words = [r["alternatives"][0]["content"] for r in transcript["results"] if r.get("type") == "word"]
        text = " ".join(words)
        print(f"\nTranscribed text: {text[:200]}...")

        assert len(text) > 0

    def test_transcribe_with_language(self, client, sample_audio_url):
        """Test transcription with explicit language setting."""
        config = TranscriptionConfig(language="en", operating_point="enhanced")

        transcript = client.transcribe(audio_url=sample_audio_url, config=config, poll_interval=3, max_attempts=60)

        assert transcript is not None
        assert "results" in transcript

    def test_transcribe_with_diarization(self, client, sample_audio_url):
        """Test transcription with speaker diarization."""
        config = TranscriptionConfig(language="en", enable_diarization=True)

        transcript = client.transcribe(audio_url=sample_audio_url, config=config, poll_interval=3, max_attempts=60)

        assert transcript is not None
        assert "results" in transcript

        # Check if speaker labels are present
        words_with_speaker = [
            r
            for r in transcript["results"]
            if r.get("type") == "word" and r.get("alternatives", [{}])[0].get("speaker") is not None
        ]
        print(f"\nWords with speaker labels: {len(words_with_speaker)}")


@pytest.mark.integration
class TestWtfConverterIntegration:
    """Integration tests for WTF format conversion with real transcripts."""

    def test_convert_real_transcript(self, client, sample_audio_url):
        """Test converting a real Speechmatics transcript to WTF format."""
        # Get real transcript
        transcript = client.transcribe(audio_url=sample_audio_url, poll_interval=3, max_attempts=60)

        # Convert to WTF
        wtf_result = convert_to_wtf(transcript, created_at="2025-01-02T12:00:00Z", processing_time=10.0)

        # Verify required fields
        assert "transcript" in wtf_result
        assert "segments" in wtf_result
        assert "metadata" in wtf_result

        # Verify transcript object
        assert wtf_result["transcript"]["text"]
        assert wtf_result["transcript"]["language"]
        assert wtf_result["transcript"]["duration"] > 0
        assert 0 <= wtf_result["transcript"]["confidence"] <= 1

        print(f"\nWTF transcript text: {wtf_result['transcript']['text'][:200]}...")
        print(f"Duration: {wtf_result['transcript']['duration']}s")
        print(f"Confidence: {wtf_result['transcript']['confidence']}")
        print(f"Segments: {len(wtf_result['segments'])}")
        print(f"Words: {len(wtf_result.get('words', []))}")

    def test_convert_with_diarization(self, client, sample_audio_url):
        """Test WTF conversion with diarized transcript."""
        config = TranscriptionConfig(language="en", enable_diarization=True)

        transcript = client.transcribe(audio_url=sample_audio_url, config=config, poll_interval=3, max_attempts=60)

        wtf_result = convert_to_wtf(transcript)

        # Check speakers object
        speakers = wtf_result.get("speakers", {})
        print(f"\nSpeakers found: {list(speakers.keys())}")

        for speaker_id, speaker_info in speakers.items():
            print(f"  {speaker_id}: {speaker_info.get('total_time', 0):.1f}s")

    def test_wtf_quality_metrics(self, client, sample_audio_url):
        """Test that quality metrics are calculated correctly."""
        transcript = client.transcribe(audio_url=sample_audio_url, poll_interval=3, max_attempts=60)

        wtf_result = convert_to_wtf(transcript)

        quality = wtf_result.get("quality", {})

        print(f"\nQuality metrics:")
        print(f"  Audio quality: {quality.get('audio_quality')}")
        print(f"  Average confidence: {quality.get('average_confidence')}")
        print(f"  Low confidence words: {quality.get('low_confidence_words')}")
        print(f"  Multiple speakers: {quality.get('multiple_speakers')}")

        assert quality.get("audio_quality") in ["high", "medium", "low"]
        assert 0 <= quality.get("average_confidence", 0) <= 1


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, client, sample_audio_url):
        """Test the complete transcription pipeline."""
        import time

        start_time = time.time()

        # Step 1: Transcribe
        print("\n1. Submitting transcription job...")
        config = TranscriptionConfig(language="en", operating_point="enhanced")

        transcript = client.transcribe(audio_url=sample_audio_url, config=config, poll_interval=3, max_attempts=60)

        transcription_time = time.time() - start_time
        print(f"   Transcription completed in {transcription_time:.1f}s")

        # Step 2: Convert to WTF
        print("2. Converting to WTF format...")
        wtf_result = convert_to_wtf(transcript, processing_time=transcription_time)

        # Step 3: Verify output
        print("3. Verifying output...")

        # Required sections
        assert "transcript" in wtf_result
        assert "segments" in wtf_result
        assert "metadata" in wtf_result

        # Transcript
        assert len(wtf_result["transcript"]["text"]) > 0

        # Segments
        assert len(wtf_result["segments"]) > 0
        for segment in wtf_result["segments"]:
            assert "id" in segment
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
            assert segment["end"] >= segment["start"]

        # Metadata
        assert wtf_result["metadata"]["provider"] == "speechmatics"
        assert wtf_result["metadata"]["processing_time"] == transcription_time

        # Extensions
        assert "extensions" in wtf_result
        assert "speechmatics" in wtf_result["extensions"]

        print("\n=== Full Pipeline Results ===")
        print(f"Text: {wtf_result['transcript']['text'][:300]}...")
        print(f"Language: {wtf_result['transcript']['language']}")
        print(f"Duration: {wtf_result['transcript']['duration']:.1f}s")
        print(f"Confidence: {wtf_result['transcript']['confidence']:.2f}")
        print(f"Segments: {len(wtf_result['segments'])}")
        print(f"Words: {len(wtf_result.get('words', []))}")
        print(f"Processing time: {transcription_time:.1f}s")
        print("✓ All verifications passed!")

    def test_save_sample_output(self, client, sample_audio_url, tmp_path):
        """Test saving real output to file for inspection."""
        transcript = client.transcribe(audio_url=sample_audio_url, poll_interval=3, max_attempts=60)

        wtf_result = convert_to_wtf(transcript)

        # Save native format
        native_path = tmp_path / "native_transcript.json"
        with open(native_path, "w") as f:
            json.dump(transcript, f, indent=2)
        print(f"\nNative transcript saved to: {native_path}")

        # Save WTF format
        wtf_path = tmp_path / "wtf_transcript.json"
        with open(wtf_path, "w") as f:
            json.dump(wtf_result, f, indent=2)
        print(f"WTF transcript saved to: {wtf_path}")

        # Also save to fixtures for reference
        fixtures_path = Path(__file__).parent / "fixtures"

        real_native_path = fixtures_path / "real_speechmatics_response.json"
        with open(real_native_path, "w") as f:
            json.dump(transcript, f, indent=2)
        print(f"Saved real response to: {real_native_path}")

        real_wtf_path = fixtures_path / "real_wtf_output.json"
        with open(real_wtf_path, "w") as f:
            json.dump(wtf_result, f, indent=2)
        print(f"Saved real WTF output to: {real_wtf_path}")
