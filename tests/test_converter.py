"""Tests for WTF format converter."""

import json
import os
import pytest
from pathlib import Path

from speechmatics_vcon_link.converter import (
    normalize_confidence,
    extract_language_code,
    calculate_duration,
    calculate_average_confidence,
    count_low_confidence_words,
    build_transcript_object,
    build_segments,
    build_words,
    build_speakers,
    build_metadata,
    build_quality,
    convert_to_wtf,
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
def minimal_response():
    """Create a minimal Speechmatics response."""
    return {
        "format": "2.9",
        "job": {"id": "test-job", "transcription_config": {"language": "en", "operating_point": "enhanced"}},
        "results": [
            {
                "type": "word",
                "start_time": 0.0,
                "end_time": 0.5,
                "alternatives": [{"content": "Hello", "confidence": 0.95}],
            }
        ],
    }


@pytest.fixture
def empty_response():
    """Create an empty Speechmatics response."""
    return {"format": "2.9", "job": {}, "results": []}


class TestNormalizeConfidence:
    """Tests for confidence score normalization."""

    def test_normalize_in_range(self):
        """Test normalization of values in range."""
        assert normalize_confidence(0.5) == 0.5
        assert normalize_confidence(0.0) == 0.0
        assert normalize_confidence(1.0) == 1.0

    def test_normalize_above_range(self):
        """Test normalization of values above range."""
        assert normalize_confidence(1.5) == 1.0
        assert normalize_confidence(2.0) == 1.0

    def test_normalize_below_range(self):
        """Test normalization of values below range."""
        assert normalize_confidence(-0.5) == 0.0
        assert normalize_confidence(-1.0) == 0.0

    def test_normalize_none(self):
        """Test normalization of None value."""
        assert normalize_confidence(None) == 0.0


class TestExtractLanguageCode:
    """Tests for language code extraction."""

    def test_extract_english(self, sample_speechmatics_response):
        """Test extraction of English language code."""
        result = extract_language_code(sample_speechmatics_response)
        assert result == "en-US"

    def test_extract_with_mapping(self):
        """Test extraction with language mapping."""
        response = {"job": {"transcription_config": {"language": "es"}}}
        assert extract_language_code(response) == "es-ES"

    def test_extract_auto(self):
        """Test extraction with auto language."""
        response = {"job": {"transcription_config": {"language": "auto"}}}
        assert extract_language_code(response) == "und"

    def test_extract_unmapped(self):
        """Test extraction with unmapped language."""
        response = {"job": {"transcription_config": {"language": "sv"}}}
        assert extract_language_code(response) == "sv"

    def test_extract_missing_config(self, empty_response):
        """Test extraction with missing config."""
        result = extract_language_code(empty_response)
        assert result == "und"


class TestCalculateDuration:
    """Tests for duration calculation."""

    def test_calculate_from_results(self, sample_speechmatics_response):
        """Test duration calculation from results."""
        duration = calculate_duration(sample_speechmatics_response)
        assert duration == 7.8  # Last end_time in sample

    def test_calculate_empty_results(self, empty_response):
        """Test duration calculation with empty results."""
        duration = calculate_duration(empty_response)
        assert duration == 0.0


class TestCalculateAverageConfidence:
    """Tests for average confidence calculation."""

    def test_calculate_average(self):
        """Test average confidence calculation."""
        words = [
            {"confidence": 0.8},
            {"confidence": 0.9},
            {"confidence": 1.0},
        ]
        assert calculate_average_confidence(words) == 0.9

    def test_calculate_empty(self):
        """Test average with empty list."""
        assert calculate_average_confidence([]) == 0.0

    def test_calculate_missing_confidence(self):
        """Test average with missing confidence values."""
        words = [
            {"confidence": 0.8},
            {"text": "hello"},  # No confidence
            {"confidence": 1.0},
        ]
        assert calculate_average_confidence(words) == 0.9


class TestCountLowConfidenceWords:
    """Tests for low confidence word counting."""

    def test_count_low_confidence(self):
        """Test counting low confidence words."""
        words = [
            {"confidence": 0.3},
            {"confidence": 0.6},
            {"confidence": 0.9},
        ]
        assert count_low_confidence_words(words, threshold=0.5) == 1

    def test_count_custom_threshold(self):
        """Test counting with custom threshold."""
        words = [
            {"confidence": 0.3},
            {"confidence": 0.6},
            {"confidence": 0.9},
        ]
        assert count_low_confidence_words(words, threshold=0.7) == 2

    def test_count_empty(self):
        """Test counting with empty list."""
        assert count_low_confidence_words([]) == 0


class TestBuildTranscriptObject:
    """Tests for transcript object building."""

    def test_build_transcript(self, sample_speechmatics_response):
        """Test building transcript object."""
        transcript = build_transcript_object(
            sample_speechmatics_response, duration=7.8, language="en-US", average_confidence=0.97
        )

        assert "text" in transcript
        assert transcript["language"] == "en-US"
        assert transcript["duration"] == 7.8
        assert transcript["confidence"] == 0.97

    def test_build_transcript_text_content(self, sample_speechmatics_response):
        """Test that transcript text is properly assembled."""
        transcript = build_transcript_object(
            sample_speechmatics_response, duration=7.8, language="en-US", average_confidence=0.97
        )

        # Text should contain words from the response
        assert "Hello" in transcript["text"]
        assert "Alice" in transcript["text"]


class TestBuildSegments:
    """Tests for segments building."""

    def test_build_segments(self, sample_speechmatics_response):
        """Test building segments from response."""
        segments = build_segments(sample_speechmatics_response)

        # Should have 3 segments (based on sentence-ending punctuation)
        assert len(segments) == 3

        # Check first segment
        first_segment = segments[0]
        assert first_segment["id"] == 0
        assert first_segment["start"] == 0.5
        assert "Hello" in first_segment["text"]
        assert "confidence" in first_segment
        assert "words" in first_segment

    def test_build_segments_with_speaker(self, sample_speechmatics_response):
        """Test that segments include speaker information."""
        segments = build_segments(sample_speechmatics_response)

        # First two segments should be speaker S1
        assert segments[0].get("speaker") == "S1"
        assert segments[1].get("speaker") == "S1"

        # Third segment should be speaker S2
        assert segments[2].get("speaker") == "S2"

    def test_build_segments_empty(self, empty_response):
        """Test building segments from empty response."""
        segments = build_segments(empty_response)
        assert segments == []


class TestBuildWords:
    """Tests for words building."""

    def test_build_words(self, sample_speechmatics_response):
        """Test building words from response."""
        words = build_words(sample_speechmatics_response)

        # Should have words and punctuation
        assert len(words) > 0

        # Check first word
        first_word = words[0]
        assert first_word["id"] == 0
        assert first_word["text"] == "Hello"
        assert first_word["start"] == 0.5
        assert first_word["end"] == 0.8
        assert first_word["confidence"] == 0.98
        assert first_word["is_punctuation"] is False

    def test_build_words_punctuation(self, sample_speechmatics_response):
        """Test that punctuation is marked correctly."""
        words = build_words(sample_speechmatics_response)

        # Find punctuation marks
        punctuation = [w for w in words if w.get("is_punctuation")]
        assert len(punctuation) > 0

        # Check a punctuation mark
        comma = next(w for w in punctuation if w["text"] == ",")
        assert comma["is_punctuation"] is True

    def test_build_words_empty(self, empty_response):
        """Test building words from empty response."""
        words = build_words(empty_response)
        assert words == []


class TestBuildSpeakers:
    """Tests for speakers building."""

    def test_build_speakers(self, sample_speechmatics_response):
        """Test building speakers from response."""
        segments = build_segments(sample_speechmatics_response)
        speakers = build_speakers(sample_speechmatics_response, segments)

        # Should have 2 speakers
        assert len(speakers) == 2
        assert "S1" in speakers
        assert "S2" in speakers

    def test_build_speakers_info(self, sample_speechmatics_response):
        """Test speaker info structure."""
        segments = build_segments(sample_speechmatics_response)
        speakers = build_speakers(sample_speechmatics_response, segments)

        speaker1 = speakers["S1"]
        assert speaker1["id"] == "S1"
        assert speaker1["label"] == "Speaker S1"
        assert "segments" in speaker1
        assert "total_time" in speaker1
        assert "confidence" in speaker1

    def test_build_speakers_empty(self, empty_response):
        """Test building speakers from empty response."""
        speakers = build_speakers(empty_response, [])
        assert speakers == {}


class TestBuildMetadata:
    """Tests for metadata building."""

    def test_build_metadata(self, sample_speechmatics_response):
        """Test building metadata from response."""
        metadata = build_metadata(
            sample_speechmatics_response, duration=7.8, processing_time=5.0, created_at="2025-01-02T12:00:00Z"
        )

        assert metadata["provider"] == "speechmatics"
        assert metadata["model"] == "enhanced"
        assert metadata["created_at"] == "2025-01-02T12:00:00Z"
        assert metadata["processing_time"] == 5.0
        assert metadata["audio"]["duration"] == 7.8

    def test_build_metadata_audio_format(self, sample_speechmatics_response):
        """Test that audio format is extracted from filename."""
        metadata = build_metadata(sample_speechmatics_response, duration=7.8)
        assert metadata["audio"].get("format") == "wav"

    def test_build_metadata_defaults(self, empty_response):
        """Test metadata with defaults."""
        metadata = build_metadata(empty_response, duration=0.0)

        assert metadata["provider"] == "speechmatics"
        assert "created_at" in metadata
        assert "processed_at" in metadata


class TestBuildQuality:
    """Tests for quality metrics building."""

    def test_build_quality_high(self):
        """Test quality assessment for high confidence."""
        words = [{"confidence": 0.95}, {"confidence": 0.98}]
        quality = build_quality(words, [], 10.0)

        assert quality["audio_quality"] == "high"
        assert quality["average_confidence"] >= 0.9
        assert quality["low_confidence_words"] == 0

    def test_build_quality_medium(self):
        """Test quality assessment for medium confidence."""
        words = [{"confidence": 0.75}, {"confidence": 0.80}]
        quality = build_quality(words, [], 10.0)

        assert quality["audio_quality"] == "medium"

    def test_build_quality_low(self):
        """Test quality assessment for low confidence."""
        words = [{"confidence": 0.3}, {"confidence": 0.4}]
        quality = build_quality(words, [], 10.0)

        assert quality["audio_quality"] == "low"
        assert quality["low_confidence_words"] == 2

    def test_build_quality_multiple_speakers(self, sample_speechmatics_response):
        """Test multiple speakers detection."""
        words = build_words(sample_speechmatics_response)
        segments = build_segments(sample_speechmatics_response)
        quality = build_quality(words, segments, 7.8)

        assert quality["multiple_speakers"] is True


class TestConvertToWtf:
    """Tests for the main convert_to_wtf function."""

    def test_convert_full_response(self, sample_speechmatics_response):
        """Test converting a full Speechmatics response."""
        result = convert_to_wtf(sample_speechmatics_response, created_at="2025-01-02T12:00:00Z", processing_time=5.0)

        # Check required sections
        assert "transcript" in result
        assert "segments" in result
        assert "metadata" in result

        # Check optional sections
        assert "words" in result
        assert "speakers" in result
        assert "quality" in result
        assert "extensions" in result

    def test_convert_transcript_structure(self, sample_speechmatics_response):
        """Test transcript object structure."""
        result = convert_to_wtf(sample_speechmatics_response)
        transcript = result["transcript"]

        assert "text" in transcript
        assert "language" in transcript
        assert "duration" in transcript
        assert "confidence" in transcript
        assert isinstance(transcript["confidence"], float)
        assert 0.0 <= transcript["confidence"] <= 1.0

    def test_convert_segments_structure(self, sample_speechmatics_response):
        """Test segments array structure."""
        result = convert_to_wtf(sample_speechmatics_response)
        segments = result["segments"]

        assert len(segments) > 0

        for segment in segments:
            assert "id" in segment
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
            assert "confidence" in segment
            assert segment["end"] >= segment["start"]

    def test_convert_metadata_structure(self, sample_speechmatics_response):
        """Test metadata object structure."""
        result = convert_to_wtf(sample_speechmatics_response)
        metadata = result["metadata"]

        assert metadata["provider"] == "speechmatics"
        assert "created_at" in metadata
        assert "processed_at" in metadata
        assert "audio" in metadata

    def test_convert_extensions(self, sample_speechmatics_response):
        """Test that Speechmatics data is preserved in extensions."""
        result = convert_to_wtf(sample_speechmatics_response)
        extensions = result["extensions"]

        assert "speechmatics" in extensions
        assert "job" in extensions["speechmatics"]

    def test_convert_minimal_response(self, minimal_response):
        """Test converting a minimal response."""
        result = convert_to_wtf(minimal_response)

        assert "transcript" in result
        assert "segments" in result
        assert "metadata" in result

    def test_convert_empty_response(self, empty_response):
        """Test converting an empty response."""
        result = convert_to_wtf(empty_response)

        assert result["transcript"]["text"] == ""
        assert result["transcript"]["duration"] == 0.0
        assert result["segments"] == []

    def test_convert_confidence_normalization(self, sample_speechmatics_response):
        """Test that all confidence scores are normalized."""
        result = convert_to_wtf(sample_speechmatics_response)

        # Check transcript confidence
        assert 0.0 <= result["transcript"]["confidence"] <= 1.0

        # Check segment confidences
        for segment in result["segments"]:
            assert 0.0 <= segment["confidence"] <= 1.0

        # Check word confidences
        for word in result.get("words", []):
            assert 0.0 <= word["confidence"] <= 1.0

    def test_convert_processing_time(self, sample_speechmatics_response):
        """Test that processing time is included."""
        result = convert_to_wtf(sample_speechmatics_response, processing_time=12.5)

        assert result["metadata"]["processing_time"] == 12.5

    def test_convert_created_at(self, sample_speechmatics_response):
        """Test that created_at timestamp is preserved."""
        created = "2025-01-02T12:00:00Z"
        result = convert_to_wtf(sample_speechmatics_response, created_at=created)

        assert result["metadata"]["created_at"] == created


class TestWtfFormatCompliance:
    """Tests for WTF format compliance per draft-howe-vcon-wtf-extension-01."""

    def test_required_fields_present(self, sample_speechmatics_response):
        """Test that all required WTF fields are present."""
        result = convert_to_wtf(sample_speechmatics_response)

        # Required top-level sections
        assert "transcript" in result
        assert "segments" in result
        assert "metadata" in result

    def test_transcript_required_fields(self, sample_speechmatics_response):
        """Test transcript object required fields."""
        result = convert_to_wtf(sample_speechmatics_response)
        transcript = result["transcript"]

        assert "text" in transcript
        assert "language" in transcript
        assert "duration" in transcript
        assert "confidence" in transcript

    def test_segment_required_fields(self, sample_speechmatics_response):
        """Test segment object required fields."""
        result = convert_to_wtf(sample_speechmatics_response)

        for segment in result["segments"]:
            assert "id" in segment
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
            assert "confidence" in segment

    def test_metadata_required_fields(self, sample_speechmatics_response):
        """Test metadata object required fields."""
        result = convert_to_wtf(sample_speechmatics_response)
        metadata = result["metadata"]

        assert "created_at" in metadata
        assert "processed_at" in metadata
        assert "provider" in metadata
        assert "model" in metadata

    def test_language_bcp47_format(self, sample_speechmatics_response):
        """Test that language code is in BCP-47 format."""
        result = convert_to_wtf(sample_speechmatics_response)
        language = result["transcript"]["language"]

        # BCP-47 format should have a hyphen for regional variants
        # or be a 2-3 letter code for generic
        assert len(language) >= 2
