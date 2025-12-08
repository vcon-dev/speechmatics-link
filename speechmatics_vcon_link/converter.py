"""WTF Format Converter

Converts Speechmatics transcription responses to the World Transcription Format (WTF)
as defined in draft-howe-vcon-wtf-extension-01.

The WTF format provides a standardized analysis framework for representing
speech-to-text transcription data from multiple providers within vCon containers.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def normalize_confidence(confidence: Optional[float]) -> float:
    """Normalize confidence score to [0.0, 1.0] range.

    Speechmatics already uses 0.0-1.0 scale, but this function
    ensures values are within bounds.

    Args:
        confidence: Raw confidence score

    Returns:
        Normalized confidence in [0.0, 1.0] range
    """
    if confidence is None:
        return 0.0
    return max(0.0, min(1.0, float(confidence)))


def extract_language_code(speechmatics_response: dict) -> str:
    """Extract BCP-47 language code from Speechmatics response.

    Args:
        speechmatics_response: Full Speechmatics API response

    Returns:
        BCP-47 formatted language code (e.g., "en-US")
    """
    # Speechmatics uses language codes like "en", "es", etc.
    # Map to BCP-47 format
    job_info = speechmatics_response.get("job", {})
    config = job_info.get("transcription_config", {})
    language = config.get("language", "und")  # "und" = undetermined

    # Common language mappings to BCP-47
    language_map = {
        "en": "en-US",
        "en-GB": "en-GB",
        "en-AU": "en-AU",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-PT",
        "pt-BR": "pt-BR",
        "nl": "nl-NL",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
        "ar": "ar-SA",
        "ru": "ru-RU",
        "auto": "und",
    }

    return language_map.get(language, language)


def calculate_duration(speechmatics_response: dict) -> float:
    """Calculate total audio duration from the response.

    Args:
        speechmatics_response: Full Speechmatics API response

    Returns:
        Duration in seconds
    """
    results = speechmatics_response.get("results", [])
    if not results:
        return 0.0

    # Find the maximum end time across all results
    max_end_time = 0.0
    for result in results:
        end_time = result.get("end_time", 0.0)
        if end_time > max_end_time:
            max_end_time = end_time

    return max_end_time


def calculate_average_confidence(words: list) -> float:
    """Calculate average confidence across all words.

    Args:
        words: List of word objects with confidence scores

    Returns:
        Average confidence score
    """
    if not words:
        return 0.0

    confidences = [w.get("confidence", 0.0) for w in words if "confidence" in w]
    if not confidences:
        return 0.0

    return sum(confidences) / len(confidences)


def count_low_confidence_words(words: list, threshold: float = 0.5) -> int:
    """Count words with confidence below threshold.

    Args:
        words: List of word objects with confidence scores
        threshold: Confidence threshold (default 0.5)

    Returns:
        Count of low confidence words
    """
    return sum(1 for w in words if w.get("confidence", 1.0) < threshold)


def build_transcript_object(
    speechmatics_response: dict, duration: float, language: str, average_confidence: float
) -> dict:
    """Build the required WTF transcript object.

    Args:
        speechmatics_response: Full Speechmatics API response
        duration: Total audio duration
        language: BCP-47 language code
        average_confidence: Overall confidence score

    Returns:
        WTF transcript object
    """
    # Build full transcript text from results
    results = speechmatics_response.get("results", [])
    transcript_parts = []

    for result in results:
        if result.get("type") == "word":
            transcript_parts.append(result.get("alternatives", [{}])[0].get("content", ""))
        elif result.get("type") == "punctuation":
            # Append punctuation without space
            if transcript_parts:
                transcript_parts[-1] += result.get("alternatives", [{}])[0].get("content", "")

    full_text = " ".join(transcript_parts)

    return {
        "text": full_text,
        "language": language,
        "duration": duration,
        "confidence": normalize_confidence(average_confidence),
    }


def build_segments(speechmatics_response: dict) -> list:
    """Build WTF segments array from Speechmatics response.

    Segments are built from sentence-like chunks. Speechmatics doesn't
    have explicit segments, so we create them from punctuation boundaries.

    Args:
        speechmatics_response: Full Speechmatics API response

    Returns:
        List of WTF segment objects
    """
    results = speechmatics_response.get("results", [])
    if not results:
        return []

    segments = []
    current_segment = {
        "id": 0,
        "start": None,
        "end": None,
        "text_parts": [],
        "word_indices": [],
        "confidences": [],
        "speaker": None,
    }
    word_index = 0
    segment_id = 0

    # Punctuation that ends a segment
    segment_endings = {".", "!", "?"}

    for result in results:
        result_type = result.get("type")

        if result_type == "word":
            alternatives = result.get("alternatives", [{}])
            content = alternatives[0].get("content", "") if alternatives else ""
            confidence = alternatives[0].get("confidence", 0.0) if alternatives else 0.0
            start_time = result.get("start_time", 0.0)
            end_time = result.get("end_time", 0.0)
            speaker = alternatives[0].get("speaker", None) if alternatives else None

            if current_segment["start"] is None:
                current_segment["start"] = start_time

            current_segment["end"] = end_time
            current_segment["text_parts"].append(content)
            current_segment["word_indices"].append(word_index)
            current_segment["confidences"].append(confidence)

            if speaker is not None and current_segment["speaker"] is None:
                current_segment["speaker"] = speaker

            word_index += 1

        elif result_type == "punctuation":
            alternatives = result.get("alternatives", [{}])
            content = alternatives[0].get("content", "") if alternatives else ""

            # Add punctuation to current segment
            if current_segment["text_parts"]:
                current_segment["text_parts"][-1] += content

            # Check if this punctuation ends a segment
            if content in segment_endings and current_segment["text_parts"]:
                # Calculate segment confidence as average of word confidences
                avg_conf = (
                    sum(current_segment["confidences"]) / len(current_segment["confidences"])
                    if current_segment["confidences"]
                    else 0.0
                )

                segment = {
                    "id": segment_id,
                    "start": current_segment["start"],
                    "end": current_segment["end"],
                    "text": " ".join(current_segment["text_parts"]),
                    "confidence": normalize_confidence(avg_conf),
                    "words": current_segment["word_indices"],
                }

                if current_segment["speaker"] is not None:
                    segment["speaker"] = current_segment["speaker"]

                segments.append(segment)
                segment_id += 1

                # Reset for next segment
                current_segment = {
                    "id": segment_id,
                    "start": None,
                    "end": None,
                    "text_parts": [],
                    "word_indices": [],
                    "confidences": [],
                    "speaker": None,
                }

    # Add any remaining content as final segment
    if current_segment["text_parts"]:
        avg_conf = (
            sum(current_segment["confidences"]) / len(current_segment["confidences"])
            if current_segment["confidences"]
            else 0.0
        )

        segment = {
            "id": segment_id,
            "start": current_segment["start"] or 0.0,
            "end": current_segment["end"] or 0.0,
            "text": " ".join(current_segment["text_parts"]),
            "confidence": normalize_confidence(avg_conf),
            "words": current_segment["word_indices"],
        }

        if current_segment["speaker"] is not None:
            segment["speaker"] = current_segment["speaker"]

        segments.append(segment)

    return segments


def build_words(speechmatics_response: dict) -> list:
    """Build WTF words array from Speechmatics response.

    Args:
        speechmatics_response: Full Speechmatics API response

    Returns:
        List of WTF word objects
    """
    results = speechmatics_response.get("results", [])
    words = []
    word_id = 0

    for result in results:
        result_type = result.get("type")
        alternatives = result.get("alternatives", [{}])

        if not alternatives:
            continue

        alt = alternatives[0]
        content = alt.get("content", "")
        confidence = alt.get("confidence", 0.0)
        speaker = alt.get("speaker", None)

        if result_type == "word":
            word = {
                "id": word_id,
                "start": result.get("start_time", 0.0),
                "end": result.get("end_time", 0.0),
                "text": content,
                "confidence": normalize_confidence(confidence),
                "is_punctuation": False,
            }

            if speaker is not None:
                word["speaker"] = speaker

            words.append(word)
            word_id += 1

        elif result_type == "punctuation":
            word = {
                "id": word_id,
                "start": result.get("start_time", 0.0),
                "end": result.get("end_time", 0.0),
                "text": content,
                "confidence": normalize_confidence(confidence),
                "is_punctuation": True,
            }

            words.append(word)
            word_id += 1

    return words


def build_speakers(speechmatics_response: dict, segments: list) -> dict:
    """Build WTF speakers object from Speechmatics response.

    Args:
        speechmatics_response: Full Speechmatics API response
        segments: Pre-built segments list

    Returns:
        Dictionary mapping speaker IDs to speaker info
    """
    results = speechmatics_response.get("results", [])

    # Collect speaker information
    speaker_data = {}  # speaker_id -> {"segments": [], "total_time": 0, "confidences": []}

    # First pass: collect from results
    for result in results:
        if result.get("type") != "word":
            continue

        alternatives = result.get("alternatives", [{}])
        if not alternatives:
            continue

        speaker = alternatives[0].get("speaker")
        if speaker is None:
            continue

        speaker_str = str(speaker)
        if speaker_str not in speaker_data:
            speaker_data[speaker_str] = {"segments": set(), "total_time": 0.0, "confidences": []}

        duration = result.get("end_time", 0.0) - result.get("start_time", 0.0)
        speaker_data[speaker_str]["total_time"] += duration

        confidence = alternatives[0].get("confidence", 0.0)
        speaker_data[speaker_str]["confidences"].append(confidence)

    # Second pass: map segments to speakers
    for segment in segments:
        speaker = segment.get("speaker")
        if speaker is not None:
            speaker_str = str(speaker)
            if speaker_str in speaker_data:
                speaker_data[speaker_str]["segments"].add(segment["id"])

    # Build final speakers object
    speakers = {}
    for speaker_id, data in speaker_data.items():
        avg_confidence = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0.0

        speakers[speaker_id] = {
            "id": speaker_id,
            "label": f"Speaker {speaker_id}",
            "segments": sorted(list(data["segments"])),
            "total_time": round(data["total_time"], 2),
            "confidence": normalize_confidence(avg_confidence),
        }

    return speakers


def build_metadata(
    speechmatics_response: dict,
    duration: float,
    processing_time: Optional[float] = None,
    created_at: Optional[str] = None,
) -> dict:
    """Build WTF metadata object.

    Args:
        speechmatics_response: Full Speechmatics API response
        duration: Audio duration in seconds
        processing_time: Time taken to process (optional)
        created_at: ISO 8601 timestamp when audio was created

    Returns:
        WTF metadata object
    """
    now = datetime.now(timezone.utc).isoformat()

    job_info = speechmatics_response.get("job", {})
    config = job_info.get("transcription_config", {})

    metadata = {
        "created_at": created_at or now,
        "processed_at": now,
        "provider": "speechmatics",
        "model": config.get("operating_point", "enhanced"),
        "audio": {"duration": duration},
        "options": {
            "language": config.get("language", "auto"),
            "operating_point": config.get("operating_point", "enhanced"),
            "diarization": "speaker" in str(config.get("diarization", "")),
        },
    }

    if processing_time is not None:
        metadata["processing_time"] = processing_time

    # Add audio format info if available
    data_name = job_info.get("data_name", "")
    if data_name:
        if "." in data_name:
            metadata["audio"]["format"] = data_name.split(".")[-1].lower()

    return metadata


def build_quality(words: list, segments: list, duration: float) -> dict:
    """Build WTF quality metrics object.

    Args:
        words: List of word objects
        segments: List of segment objects
        duration: Total audio duration

    Returns:
        WTF quality object
    """
    # Calculate metrics
    avg_confidence = calculate_average_confidence(
        [{"confidence": w.get("confidence", 0)} for w in words if not w.get("is_punctuation")]
    )
    low_conf_count = count_low_confidence_words(
        [{"confidence": w.get("confidence", 0)} for w in words if not w.get("is_punctuation")]
    )

    # Determine audio quality based on average confidence
    if avg_confidence >= 0.9:
        audio_quality = "high"
    elif avg_confidence >= 0.7:
        audio_quality = "medium"
    else:
        audio_quality = "low"

    # Check for multiple speakers
    speakers_found = set()
    for word in words:
        speaker = word.get("speaker")
        if speaker is not None:
            speakers_found.add(speaker)

    multiple_speakers = len(speakers_found) > 1

    # Calculate approximate silence ratio (rough estimate)
    # Sum of word durations vs total duration
    word_time = sum(w.get("end", 0) - w.get("start", 0) for w in words if not w.get("is_punctuation"))
    silence_ratio = max(0.0, min(1.0, 1.0 - (word_time / duration))) if duration > 0 else 0.0

    return {
        "audio_quality": audio_quality,
        "background_noise": 0.0,  # Not directly available from Speechmatics
        "multiple_speakers": multiple_speakers,
        "overlapping_speech": False,  # Would need additional analysis
        "silence_ratio": round(silence_ratio, 2),
        "average_confidence": round(avg_confidence, 2),
        "low_confidence_words": low_conf_count,
        "processing_warnings": [],
    }


def convert_to_wtf(
    speechmatics_response: dict, created_at: Optional[str] = None, processing_time: Optional[float] = None
) -> dict:
    """Convert a Speechmatics transcription response to WTF format.

    This is the main entry point for converting Speechmatics JSON responses
    to the World Transcription Format (WTF) schema as defined in
    draft-howe-vcon-wtf-extension-01.

    Args:
        speechmatics_response: Full Speechmatics API response (JSON)
        created_at: ISO 8601 timestamp when the original audio was created
        processing_time: Time taken to process the transcription

    Returns:
        Dictionary conforming to the WTF schema with:
        - transcript: Required transcript object
        - segments: Required segments array
        - metadata: Required metadata object
        - words: Optional word-level details
        - speakers: Optional speaker diarization
        - quality: Optional quality metrics
        - extensions: Provider-specific data (Speechmatics)
    """
    logger.info("Converting Speechmatics response to WTF format")

    # Extract basic info
    duration = calculate_duration(speechmatics_response)
    language = extract_language_code(speechmatics_response)

    # Build word list first (needed for other calculations)
    words = build_words(speechmatics_response)

    # Calculate average confidence from words
    word_confidences = [w.get("confidence", 0) for w in words if not w.get("is_punctuation")]
    avg_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0

    # Build segments
    segments = build_segments(speechmatics_response)

    # Build required sections
    transcript = build_transcript_object(speechmatics_response, duration, language, avg_confidence)

    metadata = build_metadata(speechmatics_response, duration, processing_time, created_at)

    # Build optional sections
    speakers = build_speakers(speechmatics_response, segments)
    quality = build_quality(words, segments, duration)

    # Construct WTF format
    wtf_result = {
        "transcript": transcript,
        "segments": segments,
        "metadata": metadata,
    }

    # Add optional sections if they have content
    if words:
        wtf_result["words"] = words

    if speakers:
        wtf_result["speakers"] = speakers

    wtf_result["quality"] = quality

    # Preserve original Speechmatics data in extensions
    wtf_result["extensions"] = {
        "speechmatics": {
            "job": speechmatics_response.get("job", {}),
            "format": speechmatics_response.get("format", ""),
        }
    }

    logger.info(f"Converted to WTF format: {len(segments)} segments, {len(words)} words")

    return wtf_result
