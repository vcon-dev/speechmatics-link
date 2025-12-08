"""Speechmatics vCon Link

A vCon server link for transcribing audio dialogs using the Speechmatics API.
Supports output in both Speechmatics native format and WTF (World Transcription Format).

This link processes vCon objects by:
1. Retrieving vCons from Redis
2. Identifying audio dialogs (type="recording")
3. Submitting audio to Speechmatics for transcription
4. Storing results as vCon analysis entries
5. Optionally converting to WTF format for standardized transcription data

Usage:
    Configure in vcon-server config.yml:

    links:
      speechmatics:
        module: speechmatics_vcon_link
        pip_name: git+https://github.com/yourusername/speechmatics-link.git
        options:
          api_key: ${SPEECHMATICS_API_KEY}
          save_native_format: true
          save_wtf_format: true
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import redis
from redis.commands.json.path import Path
import vcon

from .client import (
    SpeechmaticsClient,
    TranscriptionConfig,
    SpeechmaticsError,
    SpeechmaticsAuthError,
    SpeechmaticsTimeoutError,
)
from .converter import convert_to_wtf

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Default configuration options
default_options = {
    # Speechmatics API settings
    "api_key": os.getenv("SPEECHMATICS_API_KEY"),
    "api_url": "https://asr.api.speechmatics.com/v2",
    # Output format options
    "save_native_format": True,
    "save_wtf_format": True,
    # Transcription settings
    "model": "enhanced",  # "standard" or "enhanced"
    "language": None,  # None for auto-detect
    "enable_diarization": False,
    "diarization_max_speakers": None,
    # Polling settings
    "poll_interval": 5,
    "max_poll_attempts": 120,
    # Processing behavior
    "skip_if_exists": True,
    # Redis settings
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_db": 0,
}


def get_redis_connection(opts: dict):
    """Get Redis connection based on options.

    Args:
        opts: Configuration options

    Returns:
        Redis connection object
    """
    host = opts.get("redis_host", "localhost")
    port = opts.get("redis_port", 6379)
    db = opts.get("redis_db", 0)

    return redis.Redis(host=host, port=port, db=db, decode_responses=False)


def get_vcon_from_redis(redis_conn, vcon_uuid: str) -> Optional[vcon.Vcon]:
    """Retrieve a vCon from Redis.

    Args:
        redis_conn: Redis connection object
        vcon_uuid: UUID of the vCon to retrieve

    Returns:
        vCon object or None if not found
    """
    try:
        vcon_dict = redis_conn.json().get(f"vcon:{vcon_uuid}", Path.root_path())
        if not vcon_dict:
            return None
        return vcon.Vcon(vcon_dict)
    except Exception as e:
        logger.error(f"Error retrieving vCon {vcon_uuid} from Redis: {e}")
        return None


def store_vcon_to_redis(redis_conn, vcon_obj: vcon.Vcon) -> bool:
    """Store a vCon to Redis.

    Args:
        redis_conn: Redis connection object
        vcon_obj: vCon object to store

    Returns:
        True if successful, False otherwise
    """
    try:
        key = f"vcon:{vcon_obj.uuid}"
        vcon_dict = vcon_obj.to_dict()
        redis_conn.json().set(key, Path.root_path(), vcon_dict)
        return True
    except Exception as e:
        logger.error(f"Error storing vCon {vcon_obj.uuid} to Redis: {e}")
        return False


def has_transcription(vcon_obj: vcon.Vcon, dialog_index: int, analysis_type: str) -> bool:
    """Check if a transcription already exists for a dialog.

    Args:
        vcon_obj: The vCon object
        dialog_index: Index of the dialog
        analysis_type: Type of analysis to check for

    Returns:
        True if transcription exists, False otherwise
    """
    for analysis in vcon_obj.analysis:
        if analysis.get("dialog") == dialog_index and analysis.get("type") == analysis_type:
            return True
    return False


def get_audio_url(dialog: dict) -> Optional[str]:
    """Extract audio URL from a dialog object.

    Args:
        dialog: Dialog dictionary from vCon

    Returns:
        Audio URL or None if not found
    """
    # Check for URL field
    url = dialog.get("url")
    if url:
        return url

    # Check for filename that might be a URL
    filename = dialog.get("filename")
    if filename and (filename.startswith("http://") or filename.startswith("https://")):
        return filename

    return None


def is_audio_dialog(dialog: dict) -> bool:
    """Check if a dialog contains audio content.

    Args:
        dialog: Dialog dictionary from vCon

    Returns:
        True if dialog is an audio recording
    """
    dialog_type = dialog.get("type", "")
    mimetype = dialog.get("mimetype", "")

    # Check type field
    if dialog_type in ("recording", "audio"):
        return True

    # Check mimetype
    if mimetype.startswith("audio/"):
        return True

    return False


def run(vcon_uuid: str, link_name: str, opts: dict = None) -> Optional[str]:
    """Main link function - processes vCon through Speechmatics transcription.

    This function is called by the vCon server to process a vCon through this link.
    It iterates through audio dialogs, submits them to Speechmatics for transcription,
    and stores the results as analysis entries.

    Args:
        vcon_uuid: UUID of the vCon to process
        link_name: Name of this link instance (from config)
        opts: Configuration options for this link

    Returns:
        vcon_uuid (str) if processing should continue, None to stop the chain

    Raises:
        ValueError: If required configuration is missing
        SpeechmaticsError: If transcription fails
    """
    module_name = __name__.split(".")[-1]
    logger.info(f"Starting {module_name}:{link_name} plugin for: {vcon_uuid}")

    # Merge provided options with defaults
    merged_opts = default_options.copy()
    if opts:
        merged_opts.update(opts)
    opts = merged_opts

    # Validate required options
    api_key = opts.get("api_key")
    if not api_key:
        raise ValueError(
            "Speechmatics API key is required. "
            "Set SPEECHMATICS_API_KEY environment variable or provide api_key in options."
        )

    # Check that at least one output format is enabled
    if not opts.get("save_native_format") and not opts.get("save_wtf_format"):
        logger.warning("Neither save_native_format nor save_wtf_format is enabled. Enabling both.")
        opts["save_native_format"] = True
        opts["save_wtf_format"] = True

    # Get Redis connection
    redis_conn = get_redis_connection(opts)

    # Retrieve vCon from Redis
    vcon_obj = get_vcon_from_redis(redis_conn, vcon_uuid)
    if not vcon_obj:
        logger.error(f"vCon not found: {vcon_uuid}")
        return None

    # Initialize Speechmatics client
    client = SpeechmaticsClient(api_key=api_key, api_url=opts.get("api_url"))

    # Build transcription config
    transcription_config = TranscriptionConfig(
        language=opts.get("language"),
        operating_point=opts.get("model", "enhanced"),
        enable_diarization=opts.get("enable_diarization", False),
        diarization_max_speakers=opts.get("diarization_max_speakers"),
    )

    # Track if we made any changes
    vcon_modified = False
    dialogs_processed = 0

    # Process each dialog
    for index, dialog in enumerate(vcon_obj.dialog):
        # Skip non-audio dialogs
        if not is_audio_dialog(dialog):
            logger.debug(f"Skipping dialog {index}: not an audio recording")
            continue

        # Check if already transcribed
        if opts.get("skip_if_exists"):
            if opts.get("save_native_format") and has_transcription(vcon_obj, index, "speechmatics_transcription"):
                logger.info(f"Dialog {index} already has native transcription, skipping")
                continue

            if opts.get("save_wtf_format") and has_transcription(vcon_obj, index, "wtf_transcription"):
                logger.info(f"Dialog {index} already has WTF transcription, skipping")
                continue

        # Get audio URL
        audio_url = get_audio_url(dialog)
        if not audio_url:
            logger.warning(f"Dialog {index} has no audio URL, skipping")
            continue

        logger.info(f"Processing dialog {index}: {audio_url}")

        try:
            # Track processing time
            start_time = time.time()

            # Submit and wait for transcription
            transcript_result = client.transcribe(
                audio_url=audio_url,
                config=transcription_config,
                poll_interval=opts.get("poll_interval", 5),
                max_attempts=opts.get("max_poll_attempts", 120),
            )

            processing_time = time.time() - start_time

            # Get dialog start time for metadata
            dialog_start = dialog.get("start")

            # Save native format
            if opts.get("save_native_format"):
                vcon_obj.add_analysis(
                    type="speechmatics_transcription",
                    dialog=index,
                    vendor="speechmatics",
                    body=transcript_result,
                    encoding="json",
                )
                logger.info(f"Added native transcription for dialog {index}")

            # Save WTF format
            if opts.get("save_wtf_format"):
                wtf_result = convert_to_wtf(
                    transcript_result,
                    created_at=dialog_start,
                    processing_time=processing_time,
                )

                vcon_obj.add_analysis(
                    type="wtf_transcription",
                    dialog=index,
                    vendor="speechmatics",
                    body=wtf_result,
                    encoding="json",
                )
                logger.info(f"Added WTF transcription for dialog {index}")

            vcon_modified = True
            dialogs_processed += 1

        except SpeechmaticsAuthError as e:
            logger.error(f"Authentication error for dialog {index}: {e}")
            raise
        except SpeechmaticsTimeoutError as e:
            logger.error(f"Timeout waiting for dialog {index}: {e}")
            raise
        except SpeechmaticsError as e:
            logger.error(f"Transcription failed for dialog {index}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing dialog {index}: {e}", exc_info=True)
            raise

    # Store updated vCon back to Redis if modified
    if vcon_modified:
        if not store_vcon_to_redis(redis_conn, vcon_obj):
            logger.error(f"Failed to store updated vCon: {vcon_uuid}")
            return None
        logger.info(f"Stored updated vCon with {dialogs_processed} transcriptions")
    else:
        logger.info(f"No dialogs processed for vCon: {vcon_uuid}")

    logger.info(f"Finished {module_name}:{link_name} plugin for: {vcon_uuid}")
    return vcon_uuid
