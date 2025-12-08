"""Speechmatics API Client

A client wrapper for the Speechmatics batch transcription API with retry logic
and error handling.
"""

import logging
import time
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Speechmatics job status values."""
    RUNNING = "running"
    DONE = "done"
    REJECTED = "rejected"
    DELETED = "deleted"


@dataclass
class TranscriptionConfig:
    """Configuration for a transcription job."""
    language: Optional[str] = None  # None for auto-detect
    operating_point: str = "enhanced"  # "standard" or "enhanced"
    enable_diarization: bool = False
    diarization_max_speakers: Optional[int] = None
    enable_entities: bool = False
    output_locale: Optional[str] = None
    additional_vocab: Optional[list] = None


class SpeechmaticsError(Exception):
    """Base exception for Speechmatics API errors."""
    pass


class SpeechmaticsAuthError(SpeechmaticsError):
    """Authentication error (invalid API key)."""
    pass


class SpeechmaticsJobError(SpeechmaticsError):
    """Error related to job processing."""
    pass


class SpeechmaticsTimeoutError(SpeechmaticsError):
    """Timeout waiting for job completion."""
    pass


class SpeechmaticsClient:
    """Client for interacting with the Speechmatics batch transcription API.
    
    Attributes:
        api_key: Speechmatics API key
        api_url: Base URL for the Speechmatics API
        timeout: Request timeout in seconds
    """
    
    DEFAULT_API_URL = "https://asr.api.speechmatics.com/v2"
    
    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        timeout: int = 300
    ):
        """Initialize the Speechmatics client.
        
        Args:
            api_key: Speechmatics API key
            api_url: Base URL for the API (default: production URL)
            timeout: Request timeout in seconds
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.api_url = (api_url or self.DEFAULT_API_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
        })
    
    def _build_config(self, config: TranscriptionConfig) -> dict:
        """Build the transcription config payload.
        
        Args:
            config: Transcription configuration
            
        Returns:
            Dictionary formatted for the Speechmatics API
        """
        transcription_config = {
            "operating_point": config.operating_point,
        }
        
        if config.language:
            transcription_config["language"] = config.language
        else:
            # Auto-detect language
            transcription_config["language"] = "auto"
        
        if config.enable_diarization:
            transcription_config["diarization"] = "speaker"
            if config.diarization_max_speakers:
                transcription_config["speaker_diarization_config"] = {
                    "max_speakers": config.diarization_max_speakers
                }
        
        if config.enable_entities:
            transcription_config["enable_entities"] = True
        
        if config.output_locale:
            transcription_config["output_locale"] = config.output_locale
        
        if config.additional_vocab:
            transcription_config["additional_vocab"] = config.additional_vocab
        
        return {
            "type": "transcription",
            "transcription_config": transcription_config,
        }
    
    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def submit_job(
        self,
        audio_url: str,
        config: Optional[TranscriptionConfig] = None
    ) -> str:
        """Submit a transcription job for an audio file.
        
        Args:
            audio_url: URL of the audio file to transcribe
            config: Transcription configuration (uses defaults if not provided)
            
        Returns:
            Job ID for the submitted job
            
        Raises:
            SpeechmaticsAuthError: If API key is invalid
            SpeechmaticsError: If job submission fails
        """
        import json as json_module
        
        if config is None:
            config = TranscriptionConfig()
        
        job_config = self._build_config(config)
        
        # Add fetch configuration for URL-based audio
        job_config["fetch_data"] = {
            "url": audio_url
        }
        
        url = f"{self.api_url}/jobs"
        
        logger.info(f"Submitting transcription job for: {audio_url}")
        
        try:
            # Speechmatics API requires multipart/form-data
            # The config is sent as a JSON string in the 'config' field
            files = {
                'config': (None, json_module.dumps(job_config), 'application/json')
            }
            
            response = self._session.post(
                url,
                files=files,
                timeout=self.timeout
            )
            
            if response.status_code == 401:
                raise SpeechmaticsAuthError("Invalid API key")
            
            if response.status_code == 403:
                raise SpeechmaticsAuthError("API key does not have required permissions")
            
            response.raise_for_status()
            
            result = response.json()
            job_id = result.get("id")
            
            if not job_id:
                raise SpeechmaticsError("No job ID returned from API")
            
            logger.info(f"Job submitted successfully: {job_id}")
            return job_id
            
        except requests.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("message", str(e))
            except Exception:
                error_detail = str(e)
            raise SpeechmaticsError(f"Failed to submit job: {error_detail}") from e
    
    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_job_status(self, job_id: str) -> dict:
        """Get the status of a transcription job.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Dictionary containing job status information
            
        Raises:
            SpeechmaticsError: If status check fails
        """
        url = f"{self.api_url}/jobs/{job_id}"
        
        try:
            response = self._session.get(url, timeout=self.timeout)
            
            if response.status_code == 401:
                raise SpeechmaticsAuthError("Invalid API key")
            
            if response.status_code == 404:
                raise SpeechmaticsJobError(f"Job not found: {job_id}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.HTTPError as e:
            raise SpeechmaticsError(f"Failed to get job status: {e}") from e
    
    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_transcript(self, job_id: str, output_format: str = "json-v2") -> dict:
        """Get the transcript for a completed job.
        
        Args:
            job_id: The job ID to retrieve transcript for
            output_format: Output format (default: json-v2)
            
        Returns:
            Dictionary containing the transcript data
            
        Raises:
            SpeechmaticsError: If transcript retrieval fails
        """
        url = f"{self.api_url}/jobs/{job_id}/transcript"
        params = {"format": output_format}
        
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 401:
                raise SpeechmaticsAuthError("Invalid API key")
            
            if response.status_code == 404:
                raise SpeechmaticsJobError(f"Job or transcript not found: {job_id}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.HTTPError as e:
            raise SpeechmaticsError(f"Failed to get transcript: {e}") from e
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 5,
        max_attempts: int = 120
    ) -> dict:
        """Wait for a job to complete and return the transcript.
        
        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks
            max_attempts: Maximum number of polling attempts
            
        Returns:
            Dictionary containing the transcript data
            
        Raises:
            SpeechmaticsTimeoutError: If job doesn't complete in time
            SpeechmaticsJobError: If job fails or is rejected
        """
        logger.info(f"Waiting for job {job_id} to complete...")
        
        for attempt in range(max_attempts):
            status_info = self.get_job_status(job_id)
            status = status_info.get("job", {}).get("status", "unknown")
            
            logger.debug(f"Job {job_id} status: {status} (attempt {attempt + 1}/{max_attempts})")
            
            if status == JobStatus.DONE.value:
                logger.info(f"Job {job_id} completed successfully")
                return self.get_transcript(job_id)
            
            if status == JobStatus.REJECTED.value:
                error_msg = status_info.get("job", {}).get("errors", ["Unknown error"])
                raise SpeechmaticsJobError(f"Job was rejected: {error_msg}")
            
            if status == JobStatus.DELETED.value:
                raise SpeechmaticsJobError(f"Job was deleted: {job_id}")
            
            if status != JobStatus.RUNNING.value:
                logger.warning(f"Unexpected job status: {status}")
            
            time.sleep(poll_interval)
        
        raise SpeechmaticsTimeoutError(
            f"Job {job_id} did not complete within {max_attempts * poll_interval} seconds"
        )
    
    def transcribe(
        self,
        audio_url: str,
        config: Optional[TranscriptionConfig] = None,
        poll_interval: int = 5,
        max_attempts: int = 120
    ) -> dict:
        """Submit a job and wait for the transcript.
        
        This is a convenience method that combines submit_job and wait_for_completion.
        
        Args:
            audio_url: URL of the audio file to transcribe
            config: Transcription configuration
            poll_interval: Seconds between status checks
            max_attempts: Maximum number of polling attempts
            
        Returns:
            Dictionary containing the transcript data
        """
        job_id = self.submit_job(audio_url, config)
        return self.wait_for_completion(job_id, poll_interval, max_attempts)
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job.
        
        Args:
            job_id: The job ID to delete
            
        Returns:
            True if deletion was successful
        """
        url = f"{self.api_url}/jobs/{job_id}"
        
        try:
            response = self._session.delete(url, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Job {job_id} deleted successfully")
            return True
        except requests.HTTPError as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

