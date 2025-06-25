"""
Config flow for OpenAI TTS.
"""
from __future__ import annotations
from typing import Any
import os
import voluptuous as vol
import logging
from urllib.parse import urlparse
import uuid
import asyncio
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from homeassistant import data_entry_flow
from homeassistant.config_entries import ConfigFlow, OptionsFlow
from homeassistant.helpers.selector import selector
from homeassistant.helpers.selector import (
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.exceptions import HomeAssistantError

from .const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_VOICE,
    CONF_SPEED,
    CONF_URL,
    DOMAIN,
    MODELS,
    VOICES,
    UNIQUE_ID,
    CONF_CHIME_ENABLE,    # Use constant for chime enable toggle
    CONF_CHIME_SOUND,
    CONF_NORMALIZE_AUDIO,
    CONF_INSTRUCTIONS,
    # Add new constants for cache
    CONF_CACHE_ENABLED,
)
import os
import shutil

# Helper to fetch models from a remote OpenAI-compatible endpoint
async def fetch_models(session, api_key: str, url: str) -> list[str]:
    """
    Fetch available TTS models from the remote Kokoro/OpenAI-compatible endpoint.
    Returns a list of model IDs.
    """
    # Derive base endpoint
    if url.endswith("/audio/speech"):
        base_url = url.rsplit("/audio/speech", 1)[0]
    else:
        base_url = url.rstrip("/")
    models_url = f"{base_url}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with session.get(models_url, headers=headers, timeout=10) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch models: {resp.status}")
            data = await resp.json()
            # OpenAI returns {"data": [{"id": ...}, ...]}
            return [m["id"] for m in data.get("data", []) if m.get("id", "").startswith("tts")] or [m["id"] for m in data.get("data", [])]
    except Exception as e:
        _LOGGER.error(f"Could not fetch models from {models_url}: {e}")
        return MODELS  # fallback to static

# Helper to fetch voices from a remote OpenAI-compatible endpoint
async def fetch_voices(session, api_key: str, url: str) -> list[str]:
    """
    Fetch available voices from the remote Kokoro/OpenAI-compatible endpoint.
    Returns a list of voice names.
    """
    # Derive base endpoint
    if url.endswith("/audio/speech"):
        base_url = url.rsplit("/audio/speech", 1)[0]
    else:
        base_url = url.rstrip("/")
    voices_url = f"{base_url}/audio/voices"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with session.get(voices_url, headers=headers, timeout=10) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch voices: {resp.status}")
            data = await resp.json()
            # Kokoro returns a list, {"data": [...]}, or {"voices": [...]}
            if isinstance(data, dict):
                if "data" in data:
                    return data["data"]
                elif "voices" in data:
                    return data["voices"]
            elif isinstance(data, list):
                return data
            raise Exception(f"Unexpected voices response: {data}")
    except Exception as e:
        _LOGGER.error(f"Could not fetch voices from {voices_url}: {e}")
        return VOICES  # fallback to static

_LOGGER = logging.getLogger(__name__)

def generate_entry_id() -> str:
    return str(uuid.uuid4())

async def validate_user_input(user_input: dict):
    if user_input.get(CONF_MODEL) is None:
        raise ValueError("Model is required")
    if user_input.get(CONF_VOICE) is None:
        raise ValueError("Voice is required")

def get_chime_options() -> list[dict[str, str]]:
    """
    Scans the "chime" folder (located in the same directory as this file)
    and returns a list of options for the dropdown selector.
    Each option is a dict with 'value' (the file name) and 'label' (the file name without extension).
    """
    chime_folder = os.path.join(os.path.dirname(__file__), "chime")
    try:
        files = os.listdir(chime_folder)
    except Exception as err:
        _LOGGER.error("Error listing chime folder: %s", err)
        files = []
    options = []
    for file in files:
        if file.lower().endswith(".mp3"):
            label = os.path.splitext(file)[0].title()  # e.g. "Signal1.mp3" -> "Signal1"
            options.append({"value": file, "label": label})
    options.sort(key=lambda x: x["label"])
    return options

class OpenAITTSConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI TTS."""
    VERSION = 1

    def __init__(self):
        self._data = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """
        Step 1: Ask for base URL and API key (optional).
        """
        errors = {}
        description = (
            "<b>Base URL Examples:</b><br>"
            "<ul>"
            "<li><b>OpenAI:</b> https://api.openai.com/v1/audio/speech</li>"
            "<li><b>Local Kokoro:</b> http://localhost:8880/v1/audio/speech</li>"
            "</ul>"
            "<b>Endpoints Used:</b><br>"
            "<ul>"
            "<li><code>/v1/models</code> — fetch available models</li>"
            "<li><code>/v1/audio/voices</code> — fetch available voices</li>"
            "<li><code>/v1/audio/speech</code> — generate speech</li>"
            "</ul>"
            "<b>Instructions:</b> Enter the full base URL ending with <code>/v1/audio/speech</code>. The integration will automatically use the correct endpoints for each function."
        )
        if user_input is not None:
            self._data[CONF_API_KEY] = user_input.get(CONF_API_KEY, "")
            self._data[CONF_URL] = user_input[CONF_URL].rstrip("/")
            return await self.async_step_model()
        schema = vol.Schema({
            vol.Optional(CONF_API_KEY, default=""): str,
            vol.Required(
                CONF_URL,
                default="http://localhost:8880/v1/audio/speech"
            ): str
        })
        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
            description_placeholders={},
            description=description
        )

    async def async_step_model(self, user_input: dict[str, Any] | None = None):
        """
        Step 2: Fetch models and select model.
        """
        errors = {}
        session = async_get_clientsession(self.hass)
        base_url = self._data[CONF_URL]
        api_key = self._data.get(CONF_API_KEY, "")
        models = await fetch_models(session, api_key, f"{base_url}/v1/audio/speech")
        if user_input is not None:
            self._data[CONF_MODEL] = user_input[CONF_MODEL]
            return await self.async_step_voice()
        schema = vol.Schema({
            vol.Required(CONF_MODEL): selector({
                "select": {
                    "options": models,
                    "mode": "dropdown",
                    "sort": True,
                    "custom_value": True
                }
            })
        })
        return self.async_show_form(
            step_id="model",
            data_schema=schema,
            errors=errors
        )

    async def async_step_voice(self, user_input: dict[str, Any] | None = None):
        """
        Step 3: Fetch voices and select voice (plus speed/other options).
        """
        errors = {}
        session = async_get_clientsession(self.hass)
        base_url = self._data[CONF_URL]
        api_key = self._data.get(CONF_API_KEY, "")
        voices = await fetch_voices(session, api_key, f"{base_url}/v1/audio/speech")
        if user_input is not None:
            self._data[CONF_VOICE] = user_input[CONF_VOICE]
            self._data[CONF_SPEED] = user_input.get(CONF_SPEED, 1.0)
            # Compose the full endpoint for speech generation
            self._data[CONF_URL] = f"{base_url}/v1/audio/speech"
            entry_id = generate_entry_id()
            self._data[UNIQUE_ID] = entry_id
            await self.async_set_unique_id(entry_id)
            hostname = urlparse(self._data[CONF_URL]).hostname
            return self.async_create_entry(
                title=f"OpenAI TTS ({hostname}, {self._data[CONF_MODEL]})",
                data=self._data
            )
        schema = vol.Schema({
            vol.Required(CONF_VOICE, default=voices[0] if voices else ""): selector({
                "select": {
                    "options": voices,
                    "mode": "dropdown",
                    "sort": True,
                    "custom_value": True
                }
            }),
            vol.Optional(CONF_SPEED, default=1.0): selector({
                "number": {
                    "min": 0.25,
                    "max": 4.0,
                    "step": 0.05,
                    "mode": "slider"
                }
            })
        })
        return self.async_show_form(
            step_id="voice",
            data_schema=schema,
            errors=errors
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        return OpenAITTSOptionsFlow()

class OpenAITTSOptionsFlow(OptionsFlow):
    """Handle options flow for OpenAI TTS."""
    async def async_step_init(self, user_input: dict | None = None):
        # Handle purge action
        if user_input is not None and user_input.get("purge_cache"):
            await self._purge_tts_cache()
            return self.async_show_form(
                step_id="init",
                data_schema=self._get_options_schema(),
                description_placeholders={"info": "TTS cache purged successfully."}
            )
        if user_input is not None:
            # Ensure all returned values are strings where appropriate
            user_input[CONF_MODEL] = str(user_input.get(CONF_MODEL, ""))
            user_input[CONF_VOICE] = str(user_input.get(CONF_VOICE, ""))
            # Always save instructions as a string, default to empty string
            user_input[CONF_INSTRUCTIONS] = str(user_input.get(CONF_INSTRUCTIONS, ""))
            return self.async_create_entry(title="", data=user_input)
        return self.async_show_form(
            step_id="init",
            data_schema=self._get_options_schema(),
            description_placeholders={}
        )

    async def _purge_tts_cache(self):
        # Attempt to delete all files in the Home Assistant TTS cache directory
        tts_cache_path = os.path.join(self.hass.config.path("tts"))
        if os.path.exists(tts_cache_path):
            for filename in os.listdir(tts_cache_path):
                file_path = os.path.join(tts_cache_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    _LOGGER.error(f"Failed to delete {file_path}: {e}")

    def _get_options_schema(self):
        # Retrieve chime options using the executor to avoid blocking the event loop.
        import asyncio
        chime_options = asyncio.run(self.hass.async_add_executor_job(get_chime_options))
        # Fetch dynamic models and voices from the server
        api_key = self.config_entry.data.get(CONF_API_KEY, "")
        url = self.config_entry.data.get(CONF_URL, "http://localhost:8880/v1/audio/speech")
        if url.endswith("/v1/audio/speech"):
            base_url = url.rsplit("/v1/audio/speech", 1)[0]
        else:
            base_url = url.rstrip("/")
        session = async_get_clientsession(self.hass)
        models = asyncio.run(fetch_models(session, api_key, f"{base_url}/v1/audio/speech"))
        voices = asyncio.run(fetch_voices(session, api_key, f"{base_url}/v1/audio/speech"))
        current_voice = self.config_entry.options.get(CONF_VOICE, self.config_entry.data.get(CONF_VOICE, ""))
        if current_voice not in voices and voices:
            current_voice = voices[0]
        current_instructions = self.config_entry.options.get(CONF_INSTRUCTIONS, self.config_entry.data.get(CONF_INSTRUCTIONS, "")) or ""
        return vol.Schema({
            vol.Optional(
                CONF_CACHE_ENABLED,
                default=self.config_entry.options.get(CONF_CACHE_ENABLED, self.config_entry.data.get(CONF_CACHE_ENABLED, True))
            ): selector({"boolean": {}}),

            vol.Optional("purge_cache"): selector({"button": {"text": "Purge All TTS Cache"}}),

            vol.Optional(
                CONF_CHIME_ENABLE,
                default=self.config_entry.options.get(CONF_CHIME_ENABLE, self.config_entry.data.get(CONF_CHIME_ENABLE, False))
            ): selector({"boolean": {}}),

            vol.Optional(
                CONF_CHIME_SOUND,
                default=self.config_entry.options.get(CONF_CHIME_SOUND, self.config_entry.data.get(CONF_CHIME_SOUND, "threetone.mp3"))
            ): selector({
                "select": {
                    "options": chime_options
                }
            }),

            vol.Optional(
                CONF_MODEL,
                default=self.config_entry.options.get(CONF_MODEL, self.config_entry.data.get(CONF_MODEL, models[0] if models else ""))
            ): selector({
                "select": {
                    "options": models,
                    "mode": "dropdown",
                    "sort": True,
                    "custom_value": True
                }
            }),

            vol.Optional(
                CONF_SPEED,
                default=self.config_entry.options.get(CONF_SPEED, self.config_entry.data.get(CONF_SPEED, 1.0))
            ): selector({
                "number": {
                    "min": 0.25,
                    "max": 4.0,
                    "step": 0.05,
                    "mode": "slider"
                }
            }),

            vol.Optional(
                CONF_VOICE,
                default=current_voice
            ): selector({
                "select": {
                    "options": voices,
                    "mode": "dropdown",
                    "sort": True,
                    "custom_value": True
                }
            }),

            vol.Optional(
                 CONF_INSTRUCTIONS,
                 default=current_instructions
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT,multiline=True)
            ),

            vol.Optional(
                CONF_NORMALIZE_AUDIO,
                default=self.config_entry.options.get(CONF_NORMALIZE_AUDIO, self.config_entry.data.get(CONF_NORMALIZE_AUDIO, False))
            ): selector({"boolean": {}})
        })
        return self.async_show_form(step_id="init", data_schema=options_schema)

