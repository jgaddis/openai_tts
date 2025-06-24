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
)

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
    headers = {"Authorization": f"Bearer {api_key}"}
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
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.get(voices_url, headers=headers, timeout=10) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch voices: {resp.status}")
            data = await resp.json()
            # Kokoro returns a list or {"data": ["voice1", ...]}
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            elif isinstance(data, list):
                return data
            else:
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

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """
        Step 1: Ask for API key and endpoint.
        Step 2: Fetch models and present model/voice selection.
        """
        errors = {}
        if user_input is not None:
            # Step 2: Model/voice selection
            if CONF_MODEL in user_input and CONF_VOICE in user_input:
                try:
                    await validate_user_input(user_input)
                    entry_id = generate_entry_id()
                    user_input[UNIQUE_ID] = entry_id
                    await self.async_set_unique_id(entry_id)
                    hostname = urlparse(user_input[CONF_URL]).hostname
                    return self.async_create_entry(
                        title=f"OpenAI TTS ({hostname}, {user_input[CONF_MODEL]})",
                        data=user_input
                    )
                except data_entry_flow.AbortFlow:
                    return self.async_abort(reason="already_configured")
                except HomeAssistantError as e:
                    _LOGGER.exception(str(e))
                    errors["base"] = str(e)
                except ValueError as e:
                    _LOGGER.exception(str(e))
                    errors["base"] = str(e)
                except Exception as e:
                    _LOGGER.exception(str(e))
                    errors["base"] = "unknown_error"
            else:
                # Step 1: API key and URL submitted, now fetch models
                api_key = user_input.get(CONF_API_KEY)
                url = user_input.get(CONF_URL, "https://api.openai.com/v1/audio/speech")
                speed = user_input.get(CONF_SPEED, 1.0)
                session = async_get_clientsession(self.hass)
                models = await fetch_models(session, api_key, url)
                voices = await fetch_voices(session, api_key, url)
                # Present model/voice selection
                schema = vol.Schema({
                    vol.Required(CONF_API_KEY, default=api_key): str,
                    vol.Required(CONF_URL, default=url): str,
                    vol.Optional(CONF_SPEED, default=speed): selector({
                        "number": {
                            "min": 0.25,
                            "max": 4.0,
                            "step": 0.05,
                            "mode": "slider"
                        }
                    }),
                    vol.Required(CONF_MODEL): selector({
                        "select": {
                            "options": models,
                            "mode": "dropdown",
                            "sort": True,
                            "custom_value": True
                        }
                    }),
                    vol.Required(CONF_VOICE, default=voices[0] if voices else ""): selector({
                        "select": {
                            "options": voices,
                            "mode": "dropdown",
                            "sort": True,
                            "custom_value": True
                        }
                    })
                })
                return self.async_show_form(
                    step_id="user",
                    data_schema=schema,
                    errors=errors,
                    description_placeholders=user_input
                )
        # Step 1: Ask for API key and URL
        schema = vol.Schema({
            vol.Required(CONF_API_KEY): str,
            vol.Required(CONF_URL, default="https://api.openai.com/v1/audio/speech"): str,
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
            step_id="user",
            data_schema=schema,
            errors=errors,
            description_placeholders=user_input
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        return OpenAITTSOptionsFlow()

class OpenAITTSOptionsFlow(OptionsFlow):
    """Handle options flow for OpenAI TTS."""
    async def async_step_init(self, user_input: dict | None = None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        # Retrieve chime options using the executor to avoid blocking the event loop.
        chime_options = await self.hass.async_add_executor_job(get_chime_options)
        options_schema = vol.Schema({
            # Use constant for chime enable toggle so the label comes from strings.json
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
                default=self.config_entry.options.get(CONF_VOICE, self.config_entry.data.get(CONF_VOICE, "shimmer"))
            ): selector({
                "select": {
                    "options": VOICES,
                    "mode": "dropdown",
                    "sort": True,
                    "custom_value": True
                }
            }),


            vol.Optional(
                 CONF_INSTRUCTIONS,
                 default=self.config_entry.options.get(CONF_INSTRUCTIONS, self.config_entry.data.get(CONF_INSTRUCTIONS, None))
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT,multiline=True)
            ),

            # Normalization toggle using its constant; label will be picked from strings.json.
            vol.Optional(
                CONF_NORMALIZE_AUDIO,
                default=self.config_entry.options.get(CONF_NORMALIZE_AUDIO, self.config_entry.data.get(CONF_NORMALIZE_AUDIO, False))
            ): selector({"boolean": {}})
        })
        return self.async_show_form(step_id="init", data_schema=options_schema)
