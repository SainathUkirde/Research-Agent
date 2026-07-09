"""
watsonx_client.py
-----------------
Thin wrapper around the IBM watsonx.ai Python SDK (ibm-watsonx-ai).
All NLG tasks in the Research Agent go through this module.
Credentials are loaded from environment variables set by python-dotenv.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Lazy SDK imports (graceful failure if SDK not installed) ──────────────────
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_SDK_AVAILABLE = True
except ImportError:
    WATSONX_SDK_AVAILABLE = False
    logger.warning(
        "ibm-watsonx-ai SDK not installed.  "
        "Install with: pip install ibm-watsonx-ai"
    )


def _get_credentials() -> dict:
    """Read and validate IBM watsonx.ai credentials from the environment."""
    api_key = os.environ.get("WATSONX_API_KEY", "").strip()
    project_id = os.environ.get("WATSONX_PROJECT_ID", "").strip()
    url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()

    if not api_key:
        raise EnvironmentError(
            "WATSONX_API_KEY is not set.  "
            "Copy .env.example → .env and fill in your credentials."
        )
    if not project_id:
        raise EnvironmentError(
            "WATSONX_PROJECT_ID is not set.  "
            "Copy .env.example → .env and fill in your credentials."
        )
    return {"api_key": api_key, "project_id": project_id, "url": url}


@lru_cache(maxsize=1)
def _get_api_client() -> "APIClient":
    """Return a cached watsonx.ai API client (one per process)."""
    creds_dict = _get_credentials()
    credentials = Credentials(
        url=creds_dict["url"],
        api_key=creds_dict["api_key"],
    )
    return APIClient(credentials=credentials)


def get_model(model_id: str, parameters: dict) -> "ModelInference":
    """
    Return a ModelInference instance for the requested Granite model.

    Parameters
    ----------
    model_id : str
        e.g. "ibm/granite-3-3-8b-instruct"
    parameters : dict
        Generation parameters matching GenTextParamsMetaNames keys.
    """
    if not WATSONX_SDK_AVAILABLE:
        raise RuntimeError("ibm-watsonx-ai SDK is required but not installed.")

    creds_dict = _get_credentials()
    client = _get_api_client()

    return ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=Credentials(
            url=creds_dict["url"],
            api_key=creds_dict["api_key"],
        ),
        project_id=creds_dict["project_id"],
    )


def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_id: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> str:
    """
    Generate text from a Granite model on IBM watsonx.ai.

    Parameters
    ----------
    prompt : str
        The user-facing instruction / query.
    system_prompt : str, optional
        Prepended system instructions (from build_system_prompt()).
    model_id : str, optional
        Override the model from GRANITE_CONFIG.
    parameters : dict, optional
        Override generation parameters from GRANITE_CONFIG.

    Returns
    -------
    str
        Generated text from Granite.
    """
    from agent_config import GRANITE_CONFIG

    _model_id = model_id or GRANITE_CONFIG["model_id"]
    _params = parameters or GRANITE_CONFIG["parameters"]

    # Build the full prompt text
    if system_prompt:
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}\n\n[ASSISTANT]\n"
    else:
        full_prompt = prompt

    logger.debug("Sending prompt to Granite (%s), length=%d chars", _model_id, len(full_prompt))

    try:
        model = get_model(_model_id, _params)
        response = model.generate_text(prompt=full_prompt)

        # SDK may return a dict or a string depending on version
        if isinstance(response, dict):
            text = response.get("results", [{}])[0].get("generated_text", "")
        else:
            text = str(response)

        logger.debug("Granite response length: %d chars", len(text))
        return text.strip()

    except Exception as exc:
        logger.error("Granite generation failed: %s", exc)
        raise RuntimeError(f"Granite generation error: {exc}") from exc


def generate_chat(
    messages: list[dict],
    system_prompt: Optional[str] = None,
    model_id: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> str:
    """
    Multi-turn chat wrapper.  Converts message history into a flat prompt
    and calls generate_text().

    Parameters
    ----------
    messages : list[dict]
        List of {"role": "user"|"assistant", "content": str} dicts.
    """
    conversation = ""
    for msg in messages:
        role = msg.get("role", "user").upper()
        conversation += f"[{role}]\n{msg['content']}\n\n"
    conversation += "[ASSISTANT]\n"

    return generate_text(
        prompt=conversation,
        system_prompt=system_prompt,
        model_id=model_id,
        parameters=parameters,
    )


def health_check() -> dict:
    """
    Quick connectivity check — returns {"status": "ok"} or {"status": "error", "detail": "..."}.
    """
    try:
        _get_credentials()
        if not WATSONX_SDK_AVAILABLE:
            return {"status": "error", "detail": "ibm-watsonx-ai SDK not installed"}
        # Attempt a very short generation to confirm the key works
        result = generate_text(
            prompt="Reply with the single word: OK",
            parameters={
                "decoding_method": "greedy",
                "max_new_tokens": 5,
                "min_new_tokens": 1,
            },
        )
        return {"status": "ok", "sample": result}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
