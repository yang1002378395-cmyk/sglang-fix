import logging
import os


def load_prompts_from_file(
    path: str, logger: logging.Logger | None = None
) -> list[str]:
    logger = logger or logging.getLogger(__name__)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt text file not found: {path}")

    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in file: {path}")

    logger.info("Found %d prompts in %s", len(prompts), path)
    return prompts


def resolve_prompts(
    prompt: str | list[str] | None,
    prompt_path: str | None = None,
    legacy_prompt_path: str | None = None,
    logger: logging.Logger | None = None,
) -> list[str]:
    logger = logger or logging.getLogger(__name__)
    selected_prompt_path = prompt_path

    if selected_prompt_path is None:
        selected_prompt_path = legacy_prompt_path
    elif legacy_prompt_path is not None and os.path.abspath(
        legacy_prompt_path
    ) != os.path.abspath(selected_prompt_path):
        logger.warning(
            "Both prompt_path=%s and prompt_file_path=%s were provided; "
            "using prompt_path.",
            selected_prompt_path,
            legacy_prompt_path,
        )

    if selected_prompt_path is not None:
        return load_prompts_from_file(selected_prompt_path, logger=logger)

    if prompt is None:
        return [" "]
    if isinstance(prompt, str):
        return [prompt]
    return list(prompt)
