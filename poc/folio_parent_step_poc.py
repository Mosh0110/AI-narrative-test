from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = Path(__file__).with_name("folio_two_model_run_validation_seed42.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("folio_parent_step_poc_seed42")
INPUT_PATH = Path(os.environ.get("FOLIO_PARENT_INPUT_PATH", DEFAULT_INPUT_PATH))
OUTPUT_DIR = Path(os.environ.get("FOLIO_PARENT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
TOGETHER_KEY_PATH = REPO_ROOT / "ai-narrative-test-suite" / "together_api.txt"
DEEPINFRA_KEY_PATH = Path.home() / "Documents/Projects/keys/Deep_infra"
OPENROUTER_KEY_PATH = Path.home() / "Documents/Projects/keys/openrouter_api_key"
AI_NARRATIVE_TEST_SUITE_DIR = REPO_ROOT / "ai-narrative-test-suite"

if str(AI_NARRATIVE_TEST_SUITE_DIR) not in sys.path:
    sys.path.append(str(AI_NARRATIVE_TEST_SUITE_DIR))

from demonstration_for_prompt import demonstration

DEFAULT_TOGETHER_MODEL = "deepseek-ai/DeepSeek-R1-0528"
TOGETHER_MODEL = os.environ.get("FOLIO_TOGETHER_MODEL", DEFAULT_TOGETHER_MODEL)
DEEPINFRA_MODEL = "Qwen/Qwen3-32B"
QWEN_TOKENIZER_NAME = "Qwen/Qwen3-32B"
R1_TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1"
TOGETHER_MODEL_LABEL = "deepseek-ai/DeepSeek-R1-0528"
DEEPINFRA_MODEL_LABEL = "Qwen/Qwen3-32B@DeepInfra"
PARSE_MODEL = "google/gemini-2.5-flash"
SEMANTIC_JUDGE_MODEL = "google/gemini-2.5-flash"

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_COMPLETIONS_URL = "https://api.together.xyz/v1/completions"
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TEMPERATURE = 0.0
TOP_P = 0.0
TOP_K = 1
SEED = 42
MAX_STEPS_PER_SAMPLE = 20
MAX_WORKERS = 4
MAX_SENTENCES_PER_STEP = 3
HTTP_TIMEOUT_SECONDS = 300
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
DEEPINFRA_MAX_RETRIES = 12
DEEPINFRA_MAX_BACKOFF_SECONDS = 60
TOGETHER_MAX_RETRIES = 8

OPENROUTER_SEED = 0
YES_THRESHOLD = 8
NO_THRESHOLD = 2
MAX_FINAL_QUESTIONS = 25
SHOW_ALTERNATIVE_STEP_BOX = True
SELECT_QUESTIONS_RANDOMLY = False
MAX_TARGET_STEP_LIMIT = 19
MAX_DISPLAY_STEP_SENTENCES = 3
MAX_DISPLAY_STEP_CHARS = 300

_THREAD_LOCAL = threading.local()

PARSE_THOUGHTS_PROMPT = f"""You will be provided with a text extracted from a solution process. Your task is to copy the text verbatim and analyze it by categorizing each distinct step into one of the following types:

recalling (retrieving information from memory)

plan (describing a work plan or wondering on the next step or previous steps)

reasoning (a logical step toward an answer)



Instructions:

Copy the text exactly as given, preserving all formatting (including line breaks) and making no edits.
At the end of each step (1 sentence long), append the appropriate type tag.
Each step should end with one of the following tags:
[: recalling], [: plan], [: reasoning].
Important: Your output must include the entire text verbatim.


Example for an output:
{demonstration}

---------

"""

INSTRUCTION_SEMANTIC_NO_QUESTION = """"\

You are given two excerpts, each describing the same step from a solution processes—one is the original, and the other is an alternative version. Your task is to assess how similar these two steps are, particularly considering their meaning and role within the solution process.



Note: The alternative step ends abruptly. Take this into consideration when evaluating the similarity.



Respond with a single number from 1 to 10, where 1 means "completely different" and 10 means "semantically identical." Only reply with the number.

"""


def extract_reasoning(item: dict, provider: str) -> str:
    provider_item = item[provider]
    reasoning = (provider_item.get("reasoning") or "").strip()
    content = (provider_item.get("content") or "").strip()
    if not reasoning and "<think>" in content and "</think>" in content:
        return content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if reasoning:
        return reasoning

    message = provider_item["response"]["choices"][0]["message"]
    reasoning = (message.get("reasoning") or message.get("reasoning_content") or "").strip()
    content = (message.get("content") or "").strip()
    if not reasoning and "<think>" in content and "</think>" in content:
        return content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    return reasoning or content


def tokenizer_name_for_model_slug(model_slug: str) -> str:
    if "Qwen/Qwen3-32B" in model_slug:
        return QWEN_TOKENIZER_NAME
    if "DeepSeek-R1" in model_slug:
        return R1_TOKENIZER_NAME
    raise ValueError(f"Unsupported model slug for tokenizer resolution: {model_slug}")


def model_label_for_slug(provider: str, model_slug: str) -> str:
    tokenizer_name = tokenizer_name_for_model_slug(model_slug)
    if provider == "together" and tokenizer_name == QWEN_TOKENIZER_NAME:
        return "Qwen/Qwen3-32B@Together"
    if provider == "together" and tokenizer_name == R1_TOKENIZER_NAME:
        return model_slug
    if provider == "deepinfra" and tokenizer_name == QWEN_TOKENIZER_NAME:
        return DEEPINFRA_MODEL_LABEL
    raise ValueError(f"Unsupported provider/model combination: provider={provider} model_slug={model_slug}")


def model_metadata_for_item(raw_item: dict, provider: str) -> dict:
    provider_item = raw_item[provider]
    response_model = provider_item.get("response", {}).get("model")
    model_slug = response_model or provider_item.get("model")
    if not model_slug:
        raise ValueError(f"Missing model slug for provider={provider}")
    tokenizer_name = tokenizer_name_for_model_slug(model_slug)
    return {
        "model_slug": model_slug,
        "tokenizer_name": tokenizer_name,
        "model_label": model_label_for_slug(provider, model_slug),
    }


def parse_annotated_text_ocaml(text: str, include_step_markers: bool = False) -> list[dict]:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    if include_step_markers:
        segments = re.split(r"\[step\]", text)
        segments = [seg for seg in segments if seg.strip()]
        parsed_entries = []
        for segment in segments:
            match = re.match(r"(?s)(.*)\[:\s*(plan|reasoning|recalling)\s*\]\s*$", segment)
            if not match:
                raise ValueError(f"Invalid [step] segment: {segment[:400]!r}")
            parsed_entries.append(
                {
                    "text": match.group(1),
                    "type": match.group(2),
                }
            )
        return parsed_entries

    segments = re.split(r"\[:\s*(plan|reasoning|recalling)\s*\]", text)

    parsed_entries = []
    texts = segments[::2]
    annotations = segments[1::2]

    for segment_text, annotation_text in zip(texts, annotations):
        parsed_entries.append(
            {
                "text": segment_text,
                "type": annotation_text,
            }
        )

    return parsed_entries


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    stripped = re.sub(r"(^|\n)\s*\d+\.\s+", r"\1", stripped)
    stripped = re.sub(r"(^|\n)\s*-\s+", r"\1", stripped)
    return len([segment for segment in re.split(r"(?<=[.!?])\s+", stripped) if segment])


def is_displayable_step_text(text: str) -> bool:
    stripped = sanitize_user_text(text).strip()
    if not stripped:
        return False
    if count_sentences(stripped) > MAX_DISPLAY_STEP_SENTENCES:
        return False
    if len(stripped) > MAX_DISPLAY_STEP_CHARS:
        return False
    return True


def validate_nodes_and_types(nodes_and_types: list[dict], request_label: str = "") -> None:
    for idx, entry in enumerate(nodes_and_types, start=1):
        sentence_count = count_sentences(entry["text"])
        if sentence_count > MAX_SENTENCES_PER_STEP:
            prefix = f" | {request_label}" if request_label else ""
            raise ValueError(
                f"Parsed step exceeds {MAX_SENTENCES_PER_STEP} sentences{prefix}: step={idx} "
                f"sentences={sentence_count} text={entry['text'][:400]!r}"
            )


def normalize_reconstruction_text(text: str) -> str:
    normalized = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def canonicalize_alignment_char(char: str) -> str:
    if char.isspace():
        return ""
    if char == "’" or char == "‘":
        return "'"
    if char == "“" or char == "”":
        return '"'
    return char


def canonicalize_for_alignment(text: str) -> str:
    return "".join(canonicalize_alignment_char(char) for char in text)


def realign_entries_to_original_text(entries: list[dict], original_text: str, request_label: str = "") -> list[dict]:
    canonical_original_chars = []
    canonical_to_original_indices = []
    for original_idx, char in enumerate(original_text):
        canonical_char = canonicalize_alignment_char(char)
        if not canonical_char:
            continue
        canonical_original_chars.append(canonical_char)
        canonical_to_original_indices.append(original_idx)

    canonical_original = "".join(canonical_original_chars)
    canonical_cursor = 0
    canonical_spans = []

    for idx, entry in enumerate(entries, start=1):
        canonical_entry = canonicalize_for_alignment(entry["text"])
        if not canonical_entry:
            raise ValueError(f"Empty canonical parsed entry | {request_label} | step={idx}")
        canonical_start = canonical_original.find(canonical_entry, canonical_cursor)
        if canonical_start != canonical_cursor:
            raise ValueError(
                f"Failed to realign parsed entry | {request_label} | step={idx} "
                f"cursor={canonical_cursor} found={canonical_start} text={entry['text'][:400]!r}"
            )
        canonical_end = canonical_start + len(canonical_entry)
        canonical_spans.append((canonical_start, canonical_end))
        canonical_cursor = canonical_end

    if canonical_cursor != len(canonical_original):
        raise ValueError(
            f"Realignment did not consume full text | {request_label} "
            f"cursor={canonical_cursor} total={len(canonical_original)}"
        )

    realigned_entries = []
    raw_start = 0
    for idx, entry in enumerate(entries):
        if idx + 1 < len(entries):
            next_canonical_start = canonical_spans[idx + 1][0]
            raw_end = canonical_to_original_indices[next_canonical_start]
        else:
            raw_end = len(original_text)
        realigned_entries.append(
            {
                "text": original_text[raw_start:raw_end],
                "type": entry["type"],
            }
        )
        raw_start = raw_end

    return realigned_entries


def validate_text_reconstruction(nodes_and_types: list[dict], original_text: str, request_label: str = "") -> None:
    reconstructed = "".join(entry["text"] for entry in nodes_and_types)
    if normalize_reconstruction_text(reconstructed) != normalize_reconstruction_text(original_text):
        prefix = f" | {request_label}" if request_label else ""
        raise ValueError(
            f"Parsed text reconstruction mismatch{prefix}: "
            f"expected={original_text[:400]!r} reconstructed={reconstructed[:400]!r}"
        )


def openrouter_model_call(
    input_text: str,
    api_key: str,
    model: str,
    reasoning_prefix: str | bool = False,
    limit_tokens: int | None = None,
    clean: bool = False,
    request_label: str = "",
) -> dict:
    messages = [{"role": "user", "content": input_text}]
    if reasoning_prefix:
        messages.append({"role": "assistant", "content": "<think>" + str(reasoning_prefix)})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "seed": OPENROUTER_SEED,
        "repetition_penalty": 0,
        "presence_penalty": 0,
        "reasoning": {
            "enabled": False,
            "max_tokens": 0,
        },
    }

    if limit_tokens is not None:
        payload["max_tokens"] = limit_tokens

    session = get_http_session("openrouter")
    for attempt in range(1, 31):
        try:
            response = session.post(
                url=OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            break
        except Exception as e:
            if request_label:
                print(f"OpenRouter retry {attempt}/30 | {request_label} | {e}")
            else:
                print(f"OpenRouter retry {attempt}/30 | {e}")
            time.sleep(1)
            pass
    else:
        raise Exception("OpenRouter API call failed after 30 retries")

    if clean:
        message["content"] = message["content"].lower().strip().replace('"', "")
        if "reasoning" in message and message["reasoning"]:
            message["reasoning"] = message["reasoning"].lower().strip().replace('"', "")

    return message


def parse_reasoning_steps_once(reasoning_text: str, openrouter_key: str, request_label: str = "") -> list[dict]:
    prompt = PARSE_THOUGHTS_PROMPT + "\n" + f"Text extracted from a solution process:\n {reasoning_text}"
    response = openrouter_model_call(
        prompt,
        api_key=openrouter_key,
        model=PARSE_MODEL,
        request_label=request_label,
    )
    nodes_and_types = parse_annotated_text_ocaml(response["content"], include_step_markers=True)
    nodes_and_types = realign_entries_to_original_text(nodes_and_types, reasoning_text, request_label=request_label)
    validate_text_reconstruction(nodes_and_types, reasoning_text, request_label=request_label)
    return nodes_and_types


def recursively_refine_nodes(nodes_and_types: list[dict], openrouter_key: str, request_label: str = "") -> list[dict]:
    refined_entries = []
    for idx, entry in enumerate(nodes_and_types, start=1):
        sentence_count = count_sentences(entry["text"])
        if sentence_count <= MAX_SENTENCES_PER_STEP:
            refined_entries.append(entry)
            continue

        child_request_label = f"{request_label}/step{idx}" if request_label else f"step{idx}"
        reparsed_entries = parse_reasoning_steps_once(entry["text"], openrouter_key, request_label=child_request_label)
        child_max_sentences = max(count_sentences(child["text"]) for child in reparsed_entries)

        if len(reparsed_entries) == 1 and reparsed_entries[0]["text"] == entry["text"]:
            raise ValueError(
                f"Recursive parse made no progress | {child_request_label}: "
                f"sentences={sentence_count} text={entry['text'][:400]!r}"
            )
        if child_max_sentences >= sentence_count and len(reparsed_entries) <= 1:
            raise ValueError(
                f"Recursive parse did not reduce sentence count | {child_request_label}: "
                f"parent={sentence_count} child_max={child_max_sentences} text={entry['text'][:400]!r}"
            )

        refined_entries.extend(
            recursively_refine_nodes(reparsed_entries, openrouter_key, request_label=child_request_label)
        )

    return refined_entries


def get_nodes(
    question: str,
    reasoning_text: str,
    openrouter_key: str,
    tokenizer: object,
    request_label: str = "",
) -> list[dict]:
    reasoning_text_parts = reasoning_text.split("\n")
    reasoning_text_parts_aggregated = []
    reasoning_text_aggregated = ""

    for part in reasoning_text_parts:
        if len(tokenizer.encode(reasoning_text_aggregated)) + len(tokenizer.encode(part)) < 2000:
            reasoning_text_aggregated += part + "\n"
        else:
            reasoning_text_parts_aggregated.append(reasoning_text_aggregated)
            reasoning_text_aggregated = part + "\n"

    reasoning_text_parts_aggregated.append(reasoning_text_aggregated[:-1])

    nodes_and_types_all = []
    for chunk_idx, reasoning_text_part in enumerate(reasoning_text_parts_aggregated):
        prompt = PARSE_THOUGHTS_PROMPT + "\n" + f"Text extracted from a solution process:\n {reasoning_text_part}"
        chunk_label = f"{request_label}/chunk{chunk_idx}" if request_label else f"chunk{chunk_idx}"
        response = openrouter_model_call(
            prompt,
            api_key=openrouter_key,
            model=PARSE_MODEL,
            request_label=chunk_label,
        )
        nodes_and_types = parse_annotated_text_ocaml(response["content"])
        nodes_and_types_all += nodes_and_types

    return [{"text": question, "type": "root"}] + nodes_and_types_all


def similarity_score(original_step: str, alternative_step: str, openrouter_key: str, request_label: str = "") -> int | None:
    prompt = INSTRUCTION_SEMANTIC_NO_QUESTION + "\n\n"
    prompt += "Original step: " + original_step + "\n\n"
    prompt += "Alternative step: " + alternative_step + "\n\n"
    response = openrouter_model_call(
        prompt,
        api_key=openrouter_key,
        model=SEMANTIC_JUDGE_MODEL,
        limit_tokens=5,
        clean=True,
        request_label=request_label,
    )["content"]
    try:
        return int(response[:2])
    except Exception:
        return None


def equivalence_label(score: int | None) -> str:
    if score is None:
        return "unsure"
    if score >= YES_THRESHOLD:
        return "yes"
    if score <= NO_THRESHOLD:
        return "no"
    return "unsure"


def extract_generated_text(response_data: dict, provider: str) -> str:
    message = response_data["choices"][0]["message"]
    if provider == "together":
        return message.get("reasoning") or message.get("content") or ""

    content = message.get("content") or ""
    if "<think>" in content and "</think>" in content:
        return content.split("<think>", 1)[1].split("</think>", 1)[0]
    return content


def get_http_session(name: str) -> requests.Session:
    sessions = getattr(_THREAD_LOCAL, "sessions", None)
    if sessions is None:
        sessions = {}
        _THREAD_LOCAL.sessions = sessions

    if name not in sessions:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=MAX_WORKERS + 4, pool_maxsize=MAX_WORKERS + 4)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        sessions[name] = session

    return sessions[name]


def compute_retry_delay(attempt: int, response: requests.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, min(float(retry_after), DEEPINFRA_MAX_BACKOFF_SECONDS))
            except ValueError:
                pass
    return min(float(2**attempt), float(DEEPINFRA_MAX_BACKOFF_SECONDS))


def is_retriable_http_error(error: requests.HTTPError) -> bool:
    response = error.response
    return response is not None and response.status_code in TRANSIENT_STATUS_CODES


def post_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    session_name: str,
    request_label: str = "",
    max_retries: int = 1,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = get_http_session(session_name).post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            last_error = error
            if attempt == max_retries or not is_retriable_http_error(error):
                raise
            delay = compute_retry_delay(attempt, error.response)
            status_code = error.response.status_code if error.response is not None else "unknown"
            if request_label:
                print(
                    f"{session_name} retry {attempt}/{max_retries} | {request_label} | "
                    f"status={status_code} | sleep={delay:.1f}s"
                )
            time.sleep(delay)
        except (requests.ReadTimeout, requests.ConnectionError) as error:
            last_error = error
            if attempt == max_retries:
                raise
            delay = compute_retry_delay(attempt)
            if request_label:
                print(
                    f"{session_name} retry {attempt}/{max_retries} | {request_label} | "
                    f"{type(error).__name__} | sleep={delay:.1f}s"
                )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"post_json failed without an exception for {session_name}")


def generation_token_ids(tokenizer: object, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def canonicalize_generation_text(tokenizer: object, text: str) -> str:
    return tokenizer.decode(
        generation_token_ids(tokenizer, text),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def truncate_generation_text(tokenizer: object, text: str, max_tokens: int) -> str:
    return tokenizer.decode(
        generation_token_ids(tokenizer, text)[:max_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def count_generation_tokens(tokenizer: object, text: str) -> int:
    return len(generation_token_ids(tokenizer, text))


def together_counterfactual(
    prompt: str,
    prefix_text: str,
    max_tokens: int,
    api_key: str,
    tokenizer: object,
    request_label: str = "",
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "<think>\n" + prefix_text},
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    payload = {
        "model": TOGETHER_MODEL,
        "prompt": input_text,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "repetition_penalty": 0.0,
    }
    data = post_json(
        TOGETHER_COMPLETIONS_URL,
        api_key,
        payload,
        "together",
        request_label=request_label,
        max_retries=TOGETHER_MAX_RETRIES,
    )
    assistant_reply = data["choices"][0]["text"]
    reasoning_part = assistant_reply.split("<think>")[-1]
    reasoning_part = reasoning_part.split("</think>")[0]
    return truncate_generation_text(tokenizer, reasoning_part, max_tokens)


def deepinfra_counterfactual(
    prompt: str,
    prefix_text: str,
    max_tokens: int,
    api_key: str,
    tokenizer: object,
    request_label: str = "",
) -> str:
    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<think>\n" + prefix_text},
        ],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "seed": SEED,
    }
    data = post_json(
        DEEPINFRA_URL,
        api_key,
        payload,
        "deepinfra",
        request_label=request_label,
        max_retries=DEEPINFRA_MAX_RETRIES,
    )
    return truncate_generation_text(tokenizer, extract_generated_text(data, "deepinfra"), max_tokens)


def build_sample_bank(raw_items: list[dict], openrouter_key: str, providers: list[str] | None = None) -> tuple[list[dict], dict]:
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_NAME)
    r1_tokenizer = AutoTokenizer.from_pretrained(R1_TOKENIZER_NAME)
    tokenizers_by_name = {
        QWEN_TOKENIZER_NAME: qwen_tokenizer,
        R1_TOKENIZER_NAME: r1_tokenizer,
    }
    selected_providers = providers or ["together", "deepinfra"]

    samples = []
    samples_by_model = defaultdict(dict)
    for raw_item in raw_items:
        for provider in selected_providers:
            if provider not in raw_item:
                continue
            model_metadata = model_metadata_for_item(raw_item, provider)
            model_name = model_metadata["model_label"]
            tokenizer = tokenizers_by_name[model_metadata["tokenizer_name"]]
            request_label = (
                f"parse provider={provider} story_id={raw_item['story_id']} example_id={raw_item['example_id']}"
            )
            print(request_label)
            solution_steps = get_nodes(
                raw_item["prompt"],
                extract_reasoning(raw_item, provider),
                openrouter_key,
                tokenizer,
                request_label=request_label,
            )
            evaluation_steps = solution_steps[:MAX_STEPS_PER_SAMPLE]

            sample_idx = len(samples)
            sample = {
                "sample_idx": sample_idx,
                "provider": provider,
                "model_name": model_name,
                "model_slug": model_metadata["model_slug"],
                "story_key": raw_item["story_key"],
                "story_id": raw_item["story_id"],
                "example_id": raw_item["example_id"],
                "label": raw_item["label"],
                "main_question": raw_item["prompt"],
                "premises": raw_item["premises"],
                "conclusion": raw_item["conclusion"],
                "solution_steps": solution_steps,
                "evaluation_steps": evaluation_steps,
                "tokenizer_name": model_metadata["tokenizer_name"],
            }
            samples.append(sample)
            samples_by_model[model_name][str(sample_idx)] = sample
    return samples, samples_by_model


def counterfactual_task(
    sample: dict,
    omitted_idx: int,
    target_idx: int,
    together_key: str,
    deepinfra_key: str,
    openrouter_key: str,
    tokenizers: dict[str, object],
) -> dict:
    steps = [step["text"] for step in sample["evaluation_steps"]]
    prompt = steps[0]
    prefix_steps = steps[:omitted_idx] + steps[omitted_idx + 1:target_idx]
    target_step = steps[target_idx]
    tokenizer = tokenizers[sample["tokenizer_name"]]
    max_tokens = count_generation_tokens(tokenizer, target_step)
    request_label = (
        f"counterfactual sample_idx={sample['sample_idx']} provider={sample['provider']} "
        f"story_id={sample['story_id']} example_id={sample['example_id']} "
        f"omitted_idx={omitted_idx} target_idx={target_idx}"
    )

    if sample["provider"] == "together":
        alternative_step = together_counterfactual(
            prompt,
            "".join(prefix_steps),
            max_tokens,
            together_key,
            tokenizer,
            request_label=request_label,
        )
    else:
        alternative_step = deepinfra_counterfactual(
            prompt,
            "".join(prefix_steps),
            max_tokens,
            deepinfra_key,
            tokenizer,
            request_label=request_label,
        )

    score = similarity_score(
        target_step,
        alternative_step,
        openrouter_key,
        request_label=(
            f"judge sample_idx={sample['sample_idx']} provider={sample['provider']} "
            f"story_id={sample['story_id']} example_id={sample['example_id']} "
            f"omitted_idx={omitted_idx} target_idx={target_idx}"
        ),
    )
    return {
        "omitted_step_index": omitted_idx,
        "target_step_index": target_idx,
        "original_step": target_step,
        "alternative_step": alternative_step,
        "similarity_score": score,
        "semantic_similarity_response": score,
        "equivalent": equivalence_label(score),
    }


def evaluate_sample_links(sample: dict, together_key: str, deepinfra_key: str, openrouter_key: str, tokenizers: dict[str, object]) -> list[dict]:
    total_steps = len(sample["evaluation_steps"])
    tasks = []
    for omitted_idx in range(1, total_steps):
        for target_idx in range(omitted_idx + 2, total_steps):
            if target_idx < 4:
                continue
            tasks.append((omitted_idx, target_idx))

    print(
        f"evaluate_links sample_idx={sample['sample_idx']} provider={sample['provider']} "
        f"story_id={sample['story_id']} example_id={sample['example_id']} tasks={len(tasks)}"
    )
    links = [None] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(
                counterfactual_task,
                sample,
                omitted_idx,
                target_idx,
                together_key,
                deepinfra_key,
                openrouter_key,
                tokenizers,
            ): task_index
            for task_index, (omitted_idx, target_idx) in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            task_index = future_to_index[future]
            links[task_index] = future.result()
    print(
        f"finished_links sample_idx={sample['sample_idx']} provider={sample['provider']} "
        f"story_id={sample['story_id']} example_id={sample['example_id']}"
    )
    return links


def generate_questions(samples: list[dict]) -> list[dict]:
    sample_indexes = [sample["sample_idx"] for sample in samples]
    interleaved_samples = []
    samples_by_idx = {sample["sample_idx"]: sample for sample in samples}
    all_sample_indexes = sorted(set(sample_indexes))

    for sample in samples_by_idx.values():
        sample["model_name"] = sample["model_name"]

    for sample_idx in all_sample_indexes:
        if sample_idx in samples_by_idx:
            interleaved_samples.append(samples_by_idx[sample_idx])

    sample_data_list = interleaved_samples
    all_generated_candidates = []
    position_counts_for_generation_phase = [0, 0, 0, 0]

    for sample_info in sample_data_list:
        sample_idx = sample_info["sample_idx"]
        links_for_sample = sample_info["links"]
        model_name = sample_info["model_name"]
        solution_steps_for_sample = sample_info.get("solution_steps", [])
        if not isinstance(links_for_sample, list) or not isinstance(solution_steps_for_sample, list):
            continue

        target_to_links_for_sample = defaultdict(list)
        for link in links_for_sample:
            target_idx = link["target_step_index"]
            target_to_links_for_sample[target_idx].append(link)

        for target_idx_original, links_for_this_target in target_to_links_for_sample.items():
            if MAX_TARGET_STEP_LIMIT is not None and target_idx_original > MAX_TARGET_STEP_LIMIT:
                continue

            crucial_steps_with_alt = []
            non_crucial_steps_yes = []
            for link in links_for_this_target:
                prior_step_idx = link["omitted_step_index"]
                if prior_step_idx >= len(solution_steps_for_sample):
                    continue
                prior_step_text = solution_steps_for_sample[prior_step_idx]["text"]
                step_info = {"step_index": prior_step_idx, "text": prior_step_text, "equivalent": link["equivalent"]}
                if link["equivalent"] == "no":
                    step_info["counterfactual_text"] = link.get("alternative_step", "")
                    crucial_steps_with_alt.append(step_info)
                elif link["equivalent"] == "yes":
                    non_crucial_steps_yes.append(step_info)

            if not crucial_steps_with_alt:
                continue
            if len(non_crucial_steps_yes) < 3:
                continue

            correct_choice_candidate = None
            selected_incorrect_choices = []

            if not SELECT_QUESTIONS_RANDOMLY:
                possible_configurations_for_this_target = []

                for current_potential_correct_choice in crucial_steps_with_alt:
                    non_crucial_before = sorted(
                        [s for s in non_crucial_steps_yes if s["step_index"] < current_potential_correct_choice["step_index"]],
                        key=lambda x: x["step_index"],
                    )
                    non_crucial_after = sorted(
                        [s for s in non_crucial_steps_yes if s["step_index"] > current_potential_correct_choice["step_index"]],
                        key=lambda x: x["step_index"],
                    )

                    feasible_strategies_for_this_candidate = []
                    if len(non_crucial_after) >= 3:
                        feasible_strategies_for_this_candidate.append({"pos": 0, "before_needed": 0, "after_needed": 3})
                    if len(non_crucial_before) >= 1 and len(non_crucial_after) >= 2:
                        feasible_strategies_for_this_candidate.append({"pos": 1, "before_needed": 1, "after_needed": 2})
                    if len(non_crucial_before) >= 2 and len(non_crucial_after) >= 1:
                        feasible_strategies_for_this_candidate.append({"pos": 2, "before_needed": 2, "after_needed": 1})
                    if len(non_crucial_before) >= 3:
                        feasible_strategies_for_this_candidate.append({"pos": 3, "before_needed": 3, "after_needed": 0})

                    for strategy in feasible_strategies_for_this_candidate:
                        score = position_counts_for_generation_phase[strategy["pos"]]
                        possible_configurations_for_this_target.append(
                            {
                                "score": score,
                                "correct_choice_candidate": current_potential_correct_choice,
                                "strategy": strategy,
                                "non_crucial_before": non_crucial_before,
                                "non_crucial_after": non_crucial_after,
                            }
                        )

                if possible_configurations_for_this_target:
                    possible_configurations_for_this_target.sort(key=lambda x: (x["score"], x["strategy"]["pos"]))
                    best_overall_configuration = possible_configurations_for_this_target[0]

                    correct_choice_candidate = best_overall_configuration["correct_choice_candidate"]
                    best_strategy = best_overall_configuration["strategy"]
                    non_crucial_before_for_best = best_overall_configuration["non_crucial_before"]
                    non_crucial_after_for_best = best_overall_configuration["non_crucial_after"]

                    temp_selected_incorrect = []
                    if best_strategy["before_needed"] > 0:
                        temp_selected_incorrect.extend(random.sample(non_crucial_before_for_best, best_strategy["before_needed"]))
                    if best_strategy["after_needed"] > 0:
                        temp_selected_incorrect.extend(random.sample(non_crucial_after_for_best, best_strategy["after_needed"]))

                    if len(temp_selected_incorrect) == 3:
                        selected_incorrect_choices = temp_selected_incorrect

                if len(selected_incorrect_choices) != 3:
                    if not correct_choice_candidate:
                        if crucial_steps_with_alt:
                            correct_choice_candidate = random.choice(crucial_steps_with_alt)
                        else:
                            continue

                    selected_incorrect_choices = []
                    if len(non_crucial_steps_yes) >= 3:
                        selected_incorrect_choices = random.sample(non_crucial_steps_yes, 3)

            else:
                if crucial_steps_with_alt:
                    correct_choice_candidate = random.choice(crucial_steps_with_alt)
                else:
                    continue
                if len(non_crucial_steps_yes) >= 3:
                    selected_incorrect_choices = random.sample(non_crucial_steps_yes, 3)

            if not correct_choice_candidate or len(selected_incorrect_choices) != 3:
                continue

            correct_choice_data = {
                "step_index": correct_choice_candidate["step_index"],
                "text": correct_choice_candidate["text"],
                "is_crucial": True,
            }
            incorrect_choices_data = [
                {"step_index": npc["step_index"], "text": npc["text"], "is_crucial": False}
                for npc in selected_incorrect_choices
            ]
            choices_data = [correct_choice_data] + incorrect_choices_data
            choices_data.sort(key=lambda x: x["step_index"])

            if target_idx_original >= len(solution_steps_for_sample):
                continue
            target_text = solution_steps_for_sample[target_idx_original]["text"]

            final_choices_for_q = []
            answer_indices_for_q = []
            for i, choice_d in enumerate(choices_data):
                final_choices_for_q.append(
                    {
                        "step_index": choice_d["step_index"],
                        "text": choice_d["text"],
                        "is_crucial": choice_d["is_crucial"],
                    }
                )
                if choice_d["is_crucial"]:
                    answer_indices_for_q.append(i)

            achieved_html_pos = next((i for i, choice in enumerate(final_choices_for_q) if choice["is_crucial"]), None)
            display_texts = [target_text] + [choice["text"] for choice in final_choices_for_q]
            if not all(is_displayable_step_text(text) for text in display_texts):
                continue

            question_data_dict = {
                "question_number": 0,
                "target_step_index": target_idx_original,
                "target_step_text": target_text,
                "counterfactual_starting_text": correct_choice_candidate["counterfactual_text"],
                "choices": final_choices_for_q,
                "answer_indices": [achieved_html_pos],
                "sample_idx": sample_idx,
                "achieved_html_pos": achieved_html_pos,
                "model_name": model_name,
                "story_key": sample_info["story_key"],
                "example_id": sample_info["example_id"],
                "gold_label": sample_info["label"],
            }
            all_generated_candidates.append(
                {
                    "question_data": question_data_dict,
                    "achieved_html_pos": achieved_html_pos,
                    "sample_idx": sample_idx,
                    "target_idx_original": target_idx_original,
                }
            )
            if achieved_html_pos is not None and 0 <= achieved_html_pos < len(position_counts_for_generation_phase):
                position_counts_for_generation_phase[achieved_html_pos] += 1

    if not all_generated_candidates:
        raise ValueError("No valid question candidates generated. Check the input data and links format.")

    final_selected_question_data_list = []
    final_html_position_counts = [0, 0, 0, 0]
    questions_selected_per_model = defaultdict(int)
    questions_selected_per_sample = defaultdict(int)
    remaining_candidates = list(all_generated_candidates)

    while remaining_candidates:
        if MAX_FINAL_QUESTIONS is not None and len(final_selected_question_data_list) >= MAX_FINAL_QUESTIONS:
            break
        if not remaining_candidates:
            break

        scored_candidates = []
        for idx, candidate_info in enumerate(remaining_candidates):
            q_data = candidate_info["question_data"]
            achieved_pos = candidate_info["achieved_html_pos"]
            if achieved_pos is None or not (0 <= achieved_pos < len(final_html_position_counts)):
                continue

            score_html = final_html_position_counts[achieved_pos]
            score_model = questions_selected_per_model[q_data["model_name"]]
            score_sample = questions_selected_per_sample[q_data["sample_idx"]]
            scored_candidates.append((score_html, score_model, score_sample, idx, candidate_info))

        if not scored_candidates:
            break

        scored_candidates.sort(key=lambda x: (x[1], x[0], x[2], x[3]))
        best_scored_candidate_tuple = scored_candidates[0]
        original_idx_in_remaining = best_scored_candidate_tuple[3]
        best_candidate_info = best_scored_candidate_tuple[4]

        final_selected_question_data_list.append(best_candidate_info["question_data"])
        final_html_position_counts[best_candidate_info["achieved_html_pos"]] += 1
        questions_selected_per_model[best_candidate_info["question_data"]["model_name"]] += 1
        questions_selected_per_sample[best_candidate_info["question_data"]["sample_idx"]] += 1
        remaining_candidates.pop(original_idx_in_remaining)

    for i, q_data in enumerate(final_selected_question_data_list):
        q_data["question_number"] = i + 1

    return final_selected_question_data_list


def format_test(samples_by_model: dict, questions: list[dict]) -> dict:
    instruction_text = (
        "You are presented with a problem and a chain of thought generated by an AI when solving it. For each question, the chain of thought is shown up to a 'target step' (highlighted in yellow).\n\n"
        "<strong>Your task:</strong> From the four options provided, select the step that is crucial for the target step. All other steps being equal, the crucial step absence would cause the AI to generate a significantly different target step. \n\n"
        "Only one of the four options is crucial. The other three, if absent, would not significantly change the target step."
    )
    if SHOW_ALTERNATIVE_STEP_BOX:
        instruction_text += (
            "\n\nA snippet of text below the target step shows how the target step in yellow will begin if the crucial step was indeed omitted. Use this as a hint."
        )

    display_samples_by_model = sanitize_samples_by_model_for_display(samples_by_model)

    test_format = {
        "title": "Step Dependency Questionnaire",
        "instructions": instruction_text,
        "samples": display_samples_by_model,
        "questions": [],
    }

    for q in questions:
        sample_idx = q["sample_idx"]
        model_name = q["model_name"]
        main_question = display_samples_by_model[model_name][str(sample_idx)]["main_question"]

        formatted_q = {
            "question_number": q["question_number"],
            "sample_idx": sample_idx,
            "main_question": main_question,
            "target_step_index": q["target_step_index"],
            "target_step_text": sanitize_user_text(q["target_step_text"]),
            "counterfactual_starting_text": (
                sanitize_user_text(q.get("counterfactual_starting_text", "")) if SHOW_ALTERNATIVE_STEP_BOX else ""
            ),
            "choices": [
                {
                    "choice_letter": chr(65 + i),
                    "step_index": c["step_index"],
                    "text": sanitize_user_text(c["text"]),
                }
                for i, c in enumerate(q["choices"])
            ],
            "correct_answers": [chr(65 + i) for i in q["answer_indices"]],
            "model_name": q["model_name"],
            "story_key": q["story_key"],
            "example_id": q["example_id"],
            "gold_label": q["gold_label"],
        }
        test_format["questions"].append(formatted_q)

    return test_format


DISPLAY_SYMBOL_REPLACEMENTS = [
    ("⊕", " or "),
    ("∨", " or "),
    ("∧", " and "),
    ("→", " implies "),
    ("↔", " if and only if "),
    ("¬", "not "),
    ("∀", "for all "),
    ("∃", "there exists "),
]


def sanitize_user_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    sanitized = text
    for source, replacement in DISPLAY_SYMBOL_REPLACEMENTS:
        sanitized = sanitized.replace(source, replacement)

    sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")
    sanitized = re.sub(r" {2,}", " ", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized


def format_main_question_for_display(main_question: str) -> str:
    text = sanitize_user_text(main_question).strip()
    if not text.startswith("Premises:\n") or "\n\nConclusion:\n" not in text:
        return text

    premises_block, remainder = text.split("\n\nConclusion:\n", 1)
    conclusion_block, suffix = remainder.split("\n\nDetermine whether the conclusion is entailed by the premises.", 1)

    premise_lines = [line.strip() for line in premises_block.splitlines()[1:] if line.strip()]
    numbered_premises = "\n".join(f"{idx}. {premise}" for idx, premise in enumerate(premise_lines, start=1))

    formatted = (
        "Premises:\n\n"
        f"{numbered_premises}\n\n"
        "Conclusion:\n\n"
        f"{conclusion_block.strip()}\n\n"
        "Determine whether the conclusion is entailed by the premises."
    )
    if suffix.strip():
        formatted += "\n" + suffix.strip()
    return formatted


def sanitize_samples_by_model_for_display(samples_by_model: dict) -> dict:
    display_samples_by_model = copy.deepcopy(samples_by_model)
    for model_samples in display_samples_by_model.values():
        for sample in model_samples.values():
            sample["main_question"] = format_main_question_for_display(sample["main_question"])
            for key in ["solution_steps", "evaluation_steps"]:
                if key not in sample:
                    continue
                for step in sample[key]:
                    if isinstance(step, dict) and "text" in step:
                        step["text"] = sanitize_user_text(step["text"])
    return display_samples_by_model


def build_summary(samples: list[dict], questions: list[dict]) -> dict:
    per_sample = []
    for sample in samples:
        link_counts = defaultdict(int)
        for link in sample["links"]:
            link_counts[link["equivalent"]] += 1
        target_counts = defaultdict(lambda: {"yes": 0, "no": 0, "unsure": 0})
        for link in sample["links"]:
            target_counts[link["target_step_index"]][link["equivalent"]] += 1
        viable_targets = [
            target_idx
            for target_idx, counts in target_counts.items()
            if counts["no"] >= 1 and counts["yes"] >= 3
        ]
        per_sample.append(
            {
                "sample_idx": sample["sample_idx"],
                "model_name": sample["model_name"],
                "story_key": sample["story_key"],
                "example_id": sample["example_id"],
                "gold_label": sample["label"],
                "num_solution_steps": len(sample["solution_steps"]) - 1,
                "num_parsed_steps": len(sample["solution_steps"]) - 1,
                "num_evaluation_steps": len(sample["evaluation_steps"]) - 1,
                "link_counts": dict(link_counts),
                "viable_target_indices": viable_targets,
                "num_viable_targets": len(viable_targets),
            }
        )

    return {
        "settings": {
            "seed": SEED,
            "temperature": TEMPERATURE,
            "together_top_p": TOP_P,
            "top_k": TOP_K,
            "max_steps_per_sample": MAX_STEPS_PER_SAMPLE,
            "yes_threshold": YES_THRESHOLD,
            "no_threshold": NO_THRESHOLD,
            "max_final_questions": MAX_FINAL_QUESTIONS,
            "show_alternative_step_box": SHOW_ALTERNATIVE_STEP_BOX,
            "select_questions_randomly": SELECT_QUESTIONS_RANDOMLY,
            "max_target_step_limit": MAX_TARGET_STEP_LIMIT,
            "max_display_step_sentences": MAX_DISPLAY_STEP_SENTENCES,
            "max_display_step_chars": MAX_DISPLAY_STEP_CHARS,
        },
        "num_samples": len(samples),
        "num_questions": len(questions),
        "per_sample": per_sample,
    }


def write_preview(formatted_test: dict) -> None:
    lines = []
    for question in formatted_test["questions"]:
        lines.append(f"Q{question['question_number']} | sample={question['sample_idx']} | model={question['model_name']}")
        lines.append(f"story_key={question['story_key']} | example_id={question['example_id']} | gold_label={question['gold_label']}")
        lines.append(f"target_step_index={question['target_step_index']}")
        lines.append(f"target_step_text={question['target_step_text']}")
        lines.append(f"counterfactual_starting_text={question['counterfactual_starting_text']}")
        for choice in question["choices"]:
            lines.append(f"  {choice['choice_letter']}. step {choice['step_index']}: {choice['text']}")
        lines.append(f"correct={','.join(question['correct_answers'])}")
        lines.append("")
    (OUTPUT_DIR / "question_preview.txt").write_text("\n".join(lines))


def load_original_html_generator():
    notebook_path = AI_NARRATIVE_TEST_SUITE_DIR / "2_test_generator.ipynb.ipynb"
    notebook = json.loads(notebook_path.read_text())
    cell_source = None
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def generate_html_test" in source:
            cell_source = source
            break
    cell_source = cell_source.replace("He\\Him", "He\\\\Him")
    cell_source = cell_source.replace("She\\Her", "She\\\\Her")
    cell_source = cell_source.replace("Other\\prefer not to say", "Other\\\\prefer not to say")
    namespace = {}
    exec(cell_source, namespace)
    return namespace["generate_html_test"], namespace


def render_test_html(formatted_test: dict, prolific_mode: bool = True) -> str:
    return render_test_html_with_options(formatted_test, prolific_mode=prolific_mode)


def render_test_html_with_options(
    formatted_test: dict,
    prolific_mode: bool = True,
    show_alternative_step_box: bool | None = None,
) -> str:
    generate_html_test, namespace = load_original_html_generator()
    namespace["samples_by_model"] = {
        model_name: {int(sample_idx): sample for sample_idx, sample in samples.items()}
        for model_name, samples in formatted_test["samples"].items()
    }
    namespace["attention_check_indices"] = formatted_test.get("attention_check_indices", [])
    show_alt_box = SHOW_ALTERNATIVE_STEP_BOX if show_alternative_step_box is None else show_alternative_step_box

    html_test = copy.deepcopy(formatted_test)
    if not show_alt_box:
        hint_text = (
            "\n\nA snippet of text below the target step shows how the target step in yellow will begin "
            "if the crucial step was indeed omitted. Use this as a hint."
        )
        html_test["instructions"] = html_test["instructions"].replace(hint_text, "")
        for question in html_test["questions"]:
            question["counterfactual_starting_text"] = ""

    html = generate_html_test(html_test, show_alt_box, prolific_mode=prolific_mode)
    submission_title_line = "                testTitle: document.title,\n"
    assert submission_title_line in html
    replacement = "                testTitle: " f"{json.dumps(html_test['title'])},\n"
    if html_test.get("test_id"):
        replacement += "                testId: " f"{json.dumps(html_test['test_id'])},\n"
    html = html.replace(submission_title_line, replacement, 1)
    if "<meta charset=\"UTF-8\">" not in html and "<meta charset=\"utf-8\">" not in html:
        html = html.replace("<head>", "<head>\n    <meta charset=\"UTF-8\">", 1)
    if ".main-problem p {" not in html:
        html = html.replace(
            ".main-problem { background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
            ".main-problem { background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }\n        "
            ".main-problem p { white-space: pre-line; line-height: 1.55; margin: 0; }",
            1,
        )
    return html


def write_test_html(formatted_test: dict, output_path: Path, prolific_mode: bool = True) -> None:
    output_path.write_text(render_test_html(formatted_test, prolific_mode=prolific_mode))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_items = json.loads(INPUT_PATH.read_text())
    together_key = TOGETHER_KEY_PATH.read_text().strip()
    deepinfra_key = DEEPINFRA_KEY_PATH.read_text().strip()
    openrouter_key = OPENROUTER_KEY_PATH.read_text().strip()
    tokenizers = {
        R1_TOKENIZER_NAME: AutoTokenizer.from_pretrained(R1_TOKENIZER_NAME),
        QWEN_TOKENIZER_NAME: AutoTokenizer.from_pretrained(QWEN_TOKENIZER_NAME),
    }

    samples, samples_by_model = build_sample_bank(raw_items, openrouter_key)
    for sample in samples:
        sample["links"] = evaluate_sample_links(sample, together_key, deepinfra_key, openrouter_key, tokenizers)

    questions = generate_questions(samples)
    formatted_test = format_test(samples_by_model, questions)
    summary = build_summary(samples, questions)

    (OUTPUT_DIR / "sample_bank.json").write_text(json.dumps(samples, indent=2))
    (OUTPUT_DIR / "samples_by_model.json").write_text(json.dumps(samples_by_model, indent=2))
    (OUTPUT_DIR / "question_candidates.json").write_text(json.dumps(questions, indent=2))
    (OUTPUT_DIR / "parent_step_detection_test.json").write_text(json.dumps(formatted_test, indent=2))
    (OUTPUT_DIR / "viability_summary.json").write_text(json.dumps(summary, indent=2))
    write_preview(formatted_test)

    print("Viability summary")
    print("-----------------")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Saved artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
