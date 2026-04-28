"""LLM 클라이언트 유틸리티 — AsyncOpenAI 기반 스트리밍·구조화 응답."""
from util.genos_utils import genos_import
genos_import('openai')

import asyncio
import json
import os
import re
import openai
from typing import Any, AsyncIterator, Optional, Type, TypeVar, Union

from httpx import Timeout
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from resource.config import (
    DEFAULT_MODEL,
    ETF_AGENT_RUNTIME,
    GENOS_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    SERVING_API_KEY,
    VLM_ID
)
from resource.utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# OpenAI 클라이언트 설정 — 내부망/외부망 자동 분기
# ---------------------------------------------------------------------------
_is_external = ETF_AGENT_RUNTIME.lower() == "external_genos"

# if GENOS_BASE_URL and SERVING_API_KEY:
if GENOS_BASE_URL and SERVING_API_KEY:
    print(f"[llm.py] 내부망: GenOS 서빙 엔드포인트")
    # 내부망: GenOS 서빙 엔드포인트
    CLIENT = AsyncOpenAI(
        base_url=f"{GENOS_BASE_URL}/api/gateway/rep/serving/{VLM_ID}/v1",
        api_key=SERVING_API_KEY,
        timeout=Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    )
# else:
#     print(f"[llm.py] 외부망: OpenRouter")
#     # 외부망: OpenRouter
#     CLIENT = AsyncOpenAI(
#         base_url=OPENROUTER_BASE_URL,
#         api_key=OPENROUTER_API_KEY,
#         timeout=Timeout(connect=10.0, read=45.0, write=10.0, pool=10.0),
#     )


# ---------------------------------------------------------------------------
# SSE 이벤트 모델
# ---------------------------------------------------------------------------
class SSE(BaseModel):
    """서비스 레이어 → main.py 간 스트리밍 이벤트."""

    event: str  # "token" | "reasoning_token" | "tool_call" | "charts" 등
    data: Any


# ---------------------------------------------------------------------------
# 모델 후보 (Fallback)
# ---------------------------------------------------------------------------
def _get_model_candidates(model: str) -> list[str]:
    """환경 변수 OPENROUTER_MODEL_FALLBACKS 에서 후보 모델 목록을 구성한다."""
    raw = (
        os.getenv("OPENAI_MODEL_FALLBACKS")
        or os.getenv("OPENROUTER_MODEL_FALLBACKS")
        or ""
    )
    fallbacks = [m.strip() for m in raw.split(",") if m.strip()]
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(m: str) -> None:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)

    _add(model)
    for m in fallbacks:
        _add(m)
    return ordered


# ---------------------------------------------------------------------------
# reasoning 누출 정리 (모델 공통)
# ---------------------------------------------------------------------------
_REASONING_END_MARKERS = ["assistantfinal", "assistant final", "finalanswer"]
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _clean_reasoning_leak(text: str) -> str:
    """모델이 content에 출력한 reasoning 프리픽스를 제거한다."""
    lower = text.lower()
    for marker in _REASONING_END_MARKERS:
        idx = lower.find(marker)
        if idx != -1:
            cleaned = text[idx + len(marker):].strip()
            if cleaned:
                return cleaned
    return text


# ---------------------------------------------------------------------------
# vLLM XML tool_call 파싱 — content에 <tool_call> 태그로 출력되는 경우 처리
# ---------------------------------------------------------------------------
_XML_TOOL_CALL_BLOCK_RE = re.compile(
    r'<tool_call>([\s\S]*?)</tool_call>', re.IGNORECASE,
)
_XML_FUNC_TAG_RE = re.compile(
    r'<function\s*=?\s*["\']?(\w+)["\']?\s*>([\s\S]*?)(?:</function>|$)',
)


def _parse_xml_arguments(raw: str) -> dict:
    """XML <parameter> 태그 또는 JSON에서 인자를 추출한다."""
    # 1차: JSON 시도
    try:
        cleaned = re.sub(r'</?(?:parameter|parameters|arguments)[^>]*>', '', raw)
        cleaned = cleaned.strip()
        if cleaned.startswith('{'):
            return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    # 2차: 개별 <parameter=key>value</parameter> 파싱
    params: dict = {}
    for pm in re.finditer(r'<parameter\s*=\s*["\']?(\w+)["\']?\s*>([\s\S]*?)</parameter>', raw):
        key, val = pm.group(1), pm.group(2).strip()
        if val.lower() in ('true', 'false'):
            params[key] = val.lower() == 'true'
        elif val.isdigit():
            params[key] = int(val)
        else:
            try:
                params[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                params[key] = val
    return params


def _parse_xml_tool_calls(text: str) -> list[dict]:
    """vLLM이 content에 넣은 <tool_call> XML을 OpenAI tool_call dict 리스트로 변환한다."""
    results: list[dict] = []
    for block_match in _XML_TOOL_CALL_BLOCK_RE.finditer(text):
        inner = block_match.group(1).strip()
        func_name = ""
        args_dict: dict = {}

        # 형태 A: JSON body — {"name": "func", "arguments": {...}}
        if inner.startswith("{"):
            try:
                obj = json.loads(inner)
                func_name = obj.get("name", "")
                raw_args = obj.get("arguments", obj.get("parameters", {}))
                args_dict = raw_args if isinstance(raw_args, dict) else {}
            except (json.JSONDecodeError, TypeError):
                pass
        # 형태 B: XML — <function=name><parameter=k>v</parameter></function>
        if not func_name:
            fm = _XML_FUNC_TAG_RE.search(inner)
            if fm:
                func_name = fm.group(1)
                args_dict = _parse_xml_arguments(fm.group(2))

        if func_name:
            results.append({
                "id": f"xmlcall_{len(results)}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args_dict, ensure_ascii=False),
                },
            })
    return results


def _strip_xml_tool_calls(text: str) -> str:
    """텍스트에서 <tool_call> 블록 및 잔여 tool call 관련 태그를 모두 제거한다."""
    text = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', text, flags=re.IGNORECASE)
    # 닫히지 않은 잔여 태그 제거
    text = re.sub(r'</?tool_call[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<function[^>]*>[\s\S]*?(?:</function>|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(?:function|parameter)[^>]*>', '', text, flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------------------------
# 스트리밍 LLM 호출
# ---------------------------------------------------------------------------
async def call_llm_stream(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> AsyncIterator[Union[SSE, dict]]:
    """LLM 스트리밍 호출.

    yield:
        - SSE(event="token", data=str)           — 텍스트 토큰
        - SSE(event="reasoning_token", data=str)  — 추론 토큰
        - dict (assistant message)                — 스트림 종료 시 최종 메시지
    """
    if "max_tokens" not in kwargs:
        from resource.config import RESERVED_OUTPUT_TOKENS
        kwargs["max_tokens"] = RESERVED_OUTPUT_TOKENS

    # Docker 환경 등에서 유입되는 서로게이트 문자 제거 (UTF-8 인코딩 오류 방지)
    for msg in messages:
        if isinstance(msg.get("content"), str):
            msg["content"] = _SURROGATE_RE.sub("", msg["content"])

    # OpenRouter 외부망: reasoning_effort → extra_body.reasoning 객체로 변환
    if not _is_external and "reasoning_effort" in kwargs:
        effort = kwargs.pop("reasoning_effort")
        extra = kwargs.pop("extra_body", {})
        extra["reasoning"] = {"effort": effort}
        kwargs["extra_body"] = extra

    # Qwen 모델: thinking 모드 비활성화 + 샘플링 파라미터 최적화
    if "qwen" in model.lower():
        extra = kwargs.get("extra_body", {})
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        extra["top_k"] = 20          # Qwen 공식 권장값 (non-thinking)
        kwargs["extra_body"] = extra
        # Qwen non-thinking 권장: temperature=0.7, top_p=0.8
        kwargs.setdefault("temperature", 0.7)
        kwargs.setdefault("top_p", 0.8)
        # 반복 생성 방지 (1~100 넘버링 등 장황한 출력 억제)
        kwargs.setdefault("presence_penalty", 1.5)

    last_err: Optional[Exception] = None
    models = _get_model_candidates(model)

    for attempt in range(3):
        for m in models:
            try:
                response = await CLIENT.chat.completions.create(
                    messages=messages,
                    model=m,
                    stream=True,
                    **kwargs,
                )

                full_content_parts: list[str] = []
                full_reasoning_parts: list[str] = []
                tool_call_buf: dict[int, dict] = {}
                _tool_call_signaled: bool = False
                _leak_detecting: bool = True
                _leak_buffer: str = ""
                _post_leak_scanning: bool = False
                _post_leak_buffer: str = ""

                _chunk_timeout = 90  # 청크 간 최대 대기(초) — 스트림 stall 방지
                _chunk_iter = response.__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(_chunk_iter.__anext__(), timeout=_chunk_timeout)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        logger.error("[llm_stream] %d초 동안 청크 미수신 → 스트림 중단 (model=%s)", _chunk_timeout, m)
                        raise TimeoutError(f"LLM 스트림 청크 타임아웃 ({_chunk_timeout}s)")
                    try:
                        choice = chunk.choices[0]
                        delta = choice.delta
                    except Exception:
                        continue

                    # 추론(reasoning) 토큰 — reasoning_content 또는 reasoning 키 모두 대응
                    # (gpt-oss-120b: reasoning_content, Qwen 3.5: reasoning)
                    rc = getattr(delta, "reasoning_content", None) or ""
                    if not rc:
                        # OpenAI SDK가 비표준 'reasoning' 필드를 attribute로
                        # 노출하지 않을 수 있음 → raw 데이터에서 직접 추출
                        _raw_c0 = getattr(chunk, "choices", [None])[0] if getattr(chunk, "choices", None) else None
                        _raw_delta: Any = None
                        if _raw_c0 is not None:
                            if hasattr(_raw_c0, "model_extra") and isinstance(getattr(_raw_c0, "model_extra", None), dict):
                                _raw_delta = _raw_c0.model_extra.get("delta")
                            if _raw_delta is None and hasattr(_raw_c0, "__dict__"):
                                _d = _raw_c0.__dict__.get("delta")
                                if isinstance(_d, dict):
                                    _raw_delta = _d
                        if isinstance(_raw_delta, dict):
                            rc = _raw_delta.get("reasoning", "") or ""
                    if not rc:
                        # fallback: getattr 체인 (기존 호환)
                        r_str = getattr(delta, "reasoning", None)
                        if r_str:
                            if isinstance(r_str, str):
                                rc = r_str
                            elif isinstance(r_str, dict):
                                rc = r_str.get("text", "") or r_str.get("content", "")
                            elif hasattr(r_str, "text"):
                                rc = getattr(r_str, "text", "")
                            else:
                                rc = str(r_str) if r_str else ""
                    # reasoning_details (OpenRouter 호환)
                    if not rc:
                        rd = getattr(delta, "reasoning_details", None)
                        if rd and isinstance(rd, list):
                            for item in rd:
                                rt = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                                if rt:
                                    full_reasoning_parts.append(rt)
                                    yield SSE(event="reasoning_token", data=rt)
                    if rc:
                        full_reasoning_parts.append(rc)
                        yield SSE(event="reasoning_token", data=rc)

                    # 도구 호출 델타
                    if getattr(delta, "tool_calls", None):
                        if not _tool_call_signaled:
                            _tool_call_signaled = True
                            yield SSE(event="_tool_call_detected", data=True)
                        for i, tc in enumerate(delta.tool_calls):
                            key = getattr(tc, "index", i)
                            buf = tool_call_buf.setdefault(
                                key,
                                {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": None, "arguments": ""},
                                },
                            )
                            if getattr(tc, "id", None):
                                buf["id"] = tc.id
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                if getattr(fn, "name", None):
                                    buf["function"]["name"] = fn.name
                                if getattr(fn, "arguments", None):
                                    buf["function"]["arguments"] += (
                                        fn.arguments or ""
                                    )

                    # 텍스트 토큰 (reasoning leak 감지 포함)
                    if not getattr(delta, "tool_calls", None) and getattr(
                        delta, "content", None
                    ):
                        piece = delta.content or ""
                        if piece:
                            full_content_parts.append(piece)
                            if _leak_detecting:
                                _leak_buffer += piece
                                lower_buf = _leak_buffer.lower().lstrip()
                                if len(_leak_buffer) <= 20:
                                    if not any(lower_buf.startswith(p) for p in
                                               ("analysis", "thinking", "results")):
                                        _leak_detecting = False
                                        yield SSE(event="token", data=_leak_buffer)
                                else:
                                    found_end = False
                                    for marker in _REASONING_END_MARKERS:
                                        idx = lower_buf.find(marker)
                                        if idx != -1:
                                            actual = _leak_buffer[idx + len(marker):]
                                            yield SSE(event="reasoning_token",
                                                      data=_leak_buffer[:idx])
                                            _leak_detecting = False
                                            if actual.strip():
                                                yield SSE(event="token", data=actual)
                                            found_end = True
                                            break
                                    if not found_end and len(_leak_buffer) > 5000:
                                        yield SSE(event="reasoning_token",
                                                  data=_leak_buffer)
                                        _leak_detecting = False
                                        _post_leak_scanning = True
                            elif _post_leak_scanning:
                                # 버퍼 초과 후에도 assistantfinal 마커 탐색 지속
                                _post_leak_buffer += piece
                                lower_plb = _post_leak_buffer.lower()
                                found_marker = False
                                for marker in _REASONING_END_MARKERS:
                                    idx = lower_plb.find(marker)
                                    if idx != -1:
                                        before = _post_leak_buffer[:idx]
                                        after = _post_leak_buffer[idx + len(marker):]
                                        if before.strip():
                                            yield SSE(event="reasoning_token", data=before)
                                        _post_leak_scanning = False
                                        if after.strip():
                                            yield SSE(event="token", data=after)
                                        found_marker = True
                                        break
                                if not found_marker and len(_post_leak_buffer) > 500:
                                    yield SSE(event="token", data=_post_leak_buffer)
                                    _post_leak_scanning = False
                            else:
                                yield SSE(event="token", data=piece)

                # leak 버퍼가 남아있으면 방출
                if _leak_detecting and _leak_buffer:
                    yield SSE(event="reasoning_token", data=_leak_buffer)

                # post-leak 스캔 버퍼가 남아있으면 마커 최종 확인 후 방출
                if _post_leak_scanning and _post_leak_buffer:
                    lower_plb = _post_leak_buffer.lower()
                    flushed = False
                    for marker in _REASONING_END_MARKERS:
                        idx = lower_plb.find(marker)
                        if idx != -1:
                            before = _post_leak_buffer[:idx]
                            after = _post_leak_buffer[idx + len(marker):]
                            if before.strip():
                                yield SSE(event="reasoning_token", data=before)
                            if after.strip():
                                yield SSE(event="token", data=after)
                            flushed = True
                            break
                    if not flushed:
                        yield SSE(event="token", data=_post_leak_buffer)

                # 최종 메시지 구성
                final_message: dict[str, Any] = {"role": "assistant"}
                final_content = "".join(full_content_parts).strip()
                final_content = _clean_reasoning_leak(final_content)

                # vLLM이 XML 태그로 tool_call을 content에 넣은 경우 파싱 + 항상 스트리핑
                if "<tool_call>" in final_content or "<function" in final_content:
                    if not tool_call_buf:
                        parsed_tool_calls = _parse_xml_tool_calls(final_content)
                        if parsed_tool_calls:
                            logger.info("[llm] vLLM XML tool_call %d개 감지 → 파싱 완료", len(parsed_tool_calls))
                            tool_call_buf = {i: tc for i, tc in enumerate(parsed_tool_calls)}
                    final_content = _strip_xml_tool_calls(final_content)

                final_message["content"] = final_content or ""

                final_reasoning = "".join(full_reasoning_parts).strip()
                if final_reasoning:
                    final_message["reasoning"] = final_reasoning

                if tool_call_buf:
                    tool_calls = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for _, tc in sorted(
                            tool_call_buf.items(), key=lambda x: x[0]
                        )
                    ]
                    final_message["tool_calls"] = tool_calls

                # reasoning-only 응답도 유효하게 처리 (content/tool_calls 없어도)
                # → _run_agent_loop에서 reasoning-only 감지 후 답변 재생성
                has_output = (
                    final_message["content"]
                    or final_message.get("tool_calls")
                    or final_message.get("reasoning")
                )
                if has_output:
                    if m != model:
                        logger.info("fallback model 사용: %s", m)
                    yield final_message
                    return

            except Exception as e:
                last_err = e
                logger.warning("스트림 오류 model=%s attempt=%d: %s", m, attempt + 1, e)
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

    if last_err:
        raise last_err
    yield {"role": "assistant", "content": "LLM 응답을 받지 못했습니다."}


# ---------------------------------------------------------------------------
# 비스트리밍 LLM 호출
# ---------------------------------------------------------------------------
async def call_llm(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> str:
    """비스트리밍 LLM 호출 — 응답 텍스트만 반환."""
    if "max_tokens" not in kwargs:
        from resource.config import RESERVED_OUTPUT_TOKENS
        kwargs["max_tokens"] = RESERVED_OUTPUT_TOKENS

    # Qwen 모델: thinking 모드 비활성화 + 샘플링 파라미터 최적화
    if "qwen" in model.lower():
        extra = kwargs.get("extra_body", {})
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        extra["top_k"] = 20
        kwargs["extra_body"] = extra
        kwargs.setdefault("temperature", 0.7)
        kwargs.setdefault("top_p", 0.8)
        kwargs.setdefault("presence_penalty", 1.5)

    last_err: Optional[Exception] = None
    models = _get_model_candidates(model)

    for attempt in range(3):
        for m in models:
            try:
                resp = await CLIENT.chat.completions.create(
                    messages=messages, model=m, **kwargs
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                logger.warning("LLM 오류 model=%s attempt=%d: %s", m, attempt + 1, e)
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

    if last_err:
        raise last_err
    raise RuntimeError("LLM call failed without exception")


# ---------------------------------------------------------------------------
# 영-한 번역 헬퍼 (visible_rationale 보안용)
# ---------------------------------------------------------------------------
# 번역 전용 모델 — DEFAULT_MODEL 통일
_TRANSLATE_MODEL: str = DEFAULT_MODEL


async def translate_to_korean(text: str) -> str:
    """RAW reasoning을 핵심만 간결하게 한국어로 요약한다.

    전문 번역 대신 **핵심 요약**을 수행하여 LLM 호출 시간을 최소화한다.
    어떤 계획을 세웠는지, 어떤 행동을 하겠는지만 1~3문장으로 요약한다.

    Args:
        text: LLM이 생성한 RAW reasoning 텍스트

    Returns:
        한국어로 요약되고 보안 처리된 텍스트
    """
    # 긴 reasoning은 앞+뒤만 전달하여 입력 토큰 절약
    _MAX_INPUT_CHARS = 3000
    if len(text) > _MAX_INPUT_CHARS:
        half = _MAX_INPUT_CHARS // 2
        text = text[:half] + "\n...(중략)...\n" + text[-half:]

    messages = [
        {
            "role": "system",
            "content": (
                "당신은 요약가입니다. <text> 태그 안의 텍스트는 ETF AI의 사고 과정입니다.\n"
                "핵심만 한국어 1~3문장으로 간결하게 요약하세요.\n\n"
                "## 요약 대상 (이것만 포함)\n"
                "- 어떤 계획을 세웠는지 (예: '반도체 테마 ETF를 검색하겠습니다')\n"
                "- 어떤 데이터를 조회하는지 (예: '국내/미국 시장 데이터를 조회합니다')\n"
                "- 판단/결론 (예: '연금 가능한 ETF 중심으로 안내하겠습니다')\n\n"
                "## 보안 — 반드시 제거\n"
                "SQL문, 테이블명, 도구명, 컬럼명, 파라미터명 등 기술 구현 세부사항은 절대 포함하지 마세요.\n"
                "'데이터를 조회', '검색 수행' 등 자연어로 대체하세요.\n\n"
                "## 출력 규칙\n"
                "- 1~3문장, 최대 100자 이내\n"
                "- ETF명, 티커는 원문 유지\n"
                "- 요약 결과만 출력"
            ),
        },
        {"role": "user", "content": f"<text>{text}</text>"},
    ]
    return await call_llm(
        messages, model=_TRANSLATE_MODEL, max_tokens=512, temperature=0.1,
    )


# ---------------------------------------------------------------------------
# JSON 추출 헬퍼
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Optional[str]:
    """모델이 JSON-only 규칙을 어겼을 때, 가장 그럴듯한 JSON object/array를 추출한다."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()

    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(s[i:])
            return s[i : i + end].strip()
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# 구조화 응답 생성 (Pydantic 모델 바인딩)
# ---------------------------------------------------------------------------
async def generate_structured_response(
    *,
    messages: list[dict[str, Any]],
    response_format: Type[T],
    temperature: float = 0.0,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 4096,
    max_retries: int = 2,
) -> T:
    """LLM에게 messages를 보내고 JSON 응답을 받아 Pydantic 모델로 파싱해 반환한다."""
    schema = response_format.model_json_schema()
    cur_messages = list(messages)
    last_err: Optional[Exception] = None

    # Qwen 모델: thinking 모드 비활성화 + 샘플링 파라미터 최적화
    extra_body: dict[str, Any] = {}
    _temperature = temperature
    _top_p: Optional[float] = None
    _presence_penalty: Optional[float] = None
    if "qwen" in model.lower():
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        extra_body["top_k"] = 20
        if _temperature == 0.0:
            _temperature = 0.7  # Qwen non-thinking 권장값
        _top_p = 0.8
        _presence_penalty = 1.5

    for attempt in range(max_retries + 1):
        try:
            _api_kwargs: dict[str, Any] = {}
            if _top_p is not None:
                _api_kwargs["top_p"] = _top_p
            if _presence_penalty is not None:
                _api_kwargs["presence_penalty"] = _presence_penalty
            resp = await CLIENT.chat.completions.create(
                model=model,
                messages=cur_messages,
                temperature=_temperature,
                max_tokens=max_output_tokens,
                **_api_kwargs,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                },
                **({"extra_body": extra_body} if extra_body else {}),
            )
            choice = resp.choices[0]
            msg = choice.message
            content = (msg.content or "").strip()

            # Qwen 등 reasoning 모델이 content에 사고 과정을 함께 출력하는 경우
            # → reasoning 부분 제거하고 JSON만 추출
            if content:
                content = _clean_reasoning_leak(content)

            # 출력 토큰 부족으로 잘린 경우 즉시 재시도 (잘린 JSON 파싱 무의미)
            if getattr(choice, "finish_reason", None) == "length":
                logger.warning(
                    "[structured] 출력이 max_tokens(%d)에 의해 잘림 (attempt=%d)",
                    max_output_tokens, attempt,
                )
                if attempt < max_retries:
                    cur_messages = [
                        *cur_messages,
                        {
                            "role": "user",
                            "content": (
                                "출력이 토큰 제한으로 잘렸습니다. "
                                "사고 과정 없이 순수 JSON만 간결하게 출력하세요."
                            ),
                        },
                    ]
                    last_err = RuntimeError(
                        f"Output truncated at max_tokens={max_output_tokens}"
                    )
                    continue

            # reasoning 모델이 content 없이 reasoning에만 응답한 경우 fallback
            if not content:
                rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                if rc and not isinstance(rc, str):
                    rc = str(rc)
                if rc:
                    extracted_from_rc = _extract_json(rc)
                    if extracted_from_rc:
                        content = extracted_from_rc
                        logger.info(
                            "[structured] reasoning_content에서 JSON 추출 (attempt=%d)",
                            attempt,
                        )

            if not content:
                raise RuntimeError("LLM returned empty content")

            try:
                obj = json.loads(content)
                # LLM이 배열을 반환한 경우 → 첫 번째 dict 요소로 복구 시도
                if isinstance(obj, list):
                    if obj and isinstance(obj[0], dict):
                        logger.info(
                            "[structured] 배열 응답 → 첫 번째 요소로 복구 (attempt=%d)",
                            attempt,
                        )
                        return response_format.model_validate(obj[0])
                    raise RuntimeError(
                        f"LLM returned array instead of object: {content[:300]}"
                    )
                return response_format.model_validate(obj)
            except (json.JSONDecodeError, RuntimeError):
                raise
            except Exception:
                pass

            extracted = _extract_json(content)
            if extracted:
                try:
                    obj = json.loads(extracted)
                    if isinstance(obj, list):
                        if obj and isinstance(obj[0], dict):
                            return response_format.model_validate(obj[0])
                        raise RuntimeError(
                            f"LLM returned array instead of object: {extracted[:300]}"
                        )
                    return response_format.model_validate(obj)
                except (json.JSONDecodeError, RuntimeError):
                    raise
                except Exception:
                    pass

            raise RuntimeError(
                f"Could not parse JSON from LLM content: {content[:300]}"
            )

        except (json.JSONDecodeError, ValidationError, RuntimeError) as e:
            last_err = e
            if attempt < max_retries:
                logger.warning(
                    "[structured] attempt=%d 실패: %s → 재시도",
                    attempt, repr(last_err),
                )
                cur_messages = [
                    *cur_messages,
                    {
                        "role": "user",
                        "content": (
                            "방금 출력이 비어있거나 JSON 스키마를 만족하지 않았습니다. "
                            "반드시 content에 JSON만 출력하세요. "
                            "사고 과정/설명/주석/코드펜스 없이 순수 JSON만 출력하세요. "
                            "정확히 스키마에 맞춰 다시 출력하세요."
                        ),
                    },
                ]
                continue
            break
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(f"generate_structured_response failed: {repr(last_err)}")


# ---------------------------------------------------------------------------
# GenOS agentFlow 페이로드 생성
# ---------------------------------------------------------------------------
def make_agentflow_payload(node_id: str, reasoning_content: str) -> dict[str, Any]:
    """GenOS UI에 표시할 agentFlowExecutedData 항목을 생성한다."""
    return {
        "nodeId": node_id,
        "nodeLabel": "Visible Reasoner",
        "data": {
            "output": {
                "content": json.dumps(
                    {"visible_rationale": reasoning_content},
                    ensure_ascii=False,
                    indent=2,
                )
            },
        },
        "status": "FINISHED",
    }
