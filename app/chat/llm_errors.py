"""LLM 调用异常分类：把 provider 的原始报错收敛成有限几类，并给出可操作的中文文案。

为什么需要它：主链路此前直接把 `str(exc)` 塞进 SSE 的 error 事件与落库的 error_text，
用户看到的是 `Error code: 429 - {'error': {'message': 'Rate limit reached...'}}` 这类原始报文，
既看不懂也无从操作。大厂 agent 的做法是把异常分成有限几类，各自给「下一步该做什么」，
并标注是否值得重试——重试由 SDK 层（`LLM_MAX_RETRIES`）负责，语义由本模块负责。

容易写错的地方：
- `GraphRecursionError` 是**正常收敛**而不是故障（agent 工具往返步数打满上限）。
  文案不能说「系统出错了」，否则用户会以为平台坏了；要说明「已保留已生成内容」。
- 401/403 是配置错误（Key 失效/无该模型权限），重试没有意义，必须直接告诉属主去改配置。
- 上下文超长要引导「新建会话」，不能让用户对着同一句话反复重试。
- 判定优先用异常类型与 status_code，字符串匹配只作兜底；各 provider 的报文措辞并不统一。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 错误类别：外部按需判断，不要在这之外再扩字符串常量
KIND_RATE_LIMIT = "rate_limit"
KIND_AUTH = "auth"
KIND_CONTEXT_LENGTH = "context_length"
KIND_TIMEOUT = "timeout"
KIND_CONNECTION = "connection"
KIND_RECURSION_LIMIT = "recursion_limit"
KIND_CONTENT_FILTER = "content_filter"
KIND_SERVER = "server"
KIND_UNKNOWN = "unknown"

# 值得自动重试的类别；其余重试只会放大等待
_RETRYABLE = frozenset({KIND_RATE_LIMIT, KIND_TIMEOUT, KIND_CONNECTION, KIND_SERVER})

# 上下文超长的典型报文碎片（各家措辞不同，宁多列几条）
_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "context window",
    "too many tokens",
    "reduce the length",
    "prompt is too long",
    "input is too long",
    "exceed",
)
_RATE_MARKERS = ("rate limit", "rate_limit", "too many requests", "quota exceeded", "tpm", "rpm")
_AUTH_MARKERS = ("invalid api key", "invalid_api_key", "unauthorized", "authentication", "permission denied", "no access")
_TIMEOUT_MARKERS = ("timeout", "timed out", "read timed out", "deadline exceeded")
_CONNECTION_MARKERS = ("connection error", "connection reset", "connection refused", "name or service not known", "getaddrinfo", "ssl", "network is unreachable")
_FILTER_MARKERS = ("content filter", "content_filter", "content policy", "safety", "responsible ai")


@dataclass(frozen=True, slots=True)
class LLMErrorInfo:
    """一次 LLM 失败的分类结果。"""

    kind: str
    user_message: str
    # 是否属于「换个时间点重试可能成功」的瞬时故障（供上层决定是否提示重试）
    retryable: bool
    # 原始异常摘要，仅供服务端日志与 error_text 落库，不直接展示给用户
    detail: str


def _status_code(exc: BaseException) -> int | None:
    """尽最大努力取 HTTP 状态码：openai SDK、httpx、langchain 封装各挂在不同属性上。"""
    for attr in ("status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    # openai.APIStatusError.status_code 之外，部分封装把状态塞在 response 上
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _haystack(exc: BaseException) -> str:
    """拼接用于兜底匹配的小写文本：消息 + 类名 + 有限深度的 cause 链。"""
    parts: list[str] = []
    cur: BaseException | None = exc
    depth = 0
    while cur is not None and depth < 4:
        parts.append(str(cur))
        parts.append(type(cur).__name__)
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return " | ".join(parts).lower()


def _is_recursion_limit(exc: BaseException, text: str) -> bool:
    """递归上限既可能是 GraphRecursionError，也可能是被包装后的同义错误。"""
    try:
        from langgraph.errors import GraphRecursionError

        if isinstance(exc, GraphRecursionError):
            return True
    except Exception:
        # langgraph 缺失时退化为文本匹配，不因导入问题影响错误处理本身
        pass
    return "recursion limit" in text or "graphrecursionerror" in text


def classify_llm_error(exc: BaseException) -> LLMErrorInfo:
    """把异常归入有限类别；未识别的一律 unknown，并保留原始摘要。"""
    text = _haystack(exc)
    status = _status_code(exc)
    detail = f"{type(exc).__name__}: {exc}"[:500]

    # 顺序有讲究：递归上限必须先判，它在文本里可能同时命中别的关键词
    if _is_recursion_limit(exc, text):
        return LLMErrorInfo(
            kind=KIND_RECURSION_LIMIT,
            user_message=(
                "已达到单轮最大推理步数（工具调用往返过多），本轮提前结束。"
                "已生成的内容已保留。可以把问题拆小，或明确指出要查哪个文档/方向后重试。"
            ),
            retryable=False,
            detail=detail,
        )

    if status in (401, 403) or any(m in text for m in _AUTH_MARKERS):
        return LLMErrorInfo(
            kind=KIND_AUTH,
            user_message=(
                "模型服务鉴权失败：请检查该智能体的 API Key 是否有效、是否有该模型的调用权限，"
                "以及接口地址是否与该 Key 匹配。"
            ),
            retryable=False,
            detail=detail,
        )

    if status == 429 or any(m in text for m in _RATE_MARKERS):
        return LLMErrorInfo(
            kind=KIND_RATE_LIMIT,
            user_message="模型服务当前限流（请求过多），请稍后重试。频繁出现可降低提问频率或更换模型。",
            retryable=True,
            detail=detail,
        )

    if any(m in text for m in _CONTEXT_MARKERS):
        return LLMErrorInfo(
            kind=KIND_CONTEXT_LENGTH,
            user_message=(
                "本轮内容超出模型上下文上限，未能完成。建议新建会话重新提问，"
                "或减少附件、缩小知识库检索范围后重试。"
            ),
            retryable=False,
            detail=detail,
        )

    if any(m in text for m in _FILTER_MARKERS):
        return LLMErrorInfo(
            kind=KIND_CONTENT_FILTER,
            user_message="模型侧内容安全策略拦截了本次请求，请调整提问或所引用内容后重试。",
            retryable=False,
            detail=detail,
        )

    if isinstance(exc, TimeoutError) or any(m in text for m in _TIMEOUT_MARKERS):
        return LLMErrorInfo(
            kind=KIND_TIMEOUT,
            user_message="模型响应超时，请稍后重试；若持续超时，可换用响应更快的模型。",
            retryable=True,
            detail=detail,
        )

    if any(m in text for m in _CONNECTION_MARKERS):
        return LLMErrorInfo(
            kind=KIND_CONNECTION,
            user_message="无法连接模型服务，请检查网络连通性与该智能体配置的接口地址是否可达。",
            retryable=True,
            detail=detail,
        )

    if status is not None and 500 <= status < 600:
        return LLMErrorInfo(
            kind=KIND_SERVER,
            user_message="模型服务暂时不可用（服务端错误），请稍后重试。",
            retryable=True,
            detail=detail,
        )

    return LLMErrorInfo(
        kind=KIND_UNKNOWN,
        user_message="生成失败，请稍后重试；若持续失败，请检查该智能体的模型配置。",
        retryable=False,
        detail=detail,
    )


def friendly_error_message(exc: BaseException | str | None) -> str:
    """给用户看的一句话。传入字符串时按文本归类（兼容只有 error_text 的旧调用点）。"""
    if exc is None:
        return ""
    if isinstance(exc, str):
        # 已是中文且非空：视为上层已处理过，避免二次包装
        if exc.strip() and not exc.isascii():
            return exc.strip()
        err: BaseException = RuntimeError(exc)
    else:
        err = exc
    return classify_llm_error(err).user_message


def error_payload(exc: BaseException) -> dict[str, Any]:
    """构造 SSE error 事件载荷：用户文案 + 机器可读类别 + 是否可重试。

    detail 不放进来：原始报文可能含 base_url、Key 片段等敏感信息，只落服务端日志。
    """
    info = classify_llm_error(exc)
    return {
        "content": info.user_message,
        "kind": info.kind,
        "retryable": info.retryable,
    }
