"""统一的 LLM 模型构造入口。

放在 `app/utils` 而不是 `app/chat`：调用方横跨 chat / kb / api 三层，而 `app/kb`
已经反向依赖 `app/chat` 十几处（改造方案里的缺陷 C4）。工厂是本轮要新增的公共件，
不应再往那条错误依赖方向上加一条。`app/utils` 是被各方导入的横切层，放这里零新增依赖。

为什么需要它：改造前 9 处各自调用 `init_chat_model`，其中四项公共参数
（`model_provider="openai"` / `api_key.strip()` / `base_url.strip() or None` /
`pinned_llm_client_kwargs`）完全重复，而三项关键参数（temperature、stream_usage、
timeout/max_tokens 等额外入参）各不相同、散落在各处。任何一处被单独修改都会与
其余 8 处产生**不会报错**的漂移，运行时只表现为"某个辅助模型行为怪怪的"。

容易写错的地方：
- `temperature=None` 必须表示「**不传该参数**」，而不是 0。`vision_caption` 依赖
  ChatOpenAI 的默认值 0.7；若工厂无条件传 0，读图行为会静默改变。
- 判空必须写 `is not None`。写成 `if temperature:` 会把 0 当成假值丢掉，而且
  大多数调用点恰好传的就是 0——这个 bug 极难在运行时发现。
- 有 4 处辅助模型（打分/路由/选档/改写）在缺 API Key 时**返回 None 降级**，
  而不是抛异常。用 `chat` 会导致它们从"跳过这一步"变成"整轮失败"，故另给 `try_chat`。
- 这里**不做 SSRF 校验**。校验是配置进入系统时（`_llm_config_from_ua`）的职责，
  放进工厂会让每次构造模型都多一次 DNS 解析。
- 这里**不缓存模型对象**。`init_chat_model` 返回的对象很轻，底层 httpx 连接池
  已由 `app.utils.egress` 按 base_url 复用；加缓存反而要维护
  (model, temperature, stream_usage, api_key, timeout...) 全维度键，漏一个就串配置。
"""

from __future__ import annotations

from typing import Any, Mapping

from langchain.chat_models import init_chat_model

from app.utils.egress import pinned_llm_client_kwargs

# 历史行为：多处配置在 model_name 缺失时回落到 gpt-4，保持一致
_DEFAULT_MODEL = "gpt-4"


class LlmConfigError(ValueError):
    """模型配置不可用（缺 API Key 等）。文案可直接展示给智能体属主。"""


class LlmFactory:
    """构造 OpenAI 兼容的聊天模型。所有 LLM 客户端都应经由此处创建。"""

    @staticmethod
    def chat(
        cfg: Mapping[str, Any] | None,
        *,
        model_name: str | None = None,
        temperature: float | None = 0.0,
        stream_usage: bool | None = False,
        **overrides: Any,
    ) -> Any:
        """构造模型；配置不可用时抛 `LlmConfigError`。

        :param cfg: 含 api_key / base_url / model_name 的映射（现有 llm_config dict 直接传）
        :param model_name: 覆盖 cfg 中的模型名（如 RAG_GRADE_MODEL）
        :param temperature: 默认 0；传 None 表示不传该参数（用 SDK 默认）
        :param stream_usage: 是否让模型回传 token 用量；传 None 表示不传该参数
        :param overrides: 原样透传，如 timeout / max_tokens / max_retries
        """
        model = LlmFactory._build(
            cfg,
            model_name=model_name,
            temperature=temperature,
            stream_usage=stream_usage,
            overrides=overrides,
        )
        if model is None:
            raise LlmConfigError("该智能体未配置可用的模型 API Key，无法调用模型")
        return model

    @staticmethod
    def try_chat(
        cfg: Mapping[str, Any] | None,
        *,
        model_name: str | None = None,
        temperature: float | None = 0.0,
        stream_usage: bool | None = False,
        **overrides: Any,
    ) -> Any:
        """同 `chat`，但配置不可用时返回 None。

        用于「可选辅助模型」：打分、路由、选档、查询改写。这些环节缺 Key 时
        应当跳过该步并降级，而不是让整轮对话失败。
        """
        return LlmFactory._build(
            cfg,
            model_name=model_name,
            temperature=temperature,
            stream_usage=stream_usage,
            overrides=overrides,
        )

    @staticmethod
    def _build(
        cfg: Mapping[str, Any] | None,
        *,
        model_name: str | None,
        temperature: float | None,
        stream_usage: bool | None,
        overrides: Mapping[str, Any],
    ) -> Any:
        conf = cfg or {}
        api_key = str(conf.get("api_key") or "").strip()
        if not api_key:
            return None
        base_url = str(conf.get("base_url") or "").strip() or None

        name = str(model_name or conf.get("model_name") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

        kwargs: dict[str, Any] = {
            "model": name,
            "model_provider": "openai",
            "api_key": api_key,
            "base_url": base_url,
        }
        kwargs.update(overrides)
        # 关键顺序：显式参数必须在 overrides 之后写入，否则会被 overrides 覆盖；
        # 而 pinned client 要最后写，防止业务参数误传 http_client 破坏 DNS 固定。
        # 两个 None 语义都是「不传该参数」，不能退化成传 0 / False：
        # - temperature=None 时 ChatOpenAI 用 0.7（vision 依赖）
        # - stream_usage=None 时由 langchain 自行决定是否上报用量（vision 依赖）
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stream_usage is not None:
            kwargs["stream_usage"] = stream_usage
        kwargs.update(pinned_llm_client_kwargs(base_url))
        return init_chat_model(**kwargs)
