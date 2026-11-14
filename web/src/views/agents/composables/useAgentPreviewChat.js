import { ref, unref } from 'vue'
import { getToken, ensureFreshToken, tryRefreshToken } from '@/utils'
import {
  clearPreviewPendingJob,
  readPreviewChatJobSseStream,
  readPreviewPendingJob,
  savePreviewPendingJob,
} from '@/views/agents/composables/previewChatSse.js'

const baseApi = import.meta.env.VITE_BASE_API || '/api/v1'

/**
 * 编辑器右侧预览试聊（独立 session，不写入 recent、不刷新侧栏）。
 */
export function useAgentPreviewChat({
  agentId,
  sessionId,
  useKnowledgeRetrieval,
  useWebSearch,
  t,
}) {
  const messages = ref([])
  const sessionPhase = ref('intro')
  const sending = ref(false)
  const streamAbortController = ref(null)
  const activeJobId = ref(null)
  const activeAssistantIdx = ref(-1)
  const streamStoppedByUser = ref(false)

  function resetChat() {
    stopGeneration()
    messages.value = []
    sessionPhase.value = 'intro'
  }

  async function stopGeneration() {
    if (!sending.value) return
    streamStoppedByUser.value = true
    const jid = activeJobId.value
    const ix = activeAssistantIdx.value
    const token = getToken()
    const aid = unref(agentId)
    const sid = unref(sessionId)
    if (jid && token) {
      try {
        await fetch(`${baseApi}/user-agent/chat/jobs/${jid}/cancel`, {
          method: 'POST',
          credentials: 'include',
          headers: { token, 'Content-Type': 'application/json' },
        })
      } catch {
        /* ignore */
      }
    } else if (token && aid && sid) {
      // job_id 未知（创建请求在途被中断）：按会话兜底取消活动任务，避免孤儿任务阻塞后续对话
      try {
        await fetch(
          `${baseApi}/user-agent/chat/active_job/cancel?agent_id=${aid}&session_id=${encodeURIComponent(
            sid
          )}`,
          {
            method: 'POST',
            credentials: 'include',
            headers: { token, 'Content-Type': 'application/json' },
          }
        )
      } catch {
        /* ignore */
      }
    }
    streamAbortController.value?.abort()
    streamAbortController.value = null
    activeJobId.value = null
    activeAssistantIdx.value = -1
    if (aid && sid) clearPreviewPendingJob(aid, sid)
    if (ix >= 0 && messages.value[ix]?.role === 'assistant') {
      const row = messages.value[ix]
      messages.value[ix] = {
        ...row,
        pending: false,
        stoppedByUser: true,
        errorText: undefined,
      }
    }
    sending.value = false
  }

  async function postJobAndConsumeStream({ regenerate, message, attachmentIds, assistantIdx }) {
    const aid = unref(agentId)
    const sid = unref(sessionId)
    // 预览试聊也是原生 fetch，页面停留久了 access token 会静默过期，先换新鲜的
    let token = await ensureFreshToken()
    if (!aid || !token) throw new Error('missing agent or token')

    let jobId
    let startSeq = 0
    streamStoppedByUser.value = false
    const ac = new AbortController()
    streamAbortController.value = ac
    activeAssistantIdx.value = assistantIdx
    const idx = assistantIdx

    try {
      const postJob = () =>
        fetch(`${baseApi}/user-agent/chat/jobs`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json', token },
          body: JSON.stringify({
            agent_id: aid,
            message,
            session_id: sid,
            use_knowledge_retrieval: unref(useKnowledgeRetrieval),
            use_web_search: unref(useWebSearch),
            attachment_ids: attachmentIds,
            regenerate,
          }),
          signal: ac.signal,
        })

      let postRes = await postJob()
      if (postRes.status === 401) {
        const fresh = await tryRefreshToken()
        if (fresh) {
          token = fresh
          postRes = await postJob()
        }
      }
      if (postRes.status === 409) {
        // 旧任务刚被停止时占用锁释放存在短暂延迟，短延迟后重试一次创建；
        // 仍 409 说明有其他真实运行中的任务，回退到重连 existing_job_id
        await new Promise((resolve) => setTimeout(resolve, 300))
        postRes = await postJob()
      }

      if (postRes.status === 409) {
        const errBody = await postRes.json()
        jobId = errBody.detail?.existing_job_id
        if (!jobId) {
          throw new Error(
            typeof errBody.detail === 'string'
              ? errBody.detail
              : errBody.detail?.message || '任务冲突'
          )
        }
        const pj = readPreviewPendingJob(aid, sid)
        startSeq = pj?.seq ?? 0
      } else if (!postRes.ok) {
        let detail = `HTTP ${postRes.status}`
        try {
          const errBody = await postRes.json()
          detail = errBody.detail || errBody.msg || detail
        } catch {
          /* ignore */
        }
        throw new Error(detail)
      } else {
        const body = await postRes.json()
        jobId = body.data?.job_id
        if (!jobId) throw new Error('未返回 job_id')
        savePreviewPendingJob(aid, sid, { job_id: jobId, seq: 0 })
      }

      activeJobId.value = jobId

      const openStream = () =>
        fetch(`${baseApi}/user-agent/chat/jobs/${jobId}/stream?since_seq=${startSeq}`, {
          credentials: 'include',
          headers: { token },
          signal: ac.signal,
        })

      let streamRes = await openStream()
      if (streamRes.status === 401) {
        const fresh = await tryRefreshToken()
        if (fresh) {
          token = fresh
          streamRes = await openStream()
        }
      }

      if (!streamRes.ok) {
        let detail = `HTTP ${streamRes.status}`
        try {
          const errBody = await streamRes.json()
          detail = errBody.detail || errBody.msg || detail
        } catch {
          /* ignore */
        }
        throw new Error(detail)
      }

      const reader = streamRes.body?.getReader()
      const decoder = new TextDecoder()
      if (!reader) throw new Error('No response body')

      await readPreviewChatJobSseStream({
        reader,
        decoder,
        idx,
        jobId,
        agentId: aid,
        sessionId: sid,
        messagesRef: messages,
        initialSeq: startSeq,
      })
    } finally {
      streamAbortController.value = null
      activeJobId.value = null
      activeAssistantIdx.value = -1
    }
  }

  async function sendMessage(rawInput) {
    const aid = unref(agentId)
    const token = getToken()
    if (!aid || !token) {
      window.$message?.warning(t('views.agents.chat_msg_need_login'))
      return
    }
    if (sending.value) return
    const text = String(rawInput || '').trim()
    if (!text) return

    sending.value = true
    const userMsg = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text,
    }

    const wasIntro = sessionPhase.value === 'intro'
    messages.value.push(userMsg)
    if (wasIntro) sessionPhase.value = 'chat'

    const assistantId = `a-${Date.now()}`
    messages.value.push({
      id: assistantId,
      role: 'assistant',
      content: '',
      errorText: undefined,
      pending: true,
      thinkingOpen: false,
      thinkingItems: [],
      ragTrace: null,
    })

    const idx = messages.value.findIndex((m) => m.id === assistantId)

    try {
      await postJobAndConsumeStream({
        regenerate: false,
        message: text,
        attachmentIds: [],
        assistantIdx: idx,
      })
    } catch (error) {
      clearPreviewPendingJob(aid, unref(sessionId))
      if (streamStoppedByUser.value) {
        if (idx !== -1 && messages.value[idx]?.pending) {
          const row = messages.value[idx]
          messages.value[idx] = {
            ...row,
            pending: false,
            stoppedByUser: true,
            errorText: undefined,
          }
        }
      } else if (idx !== -1) {
        const row = messages.value[idx]
        messages.value[idx] = {
          ...row,
          errorText: `${t('views.agents.chat_msg_stream_error')}：${error?.message || error}`,
          pending: false,
        }
      }
    } finally {
      sending.value = false
    }
  }

  function toggleThinking(msg) {
    const idx = messages.value.findIndex((m) => m.id === msg.id)
    if (idx === -1) return
    const row = messages.value[idx]
    messages.value[idx] = {
      ...row,
      thinkingOpen: !(row.thinkingOpen ?? false),
    }
  }

  return {
    messages,
    sessionPhase,
    sending,
    resetChat,
    sendMessage,
    stopGeneration,
    toggleThinking,
  }
}
