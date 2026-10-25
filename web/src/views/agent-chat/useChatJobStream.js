import { onMounted, onUnmounted, ref } from 'vue'
import { getToken } from '@/utils'
import { applyThinkingItem, upsertTodoItem } from '@/utils/agentChatThinking'
import { useAgentSidebarStore, useRecentAgentsStore } from '@/store'

const PENDING_JOB_PREFIX = 'chiron_ai_chat_job_'

/** pending job 快照写入节流间隔（sessionStorage 是同步写，避免每事件一次） */
const JOB_SNAPSHOT_SAVE_MS = 250

/** 流式行 markdown 渲染节流：renderVersion 每窗口最多 bump 一次 */
const RENDER_THROTTLE_MS = 100

export function pendingJobStorageKey(agentId, sid) {
  return `${PENDING_JOB_PREFIX}${agentId}_${sid}`
}

export function savePendingChatJob(agentId, sid, payload) {
  try {
    if (agentId && sid && payload?.job_id) {
      const prev = readPendingChatJob(agentId, sid) || {}
      sessionStorage.setItem(
        pendingJobStorageKey(agentId, sid),
        JSON.stringify({ ...prev, ...payload })
      )
    }
  } catch {
    /* ignore */
  }
}

export function readPendingChatJob(agentId, sid) {
  try {
    const s = sessionStorage.getItem(pendingJobStorageKey(agentId, sid))
    if (!s) return null
    return JSON.parse(s)
  } catch {
    return null
  }
}

export function clearPendingChatJob(agentId, sid) {
  try {
    if (agentId && sid) sessionStorage.removeItem(pendingJobStorageKey(agentId, sid))
  } catch {
    /* ignore */
  }
}

/**
 * Job / SSE / 取消 / 重连。
 * `reloadSessionMessages` 在调用时解析，便于与 useChatMessages 互引。
 */
export function useChatJobStream({
  messages,
  sessionId,
  sending,
  sessionPhase,
  pageLoading,
  loadError,
  agent,
  useKnowledgeRetrieval,
  useWebSearch,
  scrollBodyToBottom,
  reloadSessionMessages,
}) {
  const baseApi = import.meta.env.VITE_BASE_API || '/api/v1'
  const streamAbortController = ref(null)
  const activeJobId = ref(null)
  const activeAssistantIdx = ref(-1)
  const streamStoppedByUser = ref(false)
  const recentAgentsStore = useRecentAgentsStore()
  const agentSidebarStore = useAgentSidebarStore()

  // —— pending job 快照节流：页面隐藏/卸载时强制落盘，
  //    保证刷新恢复用的 since_seq 不落后于实际消费进度 ——
  let lastSnapshotSaveAt = 0
  let snapshotTimer = null
  let pendingSnapshot = null

  function flushJobSnapshot() {
    if (snapshotTimer) {
      clearTimeout(snapshotTimer)
      snapshotTimer = null
    }
    if (pendingSnapshot) {
      savePendingChatJob(pendingSnapshot.agentId, pendingSnapshot.sid, pendingSnapshot.payload)
      pendingSnapshot = null
    }
  }

  function cancelJobSnapshot() {
    if (snapshotTimer) {
      clearTimeout(snapshotTimer)
      snapshotTimer = null
    }
    pendingSnapshot = null
  }

  function scheduleJobSnapshot(agentId, payload) {
    if (!agentId || !sessionId.value || !payload?.job_id) return
    pendingSnapshot = { agentId, sid: sessionId.value, payload }
    const now = Date.now()
    if (now - lastSnapshotSaveAt >= JOB_SNAPSHOT_SAVE_MS) {
      lastSnapshotSaveAt = now
      flushJobSnapshot()
    } else if (!snapshotTimer) {
      snapshotTimer = setTimeout(() => {
        lastSnapshotSaveAt = Date.now()
        flushJobSnapshot()
      }, JOB_SNAPSHOT_SAVE_MS)
    }
  }

  function flushJobSnapshotOnExit() {
    if (document.visibilityState === 'hidden') flushJobSnapshot()
  }

  onMounted(() => {
    window.addEventListener('pagehide', flushJobSnapshot)
    document.addEventListener('visibilitychange', flushJobSnapshotOnExit)
  })

  onUnmounted(() => {
    window.removeEventListener('pagehide', flushJobSnapshot)
    document.removeEventListener('visibilitychange', flushJobSnapshotOnExit)
    cancelJobSnapshot()
  })

  // —— 流式行 markdown 渲染节流：
  //    正文仍按 token 实时累积，但 renderVersion / displayContent 快照每
  //    ~100ms 才推进一次，ChatMessageItem 的 computed 只依赖快照 ——
  let renderFlushTimer = null
  let pendingRenderRow = null

  function bumpRenderVersion(row) {
    if (!row) return
    row.renderVersion = (row.renderVersion || 0) + 1
    row.renderVersionAt = Date.now()
    row.displayContent = row.content
  }

  function scheduleRenderVersion(row) {
    if (!row) return
    if (!row.renderVersionAt || Date.now() - row.renderVersionAt >= RENDER_THROTTLE_MS) {
      if (renderFlushTimer) {
        clearTimeout(renderFlushTimer)
        renderFlushTimer = null
        pendingRenderRow = null
      }
      bumpRenderVersion(row)
      return
    }
    // 每次调度都改为最新行对象：applyChatSsePayload 每事件换新对象，
    // 若 bump 落在旧引用上，renderVersion/displayContent 会写入脱离列表的孤儿行
    pendingRenderRow = row
    if (!renderFlushTimer) {
      renderFlushTimer = setTimeout(() => {
        renderFlushTimer = null
        const target = pendingRenderRow
        pendingRenderRow = null
        if (target) bumpRenderVersion(target)
      }, RENDER_THROTTLE_MS)
    }
  }

  function flushRenderVersion(row) {
    if (renderFlushTimer) {
      clearTimeout(renderFlushTimer)
      renderFlushTimer = null
      pendingRenderRow = null
    }
    if (row) bumpRenderVersion(row)
  }

  function applyChatSsePayload(data, idx) {
    if (idx === -1) return
    if (data.type === 'resume_snapshot') {
      // 后端在 Redis 事件流丢失（重启/过期）后，用 PG 快照整体恢复已生成的内容。
      // 这里必须"整体替换"而不是追加：本地可能已有部分内容，快照才是权威状态。
      // kind === 'snapshot' 说明只恢复到中途断点，给用户一句说明。
      const row = messages.value[idx]
      messages.value[idx] = {
        ...row,
        content: data.content || '',
        pending: false,
        queuedWaiting: undefined,
        mcpExecuting: false,
        thinkingOpen: row.thinkingOpen ?? false,
        ragTrace: row.ragTrace ?? null,
        errorText:
          data.kind === 'snapshot' ? '连接中断，以上为已生成的部分内容' : undefined,
      }
      flushRenderVersion(messages.value[idx])
    } else if (data.type === 'content') {
      const row = messages.value[idx]
      const resumed = !!row.mcpExecuting
      messages.value[idx] = {
        ...row,
        content: resumed ? data.content || '' : (row.content || '') + (data.content || ''),
        mcpExecuting: resumed ? false : row.mcpExecuting,
        pending: false,
        queuedWaiting: undefined,
        thinkingOpen: row.thinkingOpen ?? false,
        ragTrace: row.ragTrace ?? null,
      }
      scheduleRenderVersion(messages.value[idx])
    } else if (data.type === 'thinking_item') {
      const cur = messages.value[idx]
      const item = data.item || {}
      const patch = {
        ...cur,
        thinkingItems: applyThinkingItem(cur.thinkingItems, item, !!data.append),
        thinkingOpen: true,
        pending: cur.pending,
        queuedWaiting: undefined,
      }
      if (data.moved_from_content && item.type === 'text') {
        const text = item.text || ''
        const curContent = cur.content || ''
        patch.content =
          text && curContent.endsWith(text)
            ? curContent.slice(0, curContent.length - text.length)
            : curContent
        patch.pending = true
      }
      messages.value[idx] = patch
    } else if (data.type === 'todo_list') {
      // U9a 规划清单：整表替换（与后端 write_todos 语义一致）。
      // 不另存 msg.todos，而是就地 upsert 一条 type=todo 的思考区项——
      // 这样实时渲染与历史回放（DB 里 thinking_items 的终态）走同一套模板，
      // 不会出现「流式时一个样、刷新后另一个样」。
      const cur = messages.value[idx]
      messages.value[idx] = {
        ...cur,
        thinkingItems: upsertTodoItem(cur.thinkingItems, data.todos),
        thinkingOpen: true,
        pending: cur.pending,
        queuedWaiting: undefined,
      }
    } else if (data.type === 'trace') {
      const cur = messages.value[idx]
      messages.value[idx] = {
        ...cur,
        ragTrace: data.rag_trace || null,
        pending: cur.pending,
      }
    } else if (data.type === 'sources') {
      const cur = messages.value[idx]
      messages.value[idx] = {
        ...cur,
        sources: Array.isArray(data.sources) ? data.sources : [],
        pending: cur.pending,
      }
    } else if (data.type === 'error') {
      const cur = messages.value[idx]
      messages.value[idx] = {
        ...cur,
        errorText: data.content || '',
        pending: false,
        mcpExecuting: false,
        thinkingOpen: cur.thinkingOpen ?? false,
        ragTrace: cur.ragTrace ?? null,
      }
      flushRenderVersion(messages.value[idx])
    } else if (data.type === 'cancelled') {
      const cur = messages.value[idx]
      messages.value[idx] = {
        ...cur,
        stoppedByUser: true,
        pending: false,
        mcpExecuting: false,
        thinkingOpen: cur.thinkingOpen ?? false,
        ragTrace: cur.ragTrace ?? null,
        errorText: undefined,
      }
      flushRenderVersion(messages.value[idx])
    } else if (data.type === 'mcp_confirmation_required') {
      const cur = messages.value[idx]
      const item = data.confirmation || {}
      const list = Array.isArray(cur.mcpConfirmations) ? cur.mcpConfirmations : []
      if (item.pending_id && !list.some((x) => x.pending_id === item.pending_id)) {
        list.push(item)
      }
      messages.value[idx] = {
        ...cur,
        mcpConfirmations: list,
        pending: cur.pending,
      }
    } else if (data.type === 'approval_required') {
      // 工具审批中断：图停在「模型已产出 tool_calls、工具尚未执行」的位置。
      // 与 MCP 的 mcpConfirmations 分开存：两者字段结构完全不同
      // （这里是 interrupt_id + actions[]，MCP 是 pending_id + args_preview），
      // 合并会让模板到处是 if 分支。
      const cur = messages.value[idx]
      const req = data.request || {}
      const actions = Array.isArray(req.actions) ? req.actions : []
      if (!actions.length) return
      messages.value[idx] = {
        ...cur,
        approvalPending: {
          interruptId: req.interrupt_id || null,
          actions,
          allowedDecisions: Array.isArray(req.allowed_decisions)
            ? req.allowed_decisions
            : ['approve', 'reject'],
          // 恢复时要指回原 checkpoint。messageId 要等 done 事件才有
          // （消息在本轮收尾时才落库），先留 null，done 分支补。
          threadId: data.thread_id || cur.approvalPending?.threadId || null,
          messageId: cur.approvalPending?.messageId ?? null,
        },
        // 停掉转圈：这一轮不会再有内容产出，等用户决定
        pending: false,
        mcpExecuting: false,
        thinkingOpen: cur.thinkingOpen ?? false,
      }
      flushRenderVersion(messages.value[idx])
    } else if (data.type === 'queued') {
      // 并发闸门排队提示：生成开始前先收到该事件
      const cur = messages.value[idx]
      messages.value[idx] = {
        ...cur,
        queuedWaiting: Math.max(1, Number(data.waiting) || 1),
        pending: true,
      }
    } else if (data.type === 'done') {
      const row = messages.value[idx]
      const patch = {
        ...row,
        pending: false,
        mcpExecuting: false,
        stoppedByUser: data.cancelled ? true : row.stoppedByUser,
      }
      // 中断收尾时后端才把那条「待确认」消息落库，因此 message_id 只能在 done 里拿到。
      // 恢复接口要用它作 target_message_id（精准定位、原地更新），缺了就点不动按钮。
      if (data.interrupted && row.approvalPending) {
        patch.approvalPending = {
          ...row.approvalPending,
          messageId: data.message_id ?? row.approvalPending.messageId ?? null,
        }
      }
      messages.value[idx] = patch
      flushRenderVersion(messages.value[idx])
    }
  }

  async function readChatJobSseStream(reader, decoder, idx, jobId, agentId, initialSeq = 0) {
    let buffer = ''
    let seq = initialSeq

    for (;;) {
      let chunk
      try {
        chunk = await reader.read()
      } catch {
        break
      }
      const { done, value } = chunk
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      let eventEndIndex
      while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
        const eventStr = buffer.slice(0, eventEndIndex)
        buffer = buffer.slice(eventEndIndex + 2)

        if (!eventStr.startsWith('data: ')) continue
        const dataStr = eventStr.slice(6)
        if (dataStr === '[DONE]') {
          clearPendingChatJob(agentId, sessionId.value)
          continue
        }
        try {
          const data = JSON.parse(dataStr)
          applyChatSsePayload(data, idx)
          seq += 1
          scheduleJobSnapshot(agentId, { job_id: jobId, seq })
          if (data.type === 'done') {
            cancelJobSnapshot()
            clearPendingChatJob(agentId, sessionId.value)
          }
        } catch (e) {
          console.warn('SSE parse error:', e)
        }
      }
      scrollBodyToBottom()
    }
    if (idx !== -1 && messages.value[idx]?.pending) {
      const row = messages.value[idx]
      messages.value[idx] = { ...row, pending: false }
    }
    flushRenderVersion(idx !== -1 ? messages.value[idx] : null)
  }

  async function stopActiveChatGeneration() {
    if (!sending.value) return
    streamStoppedByUser.value = true
    const jid = activeJobId.value
    const ix = activeAssistantIdx.value
    const token = getToken()
    const aid = agent.value?.id
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
    } else if (token && aid && sessionId.value) {
      try {
        await fetch(
          `${baseApi}/user-agent/chat/active_job/cancel?agent_id=${aid}&session_id=${encodeURIComponent(
            sessionId.value
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
    if (aid && sessionId.value) {
      cancelJobSnapshot()
      clearPendingChatJob(aid, sessionId.value)
    }
    if (ix >= 0 && messages.value[ix]?.role === 'assistant') {
      const row = messages.value[ix]
      messages.value[ix] = {
        ...row,
        pending: false,
        stoppedByUser: true,
        errorText: undefined,
      }
      flushRenderVersion(messages.value[ix])
    }
    sending.value = false
  }

  async function postChatJobAndConsumeStream({
    agentId,
    token,
    regenerate,
    message,
    attachmentIds,
    assistantIdx,
    mcpApprovedPendingId = null,
    targetMessageId = null,
    resumeThreadId = null,
    resumeDecision = null,
  }) {
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
          headers: {
            'Content-Type': 'application/json',
            token,
          },
          body: JSON.stringify({
            agent_id: agentId,
            message,
            session_id: sessionId.value,
            use_knowledge_retrieval: useKnowledgeRetrieval.value,
            use_web_search: useWebSearch.value,
            attachment_ids: attachmentIds,
            regenerate,
            target_message_id: targetMessageId || undefined,
            mcp_approved_pending_id: mcpApprovedPendingId || undefined,
            // 审批恢复：thread 指向原 job 的 checkpoint，decision 决定批准/拒绝。
            // 后端据此走 `Command(resume=...)`，不会重复已完成的步骤。
            resume_thread_id: resumeThreadId || undefined,
            resume_decision: resumeDecision || undefined,
          }),
          signal: ac.signal,
        })

      let postRes = await postJob()
      if (postRes.status === 409) {
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
        const pj = readPendingChatJob(agentId, sessionId.value)
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
        savePendingChatJob(agentId, sessionId.value, {
          job_id: jobId,
          seq: 0,
          regenerate: !!regenerate,
          target_message_id: targetMessageId || null,
        })
      }

      activeJobId.value = jobId

      const streamRes = await fetch(
        `${baseApi}/user-agent/chat/jobs/${jobId}/stream?since_seq=${startSeq}`,
        {
          credentials: 'include',
          headers: { token },
          signal: ac.signal,
        }
      )

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
      if (!reader) {
        throw new Error('No response body')
      }

      await readChatJobSseStream(reader, decoder, idx, jobId, agentId, startSeq)
      await recentAgentsStore.touch(agentId)
    } finally {
      streamAbortController.value = null
      activeJobId.value = null
      activeAssistantIdx.value = -1
    }
  }

  async function maybeResumePendingChatJob() {
    const agentId = agent.value?.id
    const token = getToken()
    if (!agentId || !token || sending.value || pageLoading.value || loadError.value) return

    const pj = readPendingChatJob(agentId, sessionId.value)
    if (!pj?.job_id) return

    try {
      const metaRes = await fetch(`${baseApi}/user-agent/chat/jobs/${pj.job_id}`, {
        credentials: 'include',
        headers: { token },
      })
      if (!metaRes.ok) {
        clearPendingChatJob(agentId, sessionId.value)
        return
      }
      const metaBody = await metaRes.json()
      const meta = metaBody.data ?? metaBody
      if (meta.status !== 'running') {
        clearPendingChatJob(agentId, sessionId.value)
        return
      }
    } catch {
      return
    }

    const last = messages.value[messages.value.length - 1]
    let idx = -1
    const reuseAssistantRow = last?.role === 'assistant' && last?.pending
    if (!reuseAssistantRow && pj.regenerate && pj.target_message_id) {
      const tIdx = messages.value.findIndex(
        (m) => m.role === 'assistant' && m.messageId === pj.target_message_id
      )
      if (tIdx !== -1) {
        const row = messages.value[tIdx]
        messages.value = messages.value.slice(0, tIdx + 1)
        messages.value[tIdx] = {
          ...row,
          content: '',
          errorText: undefined,
          stoppedByUser: false,
          pending: true,
          thinkingOpen: true,
          thinkingItems: [],
          ragTrace: null,
          sources: [],
        }
        idx = tIdx
      }
    }
    if (idx !== -1) {
      // 重新生成占位已就位
    } else if (reuseAssistantRow) {
      idx = messages.value.length - 1
    } else {
      const assistantId = `a-resume-${Date.now()}`
      messages.value.push({
        id: assistantId,
        role: 'assistant',
        content: '',
        errorText: undefined,
        pending: true,
        thinkingOpen: true,
        thinkingItems: [],
        ragTrace: null,
        sources: [],
      })
      idx = messages.value.length - 1
      sessionPhase.value = 'chat'
    }

    sending.value = true
    const sinceSeq = reuseAssistantRow ? pj.seq ?? 0 : 0
    streamStoppedByUser.value = false
    const ac = new AbortController()
    streamAbortController.value = ac
    activeJobId.value = pj.job_id
    activeAssistantIdx.value = idx
    try {
      const streamRes = await fetch(
        `${baseApi}/user-agent/chat/jobs/${pj.job_id}/stream?since_seq=${sinceSeq}`,
        {
          credentials: 'include',
          headers: { token },
          signal: ac.signal,
        }
      )
      if (!streamRes.ok) {
        clearPendingChatJob(agentId, sessionId.value)
        return
      }
      const reader = streamRes.body?.getReader()
      const decoder = new TextDecoder()
      if (!reader) return
      await readChatJobSseStream(reader, decoder, idx, pj.job_id, agentId, sinceSeq)
      if (
        pj.regenerate &&
        !streamStoppedByUser.value &&
        !messages.value[idx]?.mcpConfirmations?.length &&
        // 审批待确认时不能重拉：历史行会覆盖掉刚广播进来的 approvalPending
        !messages.value[idx]?.approvalPending
      ) {
        await reloadSessionMessages(agentId)
      }
      await recentAgentsStore.touch(agentId)
    } catch (e) {
      console.warn('resume job stream:', e)
    } finally {
      streamAbortController.value = null
      activeJobId.value = null
      activeAssistantIdx.value = -1
      sending.value = false
      scrollBodyToBottom()
      agentSidebarStore.bumpRefresh()
    }
  }

  return {
    streamStoppedByUser,
    stopActiveChatGeneration,
    postChatJobAndConsumeStream,
    maybeResumePendingChatJob,
    savePendingChatJob,
    readPendingChatJob,
    clearPendingChatJob,
  }
}
