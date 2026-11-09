/** 编辑器预览试聊 SSE 工具（与 agent-chat 字段一致） */

import { applyThinkingItem } from '@/utils/agentChatThinking'

const PENDING_JOB_PREFIX = 'chiron_ai_preview_job_'

export function previewPendingJobStorageKey(agentId, sid) {
  return `${PENDING_JOB_PREFIX}${agentId}_${sid}`
}

export function savePreviewPendingJob(agentId, sid, payload) {
  try {
    if (agentId && sid && payload?.job_id) {
      sessionStorage.setItem(previewPendingJobStorageKey(agentId, sid), JSON.stringify(payload))
    }
  } catch {
    /* ignore */
  }
}

export function readPreviewPendingJob(agentId, sid) {
  try {
    const s = sessionStorage.getItem(previewPendingJobStorageKey(agentId, sid))
    if (!s) return null
    return JSON.parse(s)
  } catch {
    return null
  }
}

export function clearPreviewPendingJob(agentId, sid) {
  try {
    if (agentId && sid) sessionStorage.removeItem(previewPendingJobStorageKey(agentId, sid))
  } catch {
    /* ignore */
  }
}

/** 将单条 SSE JSON 事件应用到助手消息行 */
export function applyPreviewChatSsePayload(data, messagesRef, idx) {
  if (idx === -1) return
  const list = messagesRef.value ?? messagesRef
  if (data.type === 'content') {
    const row = list[idx]
    const resumed = !!row.mcpExecuting
    list[idx] = {
      ...row,
      content: resumed ? data.content || '' : (row.content || '') + (data.content || ''),
      mcpExecuting: resumed ? false : row.mcpExecuting,
      pending: false,
      queuedWaiting: undefined,
      thinkingOpen: row.thinkingOpen ?? false,
      ragTrace: row.ragTrace ?? null,
    }
  } else if (data.type === 'thinking_item') {
    const cur = list[idx]
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
    list[idx] = patch
  } else if (data.type === 'trace') {
    const cur = list[idx]
    list[idx] = {
      ...cur,
      ragTrace: data.rag_trace || null,
      pending: cur.pending,
    }
  } else if (data.type === 'sources') {
    const cur = list[idx]
    list[idx] = {
      ...cur,
      sources: Array.isArray(data.sources) ? data.sources : [],
      pending: cur.pending,
    }
  } else if (data.type === 'error') {
    const cur = list[idx]
    list[idx] = {
      ...cur,
      errorText: data.content || '',
      pending: false,
      thinkingOpen: cur.thinkingOpen ?? false,
      ragTrace: cur.ragTrace ?? null,
    }
  } else if (data.type === 'cancelled') {
    const cur = list[idx]
    list[idx] = {
      ...cur,
      stoppedByUser: true,
      pending: false,
      thinkingOpen: cur.thinkingOpen ?? false,
      ragTrace: cur.ragTrace ?? null,
      errorText: undefined,
    }
  } else if (data.type === 'approval_required') {
    // 试聊面板不渲染审批卡片（它没有会话上下文与 resume 链路）。
    // 但**不能静默跳过**：图就停在这里，用户看到的是「生成结束却没有回答」，
    // 完全不知道发生了什么。明确告诉他去正式对话里确认。
    const cur = list[idx]
    const actions = (data.request || {}).actions || []
    const names =
      actions
        .map((a) => a?.name)
        .filter(Boolean)
        .join('、') || '高危操作'
    list[idx] = {
      ...cur,
      errorText: `这次对话需要确认才能执行（${names}）。试聊不支持审批，请到正式对话中重试。`,
      pending: false,
      thinkingOpen: cur.thinkingOpen ?? false,
      ragTrace: cur.ragTrace ?? null,
    }
  } else if (data.type === 'queued') {
    // 并发闸门排队提示：生成开始前先收到该事件
    const cur = list[idx]
    list[idx] = {
      ...cur,
      queuedWaiting: Math.max(1, Number(data.waiting) || 1),
      pending: true,
    }
  } else if (data.type === 'done') {
    const row = list[idx]
    list[idx] = {
      ...row,
      pending: false,
      mcpExecuting: false,
      stoppedByUser: data.cancelled ? true : row.stoppedByUser,
    }
  }
}

export async function readPreviewChatJobSseStream({
  reader,
  decoder,
  idx,
  jobId,
  agentId,
  sessionId,
  messagesRef,
  initialSeq = 0,
  onChunk,
}) {
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
        clearPreviewPendingJob(agentId, sessionId)
        continue
      }
      try {
        const data = JSON.parse(dataStr)
        applyPreviewChatSsePayload(data, messagesRef, idx)
        seq += 1
        savePreviewPendingJob(agentId, sessionId, { job_id: jobId, seq })
        if (data.type === 'done') {
          clearPreviewPendingJob(agentId, sessionId)
        }
      } catch (e) {
        console.warn('Preview SSE parse error:', e)
      }
    }
    onChunk?.()
  }
  if (idx !== -1) {
    const list = messagesRef.value ?? messagesRef
    if (list[idx]?.pending) {
      list[idx] = { ...list[idx], pending: false }
    }
  }
}
