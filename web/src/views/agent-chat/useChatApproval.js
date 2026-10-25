/**
 * 工具审批确认：允许 / 拒绝一次被中断的高危操作。
 *
 * 与 `useChatMcpConfirm` 的区别是「恢复方式」：
 * - MCP 那套是「写一张批准票 + 整轮重跑」，已执行过的工具会再跑一遍；
 * - 这里走 LangGraph 的 `interrupt` 断点恢复（`Command(resume=...)`），
 *   图从中断点继续，已完成的步骤不会重跑。
 *
 * 交互形态沿用 `resumeApprovedMcp` 的范式：发一个新 job（带原 thread_id），
 * 消费同一个 SSE 流，后端会把原有那条助手消息原地更新为最终回答。
 */
import { ref } from 'vue'

export function useChatApproval({
  messages,
  sending,
  chatAgentId,
  scrollBodyToBottom,
  postChatJobAndConsumeStream,
  getToken,
  t,
}) {
  // 正在提交确认的消息行 id 集合：防重复点击（连点两次会创建两个 job）
  const approvingIds = ref(new Set())

  function isApproving(messageId) {
    return approvingIds.value.has(messageId)
  }

  function markApproving(messageId, active) {
    const next = new Set(approvingIds.value)
    if (active) next.add(messageId)
    else next.delete(messageId)
    approvingIds.value = next
  }

  async function decideApproval(assistantMsg, decision) {
    const token = getToken()
    const ap = assistantMsg?.approvalPending
    const agentId = Number(chatAgentId.value)
    if (!['approve', 'reject'].includes(decision)) return
    if (approvingIds.value.has(assistantMsg?.id)) return

    // 不要静默 return：按钮点了没反应、控制台也没请求，是最难排查的一类问题
    // （U5b 第一次人工验证就撞上了这个）。缺什么就说清楚缺什么。
    if (!token) {
      window.$message?.warning(t('views.agents.chat_msg_need_login'))
      return
    }
    if (!Number.isFinite(agentId)) {
      window.$message?.error(t('views.agents.chat_error_load_agent'))
      return
    }
    if (!ap?.threadId || !ap?.messageId) {
      window.$message?.warning(t('views.agents.chat_approval_stale'))
      return
    }

    const assistantIdx = messages.value.findIndex((x) => x.id === assistantMsg.id)
    if (assistantIdx === -1) {
      window.$message?.warning(t('views.agents.chat_approval_stale'))
      return
    }

    markApproving(assistantMsg.id, true)
    const row = messages.value[assistantIdx]
    messages.value[assistantIdx] = {
      ...row,
      pending: true,
      mcpExecuting: decision === 'approve',
      errorText: undefined,
      stoppedByUser: false,
    }
    sending.value = true
    try {
      await postChatJobAndConsumeStream({
        agentId,
        token,
        regenerate: false,
        message: '',
        attachmentIds: [],
        assistantIdx,
        resumeThreadId: ap.threadId,
        resumeDecision: decision,
        targetMessageId: ap.messageId,
      })
      // 成功走完：后端已把这条消息原地更新为最终回答，卡片可以退场。
      // 放在 finally 之外，避免提交失败时把用户的确认入口也抹掉。
      const cur = messages.value[assistantIdx]
      if (cur) {
        messages.value[assistantIdx] = { ...cur, approvalPending: undefined }
      }
    } catch (error) {
      const cur = messages.value[assistantIdx]
      if (cur) {
        messages.value[assistantIdx] = {
          ...cur,
          pending: false,
          mcpExecuting: false,
          // 不清 approvalPending：让用户能重试
          errorText: t('views.agents.chat_msg_stream_error') + `：${error?.message || error}`,
        }
      }
    } finally {
      markApproving(assistantMsg.id, false)
      sending.value = false
      scrollBodyToBottom()
    }
  }

  return { decideApproval, isApproving, approvingIds }
}
