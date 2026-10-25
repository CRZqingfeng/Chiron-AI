/**
 * 对话审批卡片的纯函数工具。
 *
 * 单独成文件而不是塞进 useChatMessages.js：那里 `import { ref } from 'vue'`，
 * 无法被 node 直接加载做纯函数验证；这里只做数组处理，零依赖。
 */

/**
 * 同一会话只保留**最后一条**待确认卡片，其余清掉 `approvalPending`。
 *
 * 为什么需要：
 * 1. 审批中断续跑本应原地覆盖（后端 `_finish_interrupted` 已改成这样），
 *    但历史数据里可能残留多条——旧版本是 append 落库的，一次提问会留下
 *    「11:52 的卡 + 12:00 的卡」两张。
 * 2. 卡片失效后**没有恢复路径**：点旧卡只会得到「审批已失效」，
 *    用户既点不动也删不掉，只能换个会话。
 * 3. 取最后一条，是因为它对应最近一次中断，也是 checkpoint 里唯一还活着的那次。
 *
 * @param {Array} list 消息数组（按时间正序）
 * @returns {Array} 新数组；无待确认时原样返回（保持引用稳定）
 */
export function keepLatestApprovalOnly(list) {
  if (!Array.isArray(list) || list.length === 0) return list
  let lastIdx = -1
  for (let i = 0; i < list.length; i += 1) {
    if (list[i] && list[i].approvalPending) lastIdx = i
  }
  // 零条或一条：不需要改写，直接返回原引用（避免无意义 re-render）
  if (lastIdx < 0) return list
  let count = 0
  for (let i = 0; i < list.length; i += 1) {
    if (list[i] && list[i].approvalPending) count += 1
  }
  if (count <= 1) return list
  return list.map((m, i) =>
    i === lastIdx || !(m && m.approvalPending) ? m : { ...m, approvalPending: undefined },
  )
}
