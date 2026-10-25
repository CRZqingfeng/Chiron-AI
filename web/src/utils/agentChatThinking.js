/**
 * 思考区时间线：工具步骤（step）、过渡文案（text）、规划清单（todo）按发生顺序交错。
 * 对话页与编辑器预览的 SSE reducer、历史回放共用。
 *
 * item 结构：
 *   { type: 'step', icon, label, detail }
 *   { type: 'text', text }
 *   { type: 'todo', todos: [{ content, status }] }   // U9a，status: pending|in_progress|completed
 */

const TODO_STATUSES = ['pending', 'in_progress', 'completed']

/** 收敛待办清单：丢掉空条目，未知状态按 pending（后端已做一遍，前端再兜一次） */
export function normalizeTodos(todos) {
  const list = Array.isArray(todos) ? todos : []
  const out = []
  for (const t of list) {
    if (!t || typeof t !== 'object') continue
    const content = String(t.content || '').trim()
    if (!content) continue
    const status = TODO_STATUSES.includes(t.status) ? t.status : 'pending'
    out.push({ content, status })
  }
  return out
}

/** 清单项的状态图标与色调（色调只用于 CSS class，不参与逻辑） */
export function todoMark(status) {
  if (status === 'completed') return { icon: '●', tone: 'done' }
  if (status === 'in_progress') return { icon: '◐', tone: 'doing' }
  return { icon: '○', tone: 'pending' }
}

/** 进度文案形如 "2/5"，全空时给空串 */
export function todoProgress(todos) {
  const list = Array.isArray(todos) ? todos : []
  if (!list.length) return ''
  const done = list.filter((t) => t?.status === 'completed').length
  return `${done}/${list.length}`
}

/**
 * 一条 todo_list 事件并入 items（整表替换语义，返回新数组）。
 *
 * 为什么是「替换最后一条 todo」而不是「每次追加」：write_todos 每次写的是完整清单，
 * 追加会让时间线堆满同一张表的多个版本。整轮只保留一条，实时覆盖、终态落库。
 */
export function upsertTodoItem(items, todos) {
  const list = Array.isArray(items) ? [...items] : []
  const normalized = normalizeTodos(todos)
  if (!normalized.length) return list
  const item = { type: 'todo', todos: normalized }
  let lastTodoIdx = -1
  for (let i = list.length - 1; i >= 0; i--) {
    if (list[i]?.type === 'todo') {
      lastTodoIdx = i
      break
    }
  }
  if (lastTodoIdx >= 0) list[lastTodoIdx] = item
  else list.push(item)
  return list
}

/** 压掉过渡文本中 3+ 连续换行，避免 Markdown 空段叠出整行空白 */
export function normalizeThinkingText(text) {
  return String(text || '').replace(/\n{3,}/g, '\n\n')
}

/**
 * 将一条 SSE thinking_item 事件并入 items（返回新数组）。
 * @param {Array} items 当前 thinkingItems
 * @param {object} item 事件携带的 item
 * @param {boolean} append true 时 text 类型并入尾部已有 text 项（流式续写）
 */
export function applyThinkingItem(items, item, append = false) {
  const list = Array.isArray(items) ? [...items] : []
  if (!item || typeof item !== 'object') return list
  if (item.type === 'text') {
    const text = normalizeThinkingText(item.text)
    if (!text.trim()) return list
    const last = list[list.length - 1]
    if (append && last && last.type === 'text') {
      list[list.length - 1] = { ...last, text: normalizeThinkingText(last.text + text) }
      return list
    }
    list.push({ type: 'text', text })
    return list
  }
  if (item.type === 'step') {
    list.push({
      type: 'step',
      icon: item.icon || '▸',
      label: item.label || '',
      detail: item.detail || '',
    })
  }
  if (item.type === 'todo') {
    return upsertTodoItem(list, item.todos)
  }
  return list
}

/**
 * 历史行兼容：优先用落库的 thinking_items；缺失时按旧字段 rag_steps + thinking_text 合成
 * （旧数据无交错信息，步骤整体排前、思考排后）。
 */
export function buildThinkingItemsFromRow(row) {
  if (Array.isArray(row?.thinking_items) && row.thinking_items.length) {
    return row.thinking_items
      .map((it) => {
        if (!it || typeof it !== 'object') return null
        if (it.type === 'text') {
          const text = normalizeThinkingText(it.text)
          return text.trim() ? { type: 'text', text } : null
        }
        if (it.type === 'step') {
          return {
            type: 'step',
            icon: it.icon || '▸',
            label: it.label || '',
            detail: it.detail || '',
          }
        }
        // U9a：漏了这一支，刷新后清单就消失了（历史回放靠这里过白名单）
        if (it.type === 'todo') {
          const todos = normalizeTodos(it.todos)
          return todos.length ? { type: 'todo', todos } : null
        }
        return null
      })
      .filter(Boolean)
  }
  const items = []
  const steps = Array.isArray(row?.rag_steps) ? row.rag_steps : []
  for (const s of steps) {
    if (!s || typeof s !== 'object') continue
    items.push({ type: 'step', icon: s.icon || '▸', label: s.label || '', detail: s.detail || '' })
  }
  const text = normalizeThinkingText(row?.thinking_text)
  if (text.trim()) items.push({ type: 'text', text })
  return items
}

/** 思考胶囊在生成中展示最近一个工具步骤名（text 项不算步骤） */
export function lastThinkingStepLabel(m) {
  const items = Array.isArray(m?.thinkingItems) ? m.thinkingItems : []
  for (let i = items.length - 1; i >= 0; i--) {
    if (items[i]?.type === 'step' && items[i].label) return items[i].label
  }
  return ''
}

/**
 * 思考胶囊文案：排队提示 > 最近工具步骤 > 思考中 / 思考结束。
 * @param {object} m 消息行（含 pending / queuedWaiting / thinkingItems）
 * @param {Function} t vue-i18n 的 $t
 */
export function thinkingPillLabel(m, t) {
  // 审批中断时 pending 已经是 false（本轮不会再有输出），胶囊会显示"思考结束"，
  // 用户看到的就是"跑完了但什么都没有"。真实状态是**在等你点确认**，所以优先说清楚。
  if (m?.approvalPending?.actions?.length) {
    return t('views.agents.chat_feed_waiting_approval')
  }
  if (m?.pending && m.queuedWaiting) {
    return t('views.agents.chat_feed_queued', { n: m.queuedWaiting })
  }
  if (m?.pending) return lastThinkingStepLabel(m) || t('views.agents.chat_feed_thinking')
  return t('views.agents.chat_feed_thinking_done')
}
