<template>
  <div
    class="agent-chat-feed-item"
    :class="{ 'agent-chat-feed-item--user': message.role === 'user' }"
  >
    <template v-if="message.role === 'user'">
      <div class="agent-chat-user-row">
        <div class="agent-chat-user-bubble">
          <div class="agent-chat-user-text">{{ message.content }}</div>
        </div>
        <div
          v-if="message.attachments?.length"
          class="agent-chat-attachment-boxes agent-chat-attachment-boxes--user"
        >
          <ChatAttachmentItem
            v-for="(a, ai) in message.attachments"
            :key="`ua-${message.id}-${ai}`"
            :agent-id="chatAgentId"
            :session-id="sessionId"
            :attachment="a"
          />
        </div>
      </div>
    </template>
    <template v-else>
      <n-avatar
        round
        :size="40"
        :src="agentAvatarSrc"
        object-fit="cover"
        class="agent-chat-feed-avatar"
      />
      <div class="agent-chat-feed-col">
        <div class="agent-chat-feed-head">
          <span class="agent-chat-feed-name">{{ agentName }}</span>
          <span class="agent-chat-feed-ai-badge">{{ $t('views.agents.chat_feed_ai_badge') }}</span>
        </div>

        <div class="agent-chat-thinking-wrap">
          <button
            type="button"
            class="agent-chat-thinking-pill"
            :aria-expanded="message.thinkingOpen"
            @click="emit('toggle-thinking', message)"
          >
            <TheIcon icon="mdi:lightbulb-outline" :size="16" class="agent-chat-thinking-icon" />
            <span>{{ thinkingPillLabel(message, $t) }}</span>
            <TheIcon
              :icon="message.thinkingOpen ? 'mdi:chevron-up' : 'mdi:chevron-down'"
              :size="18"
              class="agent-chat-thinking-chevron"
            />
          </button>
          <div v-show="message.thinkingOpen" class="agent-chat-thinking-panel">
            <div v-if="message.thinkingItems?.length" class="agent-chat-thinking-steps">
              <template v-for="(item, sIdx) in message.thinkingItems" :key="sIdx">
                <div v-if="item.type === 'step'" class="agent-chat-thinking-step-line">
                  <span class="agent-chat-thinking-step-icon">{{ item.icon || '▸' }}</span>
                  <span class="agent-chat-thinking-step-label">{{ item.label }}</span>
                  <span v-if="item.detail" class="agent-chat-thinking-step-detail">{{
                    item.detail
                  }}</span>
                </div>
                <div
                  v-else-if="item.type === 'text'"
                  class="agent-chat-thinking-step-line agent-chat-thinking-step-line--text"
                >
                  <span class="agent-chat-thinking-step-icon">💭</span>
                  <span class="agent-chat-thinking-step-label">{{
                    $t('views.agents.chat_thinking_text_label')
                  }}</span>
                  <div
                    class="agent-chat-thinking-step-body agent-chat-md"
                    v-html="renderAgentChatMarkdown(item.text)"
                  ></div>
                </div>
                <!-- U9a 规划清单：实时（SSE todo_list）与回放（thinking_items 里的终态）
                     共用这一个分支，保证流式与刷新后长得一样 -->
                <div v-else-if="item.type === 'todo'" class="agent-chat-todo-block">
                  <div class="agent-chat-todo-head">
                    <span class="agent-chat-thinking-step-icon">🗒️</span>
                    <span class="agent-chat-thinking-step-label">{{
                      $t('views.agents.chat_todo_title')
                    }}</span>
                    <span class="agent-chat-todo-progress">{{ todoProgress(item.todos) }}</span>
                  </div>
                  <ul class="agent-chat-todo-list">
                    <li
                      v-for="(todo, tIdx) in item.todos || []"
                      :key="tIdx"
                      class="agent-chat-todo-item"
                      :class="`agent-chat-todo-item--${todoMark(todo.status).tone}`"
                    >
                      <span class="agent-chat-todo-mark">{{ todoMark(todo.status).icon }}</span>
                      <span class="agent-chat-todo-text">{{ todo.content }}</span>
                    </li>
                  </ul>
                </div>
              </template>
            </div>
            <p v-if="!message.thinkingItems?.length" class="agent-chat-thinking-placeholder">
              {{ $t('views.agents.chat_feed_thinking_placeholder') }}
            </p>
            <details
              v-if="message.ragTrace && Object.keys(message.ragTrace).length"
              class="agent-chat-rag-trace"
            >
              <summary>{{ $t('views.agents.chat_thinking_trace_summary') }}</summary>
              <div class="agent-chat-rag-trace-lines">
                <div v-if="message.ragTrace.retrieval_mode" class="agent-chat-trace-line">
                  {{ $t('views.agents.chat_thinking_trace_mode') }}：{{
                    message.ragTrace.retrieval_mode
                  }}
                </div>
                <div v-if="message.ragTrace.retrieval_stage" class="agent-chat-trace-line">
                  {{ $t('views.agents.chat_thinking_trace_stage') }}：{{
                    message.ragTrace.retrieval_stage
                  }}
                </div>
                <div v-if="message.ragTrace.grade_score" class="agent-chat-trace-line">
                  {{ $t('views.agents.chat_thinking_trace_grade') }}：{{
                    message.ragTrace.grade_score
                  }}
                </div>
                <div v-if="message.ragTrace.rewrite_strategy" class="agent-chat-trace-line">
                  {{ message.ragTrace.rewrite_strategy }}
                </div>
              </div>
            </details>
          </div>
        </div>

        <div class="agent-chat-feed-body">
          <div
            v-if="!message.pending && (message.errorText || '').trim()"
            class="agent-chat-msg-error"
            role="alert"
          >
            <div class="agent-chat-msg-error-label">
              {{ $t('views.agents.chat_feed_error_title') }}
            </div>
            <div class="agent-chat-msg-error-text">{{ message.errorText }}</div>
          </div>
          <div
            v-if="!message.pending && message.stoppedByUser && !(message.errorText || '').trim()"
            class="agent-chat-msg-stopped"
            role="status"
          >
            {{ $t('views.agents.chat_msg_aborted') }}
          </div>
          <!-- U5c 之前的 MCP 假审批记录：票据存活在 Redis（5 分钟），早就没了。
               不能提示「请刷新重试」——刷新不会变好，只会让用户反复试。 -->
          <div
            v-if="message.approvalLegacy && !message.pending"
            class="agent-chat-msg-stopped"
            role="status"
          >
            {{ $t('views.agents.chat_approval_legacy') }}
          </div>
          <div
            v-if="message.approvalPending?.actions?.length"
            class="agent-chat-approval"
          >
            <div class="agent-chat-approval-title">
              {{ $t('views.agents.chat_approval_title') }}
            </div>
            <div
              v-for="(act, ai) in message.approvalPending.actions"
              :key="`ap-${ai}`"
              class="agent-chat-approval-item"
            >
              <div class="agent-chat-approval-name">{{ act.description || act.name }}</div>
              <pre
                v-if="formatApprovalArgs(act.args)"
                class="agent-chat-approval-args"
              >{{ formatApprovalArgs(act.args) }}</pre>
            </div>
            <div class="agent-chat-approval-actions">
              <n-button
                size="tiny"
                type="primary"
                :disabled="sending || approvingIds.has(message.id)"
                @click="emit('approval-decide', message, 'approve')"
              >
                {{ $t('views.agents.chat_approval_allow') }}
              </n-button>
              <n-button
                size="tiny"
                quaternary
                :disabled="sending || approvingIds.has(message.id)"
                @click="emit('approval-decide', message, 'reject')"
              >
                {{ $t('views.agents.chat_approval_reject') }}
              </n-button>
            </div>
            <div class="agent-chat-approval-hint">
              {{ $t('views.agents.chat_approval_hint') }}
            </div>
          </div>
          <div
            v-if="!message.pending && (message.content || '').trim()"
            class="agent-chat-md"
            @click="emit('md-click', $event)"
            v-html="renderedContent"
          />
          <div v-if="!message.pending && message.sources?.length" class="agent-chat-sources">
            <span class="agent-chat-sources-label">{{
              $t('views.agents.chat_sources_label')
            }}</span>
            <template v-for="src in message.sources" :key="`src-${src.chunk_id || src.index}`">
              <a
                v-if="safeExternalHref(src.url)"
                :href="safeExternalHref(src.url)"
                target="_blank"
                rel="noopener noreferrer"
                class="agent-chat-source-chip"
                :data-kcite-chip="src.index"
              >
                [{{ src.index }}] {{ src.title || src.url }}
              </a>
              <a
                v-else-if="safeExternalHref(toSameOriginMediaUrl(src.image_url))"
                :href="safeExternalHref(toSameOriginMediaUrl(src.image_url))"
                target="_blank"
                rel="noopener noreferrer"
                class="agent-chat-source-chip"
                :data-kcite-chip="src.index"
              >
                [{{ src.index }}] {{ src.filename
                }}<template v-if="src.page_number && src.page_number !== 'N/A'">
                  · P{{ src.page_number }}</template
                >
              </a>
              <span v-else class="agent-chat-source-chip" :data-kcite-chip="src.index">
                [{{ src.index }}] {{ src.filename
                }}<template v-if="src.page_number && src.page_number !== 'N/A'">
                  · P{{ src.page_number }}</template
                >
              </span>
            </template>
          </div>
        </div>
        <div
          v-if="
            !message.pending &&
            ((message.content || '').trim() ||
              (message.errorText || '').trim() ||
              message.stoppedByUser)
          "
          class="agent-chat-assistant-actions"
        >
          <span v-if="(message.versionCount || 1) > 1" class="agent-chat-version-switch">
            <n-button
              quaternary
              circle
              size="small"
              class="agent-chat-copy-btn"
              :disabled="sending || switchingBranch || (message.versionIndex || 1) <= 1"
              :aria-label="$t('views.agents.chat_version_prev_tooltip')"
              @click="emit('switch-version', message, -1)"
            >
              <TheIcon icon="mdi:chevron-left" :size="16" />
            </n-button>
            <span class="agent-chat-version-label">
              {{ message.versionIndex || 1 }}/{{ message.versionCount }}
            </span>
            <n-button
              quaternary
              circle
              size="small"
              class="agent-chat-copy-btn"
              :disabled="
                sending ||
                switchingBranch ||
                (message.versionIndex || 1) >= (message.versionCount || 1)
              "
              :aria-label="$t('views.agents.chat_version_next_tooltip')"
              @click="emit('switch-version', message, 1)"
            >
              <TheIcon icon="mdi:chevron-right" :size="16" />
            </n-button>
          </span>
          <n-tooltip :show-arrow="false" placement="top">
            <template #trigger>
              <n-button
                quaternary
                circle
                size="small"
                class="agent-chat-copy-btn"
                :disabled="sending || switchingBranch || !message.messageId"
                :aria-label="$t('views.agents.chat_regenerate_tooltip')"
                @click="emit('regenerate', message)"
              >
                <TheIcon icon="mdi:refresh" :size="18" />
              </n-button>
            </template>
            {{ $t('views.agents.chat_regenerate_tooltip') }}
          </n-tooltip>
          <n-tooltip :show-arrow="false" placement="top">
            <template #trigger>
              <n-button
                quaternary
                circle
                size="small"
                class="agent-chat-copy-btn"
                :aria-label="$t('views.agents.chat_copy_plain_tooltip')"
                @click="emit('copy-plain', message)"
              >
                <TheIcon icon="lucide:copy" :size="18" />
              </n-button>
            </template>
            {{ $t('views.agents.chat_copy_plain_tooltip') }}
          </n-tooltip>
          <n-tooltip :show-arrow="false" placement="top">
            <template #trigger>
              <n-button
                quaternary
                circle
                size="small"
                class="agent-chat-copy-btn"
                :aria-label="$t('views.agents.chat_copy_md_tooltip')"
                @click="emit('copy-md', message)"
              >
                <TheIcon icon="simple-icons:markdown" :size="18" />
              </n-button>
            </template>
            {{ $t('views.agents.chat_copy_md_tooltip') }}
          </n-tooltip>
        </div>
        <div v-if="message.attachments?.length" class="agent-chat-attachment-boxes">
          <ChatAttachmentItem
            v-for="(a, ai) in message.attachments"
            :key="`aa-${message.id}-${ai}`"
            :agent-id="chatAgentId"
            :session-id="sessionId"
            :attachment="a"
          />
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { NAvatar, NButton, NTooltip } from 'naive-ui'
import TheIcon from '@/components/icon/TheIcon.vue'
import ChatAttachmentItem from './ChatAttachmentItem.vue'
import {
  renderAgentChatMarkdown,
  safeExternalHref,
  toSameOriginMediaUrl,
} from '@/utils/agentChatMarkdown'
import { thinkingPillLabel, todoMark, todoProgress } from '@/utils/agentChatThinking'

const props = defineProps({
  message: { type: Object, required: true },
  chatAgentId: { type: Number, default: 0 },
  sessionId: { type: String, default: '' },
  agentAvatarSrc: { type: String, default: '' },
  agentName: { type: String, default: '—' },
  sending: { type: Boolean, default: false },
  switchingBranch: { type: Boolean, default: false },
  approvingIds: { type: Object, default: () => new Set() },
})

const emit = defineEmits([
  'toggle-thinking',
  'approval-decide',
  'md-click',
  'switch-version',
  'regenerate',
  'copy-plain',
  'copy-md',
])

/**
 * 审批卡片里的参数预览。
 *
 * 后端 `actions[].args` 是原样的对象/数组，直接拼进模板会显示成 [object Object]。
 * 截断到 800 字符与 MCP 卡片的 `args_preview[:800]` 保持一致，避免大段文件内容撑爆界面。
 */
function formatApprovalArgs(args) {
  if (args == null) return ''
  let text
  try {
    text = typeof args === 'string' ? args : JSON.stringify(args, null, 2)
  } catch {
    text = String(args)
  }
  if (!text) return ''
  return text.length > 800 ? `${text.slice(0, 800)}…` : text
}

// 流式行渲染降频：renderVersion 由 useChatJobStream 按 ~100ms 窗口 bump，
// displayContent 是 bump 时刻的内容快照；历史行没有这两个字段，直接依赖 content
const renderedContent = computed(() => {
  // 显式读取建立依赖：快照路径下不再追踪 content，避免每 token 重新解析 markdown
  void props.message.renderVersion
  return renderAgentChatMarkdown(
    props.message.displayContent ?? props.message.content,
    props.message.sources
  )
})
</script>

<style src="./agent-chat-ui.css"></style>
<style>
.agent-chat-feed-item {
  max-width: var(--agent-chat-column-max, 880px);
  margin-left: auto;
  margin-right: auto;
  padding-bottom: 28px;
}

.agent-chat-version-switch {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  margin-right: 4px;
}

.agent-chat-version-label {
  font-size: 12px;
  color: var(--n-text-color-3);
  min-width: 30px;
  text-align: center;
  user-select: none;
}

.agent-chat-mcp-confirm {
  margin-bottom: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.agent-chat-mcp-confirm-item {
  border: 1px solid rgba(245, 158, 11, 0.45);
  background: rgba(245, 158, 11, 0.08);
  border-radius: 8px;
  padding: 10px 12px;
  box-sizing: border-box;
}

.agent-chat-mcp-confirm-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--n-warning-color);
  margin-bottom: 4px;
}

.agent-chat-mcp-confirm-text {
  font-size: 13px;
  margin-bottom: 6px;
}

.agent-chat-mcp-confirm-args {
  margin: 0 0 8px;
  max-height: 160px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  color: var(--n-text-color-3);
}

.agent-chat-mcp-confirm-actions {
  display: flex;
  gap: 8px;
}

/* 审批卡片：与 MCP 确认卡片同色系（琥珀警示），避免用户误以为是两种不同的东西 */
.agent-chat-approval {
  margin-bottom: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.agent-chat-approval-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--n-warning-color);
}

.agent-chat-approval-item {
  border: 1px solid rgba(245, 158, 11, 0.45);
  background: rgba(245, 158, 11, 0.08);
  border-radius: 8px;
  padding: 10px 12px;
  box-sizing: border-box;
}

.agent-chat-approval-name {
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 6px;
}

.agent-chat-approval-args {
  margin: 0;
  max-height: 160px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  color: var(--n-text-color-3);
}

.agent-chat-approval-actions {
  display: flex;
  gap: 8px;
}

.agent-chat-approval-hint {
  font-size: 12px;
  color: var(--n-text-color-3);
}

.agent-chat-sources {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}

.agent-chat-sources-label {
  font-size: 12px;
  color: #64748b;
  margin-right: 2px;
}

.agent-chat-source-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 10px;
  border-radius: 999px;
  background: #eef2f7;
  color: #334155;
  font-size: 12px;
  line-height: 1.7;
  text-decoration: none;
  border: 1px solid #e2e8f0;
  scroll-margin: 8px;
}

a.agent-chat-source-chip:hover {
  border-color: #94a3b8;
  color: #0f172a;
}

html.dark .agent-chat-sources-label {
  color: #94a3b8;
}

html.dark .agent-chat-source-chip {
  background: rgba(148, 163, 184, 0.14);
  border-color: rgba(148, 163, 184, 0.28);
  color: #cbd5e1;
}

html.dark a.agent-chat-source-chip:hover {
  border-color: #cbd5e1;
  color: #f1f5f9;
}
</style>
