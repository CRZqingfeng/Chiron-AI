<template>
  <n-drawer v-model:show="visible" :width="440" placement="right">
    <n-drawer-content closable>
      <template #header>
        <div class="ws-title">
          <TheIcon icon="mdi:folder-text-outline" :size="20" />
          <span>{{ t('views.agents.chat_files_title') }}</span>
          <n-tag v-if="items.length && !preview" size="tiny" :bordered="false">
            {{ items.length }}
          </n-tag>
        </div>
      </template>

      <!-- 预览态：抽屉内切换，不新开弹窗——否则叠两层遮罩，返回还得点两次 -->
      <template v-if="preview">
        <div class="ws-preview-bar">
          <n-button size="tiny" quaternary @click="closePreview">
            <TheIcon icon="mdi:arrow-left" :size="14" />
            {{ t('views.agents.chat_files_back') }}
          </n-button>
          <code class="ws-preview-path">{{ preview.path }}</code>
          <n-button
            size="tiny"
            quaternary
            type="primary"
            @click="downloadByPath(preview.path, baseName(preview.path))"
          >
            <TheIcon icon="mdi:download" :size="14" />
          </n-button>
        </div>
        <n-alert v-if="preview.truncated" type="warning" :bordered="false" mt-8>
          {{ t('views.agents.chat_files_truncated') }}
        </n-alert>
        <pre class="ws-preview-body">{{ preview.content }}</pre>
      </template>

      <!-- 列表态 -->
      <template v-else>
        <p class="ws-hint">{{ t('views.agents.chat_files_hint') }}</p>

        <div v-if="currentPath" class="ws-crumb">
          <n-button size="tiny" quaternary @click="goUp">
            <TheIcon icon="mdi:arrow-up-left" :size="14" />
            {{ t('views.agents.chat_files_up') }}
          </n-button>
          <code>{{ currentPath }}</code>
        </div>

        <n-spin :show="loading">
          <div v-if="loadError" class="ws-empty">
            <div class="ws-empty-title">{{ t('views.agents.chat_files_load_failed') }}</div>
            <n-button size="small" quaternary type="primary" mt-12 @click="load">
              {{ t('views.agents.chat_files_retry') }}
            </n-button>
          </div>

          <!-- 空态要说清「为什么空」：刚开的会话本来就没有产出 -->
          <div v-else-if="!items.length" class="ws-empty">
            <div class="ws-empty-title">{{ t('views.agents.chat_files_empty') }}</div>
            <div class="ws-empty-desc">{{ t('views.agents.chat_files_empty_hint') }}</div>
          </div>

          <ul v-else class="ws-list">
            <li v-for="it in items" :key="it.path" class="ws-item">
              <button class="ws-item-main" type="button" @click="onOpen(it)">
                <TheIcon
                  :icon="it.is_dir ? 'mdi:folder-outline' : fileIcon(it.name)"
                  :size="18"
                  class="ws-item-icon"
                />
                <span class="ws-item-name">{{ it.name }}</span>
                <span class="ws-item-meta">{{ it.is_dir ? '' : formatSize(it.size) }}</span>
              </button>
              <n-button
                v-if="!it.is_dir"
                size="tiny"
                quaternary
                :loading="downloading === it.path"
                :title="t('views.agents.chat_files_download')"
                @click.stop="downloadByPath(it.path, it.name)"
              >
                <TheIcon icon="mdi:download" :size="15" />
              </n-button>
            </li>
          </ul>
        </n-spin>
      </template>
    </n-drawer-content>
  </n-drawer>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { NAlert, NButton, NDrawer, NDrawerContent, NSpin, NTag, useMessage } from 'naive-ui'
import TheIcon from '@/components/icon/TheIcon.vue'
import api from '@/api'

const props = defineProps({
  show: { type: Boolean, default: false },
  agentId: { type: Number, default: 0 },
  sessionId: { type: String, default: '' },
})
const emit = defineEmits(['update:show'])

const { t } = useI18n()
const message = useMessage()

const visible = computed({
  get: () => props.show,
  set: (v) => emit('update:show', v),
})

const items = ref([])
const loading = ref(false)
const loadError = ref(false)
const currentPath = ref('')
const preview = ref(null)
const downloading = ref('')
// 只在打开时拉：文件列表是低频数据，没必要跟着对话流一直查
let loadedFor = ''

const TEXT_RE =
  /\.(md|markdown|txt|rst|log|csv|tsv|json|jsonl|ya?ml|toml|ini|cfg|conf|env|py|js|mjs|cjs|ts|tsx|jsx|vue|html?|css|scss|less|sh|bash|bat|ps1|sql|java|go|rs|c|h|cpp|hpp|rb|php|kt|xml|svg)$/i

function fileIcon(name) {
  if (TEXT_RE.test(name)) return 'mdi:file-document-outline'
  return 'mdi:file-outline'
}

function baseName(p) {
  return (
    String(p || '')
      .split('/')
      .pop() || p
  )
}

function formatSize(size) {
  const n = Number(size) || 0
  if (n < 1024) return `${n}B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`
  return `${(n / 1024 / 1024).toFixed(1)}MB`
}

async function load() {
  if (!props.agentId || !props.sessionId) return
  loading.value = true
  loadError.value = false
  try {
    const res = await api.listAgentWorkspaceFiles({
      agent_id: props.agentId,
      session_id: props.sessionId,
      path: currentPath.value,
    })
    // 后端统一包一层 { code, msg, data }；code 非 200 时必须走失败态，
    // 否则会把错误响应当成「空列表」——那是最难分辨的一种假成功
    if (res?.code !== undefined && res.code !== 200 && res.code !== 0) {
      throw new Error(res?.msg || 'load failed')
    }
    items.value = res?.data?.items || []
  } catch {
    loadError.value = true
    items.value = []
  } finally {
    loading.value = false
  }
}

function onOpen(it) {
  if (it.is_dir) {
    currentPath.value = it.path
    load()
    return
  }
  if (TEXT_RE.test(it.name)) {
    openPreview(it.path)
    return
  }
  downloadByPath(it.path, it.name)
}

async function openPreview(path) {
  try {
    const res = await api.previewAgentWorkspaceFile({
      agent_id: props.agentId,
      session_id: props.sessionId,
      path,
    })
    if (res?.code !== undefined && res.code !== 200 && res.code !== 0) {
      // 不支持预览的类型（后端 400）直接转下载，别让用户卡在错误提示上
      message.warning(res?.msg || t('views.agents.chat_files_preview_unsupported'))
      downloadByPath(path, baseName(path))
      return
    }
    preview.value = res.data
  } catch {
    message.error(t('views.agents.chat_files_load_failed'))
  }
}

function closePreview() {
  preview.value = null
}

function goUp() {
  const parts = currentPath.value.split('/').filter(Boolean)
  parts.pop()
  currentPath.value = parts.join('/')
  load()
}

async function downloadByPath(path, name) {
  downloading.value = path
  try {
    const blob = await api.fetchAgentWorkspaceFileBlob({
      agent_id: props.agentId,
      session_id: props.sessionId,
      path,
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name || baseName(path)
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    // 立刻 revoke 在部分浏览器会让下载变成 0 字节，延后一拍
    setTimeout(() => URL.revokeObjectURL(url), 2000)
  } catch {
    message.error(t('views.agents.chat_files_download_failed'))
  } finally {
    downloading.value = ''
  }
}

// 打开时拉一次；同一会话重复打开不重复拉，除非切换了会话或 agent
watch(
  () => [props.show, props.agentId, props.sessionId],
  ([show]) => {
    if (!show) return
    const key = `${props.agentId}:${props.sessionId}`
    if (key === loadedFor) return
    loadedFor = key
    currentPath.value = ''
    preview.value = null
    load()
  },
  { immediate: true }
)
</script>

<style scoped>
.ws-title {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ws-hint {
  margin: 0 0 12px;
  font-size: 13px;
  line-height: 1.6;
  opacity: 0.72;
}
.ws-crumb {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 12px;
  opacity: 0.85;
}
.ws-crumb code {
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(128, 128, 128, 0.14);
  font-size: 12px;
}
.ws-empty {
  padding: 40px 12px;
  text-align: center;
}
.ws-empty-title {
  font-size: 14px;
  font-weight: 500;
}
.ws-empty-desc {
  margin-top: 8px;
  font-size: 12.5px;
  line-height: 1.65;
  opacity: 0.68;
}
.ws-list {
  margin: 0;
  padding: 0;
  list-style: none;
}
.ws-item {
  display: flex;
  align-items: center;
  gap: 6px;
  border-radius: 6px;
  padding: 2px 4px;
}
.ws-item:hover {
  background: rgba(128, 128, 128, 0.1);
}
.ws-item-main {
  display: flex;
  flex: 1;
  align-items: center;
  gap: 10px;
  min-width: 0;
  padding: 8px 6px;
  border: none;
  background: transparent;
  color: inherit;
  cursor: pointer;
  text-align: left;
  font-size: 13.5px;
}
.ws-item-icon {
  flex: none;
  opacity: 0.8;
}
.ws-item-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ws-item-meta {
  flex: none;
  font-size: 12px;
  opacity: 0.55;
}
.ws-preview-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.ws-preview-path {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
  opacity: 0.75;
}
.ws-preview-body {
  max-height: calc(100vh - 220px);
  margin: 0;
  padding: 12px;
  overflow: auto;
  border-radius: 8px;
  background: rgba(128, 128, 128, 0.1);
  font-size: 12.5px;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
