<template>
  <n-modal
    v-model:show="visible"
    preset="card"
    :style="{ width: '680px' }"
    size="huge"
    :bordered="false"
    :mask-closable="true"
  >
    <template #header>
      <div class="mem-title">
        <TheIcon icon="mdi:brain" :size="20" />
        <span>{{ $t('views.agents.chat_memory_title') }}</span>
        <n-tag v-if="items.length" size="tiny" :bordered="false">{{ items.length }}</n-tag>
      </div>
    </template>

    <p class="mem-hint">{{ $t('views.agents.chat_memory_hint') }}</p>

    <n-spin :show="loading">
      <div v-if="loadError" class="mem-empty">
        <div class="mem-empty-title">{{ $t('views.agents.chat_memory_load_failed') }}</div>
        <n-button size="small" quaternary type="primary" mt-12 @click="load">
          {{ $t('views.agents.chat_memory_retry') }}
        </n-button>
      </div>

      <!-- 空态要说清「为什么空」，否则用户会以为功能坏了：
           记忆要攒够 5 轮才提炼一次，新智能体在新会话里本来就是空的 -->
      <div v-else-if="!items.length" class="mem-empty">
        <div class="mem-empty-title">{{ $t('views.agents.chat_memory_empty') }}</div>
        <div class="mem-empty-desc">{{ $t('views.agents.chat_memory_empty_hint') }}</div>
      </div>

      <ul v-else class="mem-list">
        <li v-for="it in items" :key="it.id" class="mem-item">
          <div class="mem-item-main">
            <div class="mem-item-head">
              <n-tag size="tiny" :bordered="false" :type="kindTagType(it.kind)">
                {{ kindLabel(it.kind) }}
              </n-tag>
              <code class="mem-item-key">{{ it.mem_key }}</code>
              <span class="mem-item-meta">
                {{ $t('views.agents.chat_memory_confidence', { n: confidencePct(it) }) }}
                <template v-if="it.updated_at"> · {{ formatTime(it.updated_at) }} </template>
              </span>
            </div>
            <div class="mem-item-text">{{ it.text }}</div>
          </div>
          <n-button
            size="tiny"
            quaternary
            type="error"
            :loading="deletingId === it.id"
            @click="onDelete(it)"
          >
            {{ $t('common.buttons.delete') }}
          </n-button>
        </li>
      </ul>
    </n-spin>

    <template #footer>
      <div flex justify-end>
        <n-button size="small" @click="visible = false">
          {{ $t('views.agents.chat_memory_close') }}
        </n-button>
      </div>
    </template>
  </n-modal>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton, NModal, NSpin, NTag, useDialog, useMessage } from 'naive-ui'
import TheIcon from '@/components/icon/TheIcon.vue'
import api from '@/api'

const props = defineProps({
  show: { type: Boolean, default: false },
  agentId: { type: Number, default: 0 },
})
const emit = defineEmits(['update:show'])

const { t, locale } = useI18n()
const message = useMessage()
const dialog = useDialog()

const visible = computed({
  get: () => props.show,
  set: (v) => emit('update:show', v),
})

const items = ref([])
const loading = ref(false)
const loadError = ref(false)
const deletingId = ref(0)
// 只在面板打开时拉取：记忆是低频数据，没必要在对话页生命周期里一直查
let loadedForAgent = 0

const KIND_TAG = {
  fact: 'default',
  preference: 'info',
  convention: 'success',
  decision: 'warning',
}

function kindLabel(kind) {
  const key = `views.agents.chat_memory_kind_${kind}`
  const label = t(key)
  // 后端未来可能新增 kind，缺键时回退显示原始值，而不是露出 i18n 的 key 路径
  return label === key ? kind || '-' : label
}

function kindTagType(kind) {
  return KIND_TAG[kind] || 'default'
}

function confidencePct(it) {
  return Math.round((Number(it?.confidence) || 0) * 100)
}

function parseServerTime(iso) {
  // 后端 `updated_at` 是 `datetime.utcnow().isoformat()`——**naive UTC，没有 Z**。
  // 直接 `new Date(iso)` 会被 JS 当成本地时间，界面就差 8 小时（06:34 显示成 06:34
  // 而不是 14:34）。带 Z/偏移量就按原样，没有就补 Z 明确按 UTC 解析。
  const hasTz = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(iso || '')
  return new Date(hasTz ? iso : `${iso}Z`)
}

function formatTime(iso) {
  const d = parseServerTime(iso)
  if (Number.isNaN(d.getTime())) return ''
  return new Intl.DateTimeFormat(locale.value === 'en' ? 'en-US' : 'zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(d)
}

async function load() {
  if (!props.agentId) return
  loading.value = true
  loadError.value = false
  try {
    const res = await api.getAgentMemories(props.agentId)
    // 后端统一包一层 { code, msg, data }；code 非 200 时这里也要走失败态，
    // 否则会把错误响应当成「空列表」——那是最难分辨的一种假成功
    if (res?.code !== undefined && res.code !== 200 && res.code !== 0) {
      throw new Error(res?.msg || 'load failed')
    }
    items.value = res?.data?.items || []
    loadedForAgent = props.agentId
  } catch (e) {
    loadError.value = true
    items.value = []
    console.error('[memory] 加载失败', e)
  } finally {
    loading.value = false
  }
}

function onDelete(item) {
  dialog.warning({
    title: t('views.agents.chat_memory_delete_title'),
    content: t('views.agents.chat_memory_delete_confirm', { text: item.text }),
    positiveText: t('common.buttons.delete'),
    negativeText: t('views.agents.chat_memory_cancel'),
    onPositiveClick: async () => {
      deletingId.value = item.id
      try {
        const res = await api.deleteAgentMemory(props.agentId, item.id)
        if (res?.code !== undefined && res.code !== 200 && res.code !== 0) {
          throw new Error(res?.msg || 'delete failed')
        }
        items.value = items.value.filter((x) => x.id !== item.id)
        message.success(t('views.agents.chat_memory_deleted'))
      } catch (e) {
        // 删除失败必须让用户知道：他会以为已经删掉了，而记忆还在影响后续回答
        message.error(t('views.agents.chat_memory_delete_failed'))
        console.error('[memory] 删除失败', e)
      } finally {
        deletingId.value = 0
      }
    },
  })
}

watch(
  () => props.show,
  (open) => {
    if (!open) return
    // 换了智能体才重新拉：同一个智能体反复开关不必重复请求
    if (loadedForAgent !== props.agentId || loadError.value) load()
  }
)

watch(
  () => props.agentId,
  () => {
    items.value = []
    loadedForAgent = 0
  }
)
</script>

<style scoped>
.mem-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
}

.mem-hint {
  margin: 0 0 12px;
  font-size: 12px;
  line-height: 1.6;
  opacity: 0.65;
}

.mem-empty {
  padding: 28px 12px;
  text-align: center;
}

.mem-empty-title {
  font-size: 14px;
  font-weight: 500;
}

.mem-empty-desc {
  margin-top: 8px;
  font-size: 12px;
  line-height: 1.7;
  opacity: 0.6;
}

.mem-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 52vh;
  overflow-y: auto;
}

.mem-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid var(--n-border-color, rgba(128, 128, 128, 0.22));
}

.mem-item-main {
  flex: 1;
  min-width: 0;
}

.mem-item-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.mem-item-key {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(128, 128, 128, 0.14);
}

.mem-item-meta {
  font-size: 11px;
  opacity: 0.55;
}

.mem-item-text {
  margin-top: 6px;
  font-size: 13px;
  line-height: 1.65;
  word-break: break-word;
}
</style>
