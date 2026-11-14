import { lStorage, sStorage } from '@/utils/storage'

const TOKEN_CODE = 'access_token'

/**
 * 提前续期窗口。
 * 不等到 exp 那一刻才刷：请求从点击到服务端校验有网络往返，
 * 卡在边界上会出现「本地没过期、服务端已过期」的 401。
 */
const REFRESH_SKEW_MS = 60 * 1000

function migrateLegacyToken() {
  if (sStorage.get(TOKEN_CODE)) {
    lStorage.remove(TOKEN_CODE)
    return
  }
  const old = lStorage.get(TOKEN_CODE)
  if (old) {
    sStorage.set(TOKEN_CODE, old)
    lStorage.remove(TOKEN_CODE)
  }
}

export function getToken() {
  migrateLegacyToken()
  return sStorage.get(TOKEN_CODE)
}

export function setToken(token) {
  sStorage.set(TOKEN_CODE, token)
  lStorage.remove(TOKEN_CODE)
}

export function removeToken() {
  sStorage.remove(TOKEN_CODE)
  lStorage.remove(TOKEN_CODE)
}

let refreshing = null

/** 单次刷新请求：成功则落盘并返回新 token，失败返回 null */
async function doRefresh() {
  try {
    const base = import.meta.env.VITE_BASE_API || '/api/v1'
    const res = await fetch(`${base}/base/refresh`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
    })
    const data = await res.json().catch(() => null)
    const tok = data?.data?.access_token
    if (data?.code === 200 && tok) {
      setToken(tok)
      return tok
    }
    return null
  } catch {
    return null
  }
}

export async function tryRefreshToken() {
  if (refreshing) return refreshing
  refreshing = (async () => {
    try {
      const tok = await doRefresh()
      if (tok) return tok
      /**
       * 并发兜底：refresh token 是一次性消费的，多标签页同时刷新时，
       * 后到的请求拿的是已被上一个请求消费掉的 jti，必然 401。
       * cookie 是浏览器级共享的——另一个标签页成功轮换出的新 jti 此刻已经落到本标签页，
       * 隔一拍重试即可成功。凭证真的失效时，这里只是多一次无谓请求。
       */
      await new Promise((resolve) => setTimeout(resolve, 300))
      return await doRefresh()
    } finally {
      refreshing = null
    }
  })()
  return refreshing
}

/**
 * 解 JWT payload（只取 exp，不验签）。
 * 必须走 TextDecoder 而不是直接 JSON.parse(atob())：
 * payload 里有中文 username，atob 返回的是 latin1，
 * 直接 parse 会因多字节字符断裂而抛 SyntaxError，进而被误判成过期。
 */
function decodeJwtPayload(token) {
  const seg = String(token || '').split('.')[1]
  if (!seg) return null
  const b64 = seg.replace(/-/g, '+').replace(/_/g, '/')
  const padded = b64 + '='.repeat((4 - (b64.length % 4)) % 4)
  const bin = atob(padded)
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0))
  return JSON.parse(new TextDecoder().decode(bytes))
}

/**
 * access token 是否已过期或即将过期。
 * 解不开/没有 exp 一律按过期处理——宁可多刷一次，也不要拿可疑 token 去撞 401。
 */
export function isTokenExpiring(token, skewMs = REFRESH_SKEW_MS) {
  if (!token) return true
  let payload
  try {
    payload = decodeJwtPayload(token)
  } catch {
    return true
  }
  if (!payload?.exp) return true
  return payload.exp * 1000 - Date.now() < skewMs
}

/**
 * 拿一个「现在就能用」的 access token，续不上返回 null。
 *
 * 为什么需要它：`getToken()` 只读 storage，不管 exp。axios 那条链有 401 拦截器兜底，
 * 但聊天 / 附件 / 工作区下载走的是原生 fetch，历史上没有接进刷新流程——
 * 页面停留超过 access token 有效期后，第一次 fetch 就会直接 401。
 * 所有原生 fetch 入口在发请求前都应先经过这里。
 */
export async function ensureFreshToken() {
  const token = getToken()
  if (!isTokenExpiring(token)) return token
  return (await tryRefreshToken()) || null
}
