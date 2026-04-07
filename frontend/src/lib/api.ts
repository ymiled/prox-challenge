import type {
  ArtifactEvent,
  DoneEvent,
  ErrorEvent,
  ImageEvent,
  SSEEvent,
  TextDeltaEvent,
} from './types'

export interface StreamCallbacks {
  onTextDelta: (event: TextDeltaEvent) => void
  onImage: (event: ImageEvent) => void
  onArtifact: (event: ArtifactEvent) => void
  onDone: (event: DoneEvent) => void
  onError: (event: ErrorEvent) => void
}

export interface StreamChatOptions {
  voiceMode?: boolean
  anthropicApiKey?: string
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
const isBrowser = typeof window !== 'undefined'

export function apiUrl(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : path
}

function buildNetworkError(path: string, error: unknown): Error {
  const target = apiUrl(path)
  const pageOrigin = isBrowser ? window.location.origin : 'unknown origin'
  const protocolMismatch =
    isBrowser &&
    window.location.protocol === 'https:' &&
    /^http:\/\//i.test(target)

  const details = [
    `Could not reach ${target}.`,
    `Page origin: ${pageOrigin}.`,
    API_BASE
      ? `Configured API base: ${API_BASE}.`
      : 'No VITE_API_BASE_URL is configured, so the frontend is calling its own origin.',
  ]

  if (protocolMismatch) {
    details.push('The site is loaded over HTTPS but the API base uses HTTP, which browsers block as mixed content.')
  } else {
    details.push('Common causes: wrong VITE_API_BASE_URL, CORS not allowing the frontend origin, backend unavailable, or an invalid SSL certificate.')
  }

  if (error instanceof Error && error.message) {
    details.push(`Browser error: ${error.message}`)
  }

  return new Error(details.join(' '))
}

export async function fetchHealth(): Promise<{
  local_tts_ready?: boolean
  local_tts_enabled?: boolean
  deployment_env?: string
  anthropic_enabled?: boolean
}> {
  let response: Response
  try {
    response = await fetch(apiUrl('/health'))
  } catch (error) {
    throw buildNetworkError('/health', error)
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return (await response.json()) as {
    local_tts_ready?: boolean
    local_tts_enabled?: boolean
    deployment_env?: string
    anthropic_enabled?: boolean
  }
}

export async function validateAnthropicKey(key: string): Promise<{ valid: boolean; error?: string }> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 6000)
  try {
    const response = await fetch(apiUrl('/validate-key'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ anthropic_api_key: key }),
      signal: controller.signal,
    })
    return (await response.json()) as { valid: boolean; error?: string }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      return { valid: false, error: 'Validation timed out. Check your connection and try again.' }
    }
    return { valid: false, error: 'Could not reach the server. Check your connection and try again.' }
  } finally {
    clearTimeout(timeout)
  }
}

export async function synthesizeSpeech(text: string): Promise<string> {
  let response: Response
  try {
    response = await fetch(apiUrl('/speech'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
  } catch (error) {
    throw buildNetworkError('/speech', error)
  }

  const payload = (await response.json()) as { status?: string; url?: string; message?: string }
  if (!response.ok || payload.status !== 'ok' || !payload.url) {
    throw new Error(payload.message || `HTTP ${response.status}`)
  }
  return apiUrl(payload.url)
}

export async function streamChat(
  message: string,
  history: Array<{ role: string; content: string }>,
  callbacks: StreamCallbacks,
  options: StreamChatOptions = {}
): Promise<void> {
  let response: Response
  try {
    response = await fetch(apiUrl('/chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        history,
        voice_mode: Boolean(options.voiceMode),
        anthropic_api_key: options.anthropicApiKey?.trim() || undefined,
      }),
    })
  } catch (error) {
    callbacks.onError({ type: 'error', message: buildNetworkError('/chat', error).message })
    return
  }

  if (!response.ok) {
    const text = await response.text()
    callbacks.onError({ type: 'error', message: `HTTP ${response.status}: ${text}` })
    return
  }

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (!raw) continue

      let event: SSEEvent
      try {
        event = JSON.parse(raw) as SSEEvent
      } catch {
        continue
      }

      if (event.type === 'text_delta') callbacks.onTextDelta(event)
      else if (event.type === 'image') callbacks.onImage(event)
      else if (event.type === 'artifact') callbacks.onArtifact(event)
      else if (event.type === 'done') callbacks.onDone(event)
      else if (event.type === 'error') callbacks.onError(event)
    }
  }
}
