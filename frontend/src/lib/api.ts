import type {
  DoneEvent,
  ErrorEvent,
  ImageEvent,
  SSEEvent,
  TextDeltaEvent,
} from './types'

export interface StreamCallbacks {
  onTextDelta: (event: TextDeltaEvent) => void
  onImage: (event: ImageEvent) => void
  onDone: (event: DoneEvent) => void
  onError: (event: ErrorEvent) => void
}

export interface StreamChatOptions {
  voiceMode?: boolean
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function apiUrl(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : path
}

export async function fetchHealth(): Promise<{ local_tts_ready?: boolean; local_tts_enabled?: boolean; hosted_demo?: boolean }> {
  const response = await fetch(apiUrl('/health'))
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return (await response.json()) as { local_tts_ready?: boolean; local_tts_enabled?: boolean; hosted_demo?: boolean }
}

export async function synthesizeSpeech(text: string): Promise<string> {
  const response = await fetch(apiUrl('/speech'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })

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
  const response = await fetch(apiUrl('/chat'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, voice_mode: Boolean(options.voiceMode) }),
  })

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
      else if (event.type === 'done') callbacks.onDone(event)
      else if (event.type === 'error') callbacks.onError(event)
    }
  }
}
