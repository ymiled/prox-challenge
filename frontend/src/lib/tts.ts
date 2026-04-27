import { apiUrl, getCsrfToken } from './api'

/**
 * Pulls complete sentences from the front of an accumulator string.
 * Anything at the tail that hasn't been closed yet stays in `remaining`
 * so the next delta can complete it, enabling sentence-level streaming TTS.
 */
export function drainSentences(buffer: string): { sentences: string[]; remaining: string } {
  const sentences: string[] = []
  let start = 0
  let cursor = 0

  while (cursor < buffer.length) {
    const char = buffer[cursor]
    if (!/[.!?]/.test(char)) {
      cursor += 1
      continue
    }

    let punctEnd = cursor + 1
    while (punctEnd < buffer.length && /[.!?]/.test(buffer[punctEnd])) punctEnd += 1

    let gapEnd = punctEnd
    while (gapEnd < buffer.length && /\s/.test(buffer[gapEnd])) gapEnd += 1

    if (gapEnd === punctEnd) {
      cursor = punctEnd
      continue
    }

    const nextChar = buffer[gapEnd]
    const shouldSplit =
      nextChar === undefined ||
      nextChar === '\n' ||
      /["'([{A-Z0-9]/.test(nextChar)

    if (!shouldSplit) {
      cursor = punctEnd
      continue
    }

    const sentence = buffer.slice(start, punctEnd).trim()
    if (sentence.length > 2) sentences.push(sentence)
    start = gapEnd
    cursor = gapEnd
  }

  return { sentences, remaining: buffer.slice(start) }
}

type Task = () => Promise<void>

/**
 * Queues sentences and plays them sequentially using Deepgram Aura TTS.
 *
 * Uses the Web Audio API (AudioContext + decodeAudioData) when a pre-unlocked
 * AudioContext is provided. This bypasses browser autoplay restrictions that
 * block HTMLAudioElement.play() in async callbacks — the root cause of "must
 * press Speak manually in production". Falls back to HTMLAudioElement only
 * when no AudioContext is available.
 */
export class TTSPlayer {
  private queue: Task[] = []
  private processing = false
  private stopped = false
  private currentSource: AudioBufferSourceNode | null = null
  private currentAudio: HTMLAudioElement | null = null
  private firstStarted = false

  constructor(
    private readonly onPlayStart?: () => void,
    private readonly onPlayEnd?: () => void,
    private readonly onError?: (message: string) => void,
    private readonly deepgramApiKey?: string,
    // Provide a getter so callers can pass the ref value at play-time,
    // not at construction-time (the context may not exist yet at construction).
    private readonly getAudioContext?: () => AudioContext | null,
  ) {}

  enqueue(text: string): void {
    if (this.stopped || !text.trim()) return
    this.queue.push(() => this.fetchAndPlay(text))
    if (!this.processing) void this.drain()
  }

  private async drain(): Promise<void> {
    this.processing = true
    this.firstStarted = false
    while (this.queue.length > 0 && !this.stopped) {
      const task = this.queue.shift()!
      try {
        await task()
      } catch {
        // one failed sentence must not silence the rest
      }
    }
    this.processing = false
    if (!this.stopped) this.onPlayEnd?.()
  }

  private async fetchAndPlay(text: string): Promise<void> {
    if (this.stopped) return

    const response = await fetch(apiUrl('/speech/stream'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': getCsrfToken(),
      },
      body: JSON.stringify({ text, deepgram_api_key: this.deepgramApiKey?.trim() || undefined }),
      credentials: 'include',
    })

    if (!response.ok) {
      let message = `TTS HTTP ${response.status}`
      try {
        const payload = (await response.json()) as { error?: string }
        if (payload.error) message = payload.error
      } catch {
        // keep fallback message
      }
      this.onError?.(message)
      throw new Error(message)
    }

    const arrayBuffer = await response.arrayBuffer()
    if (this.stopped) return

    // Prefer Web Audio API — once an AudioContext is resumed via a user gesture
    // it can decode+play audio freely without further gesture requirements,
    // which is exactly what we need for auto-speak in async callbacks.
    const ctx = this.getAudioContext?.()
    if (ctx && ctx.state !== 'closed') {
      await this.playWithAudioContext(ctx, arrayBuffer)
    } else {
      await this.playWithHtmlAudio(arrayBuffer)
    }
  }

  private async playWithAudioContext(ctx: AudioContext, arrayBuffer: ArrayBuffer): Promise<void> {
    // Do NOT call ctx.resume() here — this is an async callback (outside any
    // user gesture), so Safari HTTPS will reject it silently. The app-layer
    // unlock effect keeps the context running via synchronous gesture handlers.
    // If the context is still suspended, fall back to HTMLAudioElement.
    if (ctx.state === 'suspended') {
      await this.playWithHtmlAudio(arrayBuffer)
      return
    }

    let audioBuffer: AudioBuffer
    try {
      // decodeAudioData consumes the buffer; pass a copy so the original is untouched
      audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0))
    } catch {
      // Format unsupported in this browser — fall back to HTML Audio
      await this.playWithHtmlAudio(arrayBuffer)
      return
    }

    if (this.stopped) return

    await new Promise<void>((resolve) => {
      if (this.stopped) { resolve(); return }

      const source = ctx.createBufferSource()
      source.buffer = audioBuffer
      source.connect(ctx.destination)
      this.currentSource = source

      source.onended = () => {
        if (this.currentSource === source) this.currentSource = null
        resolve()
      }

      if (!this.firstStarted) {
        this.firstStarted = true
        this.onPlayStart?.()
      }

      try {
        source.start()
      } catch {
        if (this.currentSource === source) this.currentSource = null
        resolve()
      }
    })
  }

  private async playWithHtmlAudio(arrayBuffer: ArrayBuffer): Promise<void> {
    const blob = new Blob([arrayBuffer], { type: 'audio/mpeg' })
    const url = URL.createObjectURL(blob)

    await new Promise<void>((resolve) => {
      if (this.stopped) { URL.revokeObjectURL(url); resolve(); return }

      const audio = new Audio(url)
      this.currentAudio = audio

      audio.onplay = () => {
        if (!this.firstStarted) {
          this.firstStarted = true
          this.onPlayStart?.()
        }
      }
      audio.onended = () => {
        URL.revokeObjectURL(url)
        if (this.currentAudio === audio) this.currentAudio = null
        resolve()
      }
      audio.onerror = () => {
        URL.revokeObjectURL(url)
        if (this.currentAudio === audio) this.currentAudio = null
        resolve()
      }
      audio.play().catch(() => {
        URL.revokeObjectURL(url)
        if (this.currentAudio === audio) this.currentAudio = null
        resolve()
      })
    })
  }

  stop(): void {
    this.stopped = true
    this.queue = []
    this.processing = false
    try { this.currentSource?.stop() } catch { /* already stopped */ }
    this.currentSource = null
    if (this.currentAudio) {
      this.currentAudio.pause()
      URL.revokeObjectURL(this.currentAudio.src)
      this.currentAudio = null
    }
  }

  reset(): void {
    this.stopped = false
  }

  get isSpeaking(): boolean {
    return this.processing || this.currentSource !== null || this.currentAudio !== null
  }
}
