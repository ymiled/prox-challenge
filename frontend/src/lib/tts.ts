import { apiUrl, getCsrfToken } from './api'

/**
 * Pulls complete sentences from the front of an accumulator string.
 *
 * Only splits when the character *after* the trailing whitespace is visible
 * (capital letter, digit, quote, newline). This prevents premature splits at
 * the tail of the current buffer — e.g. "Hi! " alone will NOT be extracted;
 * we wait for the next LLM token so we can confirm a new sentence is starting.
 * The onDone handler flushes whatever remains at stream end.
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

/**
 * Queues sentences and plays them sequentially using Deepgram Aura TTS.
 */
export class TTSPlayer {
  private queue: string[] = []
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
    private readonly getAudioContext?: () => AudioContext | null,
  ) {}

  enqueue(text: string): void {
    if (this.stopped || !text.trim()) return
    this.queue.push(text)
    if (!this.processing) void this.drain()
  }

  private async drain(): Promise<void> {
    this.processing = true
    this.firstStarted = false

    let nextFetch: Promise<ArrayBuffer | null> | null =
      this.queue.length > 0 ? this.fetchAudio(this.queue.shift()!) : null

    while (nextFetch !== null && !this.stopped) {
      const bufPromise = nextFetch

      nextFetch = this.queue.length > 0 && !this.stopped
        ? this.fetchAudio(this.queue.shift()!)
        : null

      const buf = await bufPromise

      if (nextFetch === null && this.queue.length > 0 && !this.stopped) {
        nextFetch = this.fetchAudio(this.queue.shift()!)
      }

      if (buf && !this.stopped) {
        try {
          await this.playBuffer(buf)
        } catch {
          // one failed sentence must not silence the rest
        }
      }

      if (nextFetch === null && this.queue.length > 0 && !this.stopped) {
        nextFetch = this.fetchAudio(this.queue.shift()!)
      }
    }

    this.processing = false
    if (!this.stopped) this.onPlayEnd?.()
  }

  private async fetchAudio(text: string): Promise<ArrayBuffer | null> {
    if (this.stopped) return null
    try {
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
        return null
      }

      const arrayBuffer = await response.arrayBuffer()
      return this.stopped ? null : arrayBuffer
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'TTS request failed'
      if (!this.stopped) this.onError?.(msg)
      return null
    }
  }

  private async playBuffer(arrayBuffer: ArrayBuffer): Promise<void> {
    const ctx = this.getAudioContext?.()

    if (ctx && ctx.state !== 'closed') {
      if (ctx.state === 'suspended') {
        try {
          await ctx.resume()
        } catch {
          // ignore
        }
      }
      await this.playWithAudioContext(ctx, arrayBuffer)
    } else {
      await this.playWithHtmlAudio(arrayBuffer)
    }
  }

  private async playWithAudioContext(ctx: AudioContext, arrayBuffer: ArrayBuffer): Promise<void> {
    if (ctx.state === 'suspended') {
      await this.playWithHtmlAudio(arrayBuffer)
      return
    }

    let audioBuffer: AudioBuffer
    try {
      audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0))
    } catch {
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
