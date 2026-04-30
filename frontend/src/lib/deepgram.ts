import { getCsrfToken, webSocketUrl } from './api'

export interface TranscriptResult {
  transcript: string
  isFinal: boolean
  speechFinal: boolean
}

/**
 * Streams raw linear16 PCM audio from the microphone to the backend Deepgram
 * WebSocket proxy and delivers real-time transcripts.
 *
 * Uses ScriptProcessorNode (still well-supported cross-browser) to capture
 * 16 kHz mono PCM — matching the Deepgram `encoding=linear16&sample_rate=16000`
 * parameters set on the server.
 */
export class DeepgramASR {
  private ws: WebSocket | null = null
  private audioCtx: AudioContext | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private processor: ScriptProcessorNode | null = null
  private stream: MediaStream | null = null
  private started = false

  async start(
    onTranscript: (result: TranscriptResult) => void,
    onError: (message: string) => void,
    apiKey?: string,
    onStreamReady?: () => void,
  ): Promise<void> {
    if (this.started) return

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
    } catch {
      onError('Microphone access denied. Please allow microphone permission and try again.')
      return
    }

    // getUserMedia resolved — AVAudioSession is now .playAndRecord.
    // Notify the caller so it can try to revive the TTS AudioContext while
    // the session mode supports both recording and playback.
    onStreamReady?.()

    const params = new URLSearchParams()
    if (apiKey?.trim()) params.set('deepgram_api_key', apiKey.trim())
    const csrfToken = getCsrfToken()
    if (csrfToken) params.set('csrf_token', csrfToken)
    try {
      this.ws = new WebSocket(webSocketUrl('/ws/transcribe', params))
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Could not open the voice connection.')
      this.stream.getTracks().forEach((track) => track.stop())
      this.stream = null
      return
    }
    this.ws.binaryType = 'arraybuffer'

    this.ws.onopen = () => this.setupCapture()

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string) as {
          transcript?: string
          is_final?: boolean
          speech_final?: boolean
          error?: string
        }
        if (data.error) {
          onError(data.error)
          return
        }
        if (data.transcript !== undefined) {
          onTranscript({
            transcript: data.transcript,
            isFinal: Boolean(data.is_final),
            speechFinal: Boolean(data.speech_final),
          })
        }
      } catch {
        // ignore parse errors
      }
    }

    this.ws.onerror = () => onError('Deepgram connection error. Check that the backend is running.')
    this.ws.onclose = () => {
      this.started = false
    }

    this.started = true
  }

  private setupCapture(): void {
    if (!this.stream || !this.ws) return

    // AudioContext resamples to 16 kHz at the OS level via constraints above;
    // sampleRate here ensures we tell WebAudio to operate at 16 kHz.
    this.audioCtx = new AudioContext({ sampleRate: 16000 })
    this.source = this.audioCtx.createMediaStreamSource(this.stream)
    // 4096-sample buffer → ~256 ms chunks at 16 kHz, acceptable latency
    this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1)

    this.processor.onaudioprocess = (e) => {
      if (this.ws?.readyState !== WebSocket.OPEN) return
      const float32 = e.inputBuffer.getChannelData(0)
      const int16 = new Int16Array(float32.length)
      for (let i = 0; i < float32.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32768)))
      }
      this.ws.send(int16.buffer)
    }

    this.source.connect(this.processor)
    // Connect to destination so ScriptProcessor fires (required by Web Audio spec)
    this.processor.connect(this.audioCtx.destination)
  }

  stop(): void {
    this.started = false
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send('EOS')
      } catch {
        // ignore
      }
    }
    // Give Deepgram time to flush before closing
    setTimeout(() => this.ws?.close(), 200)
    this.processor?.disconnect()
    this.source?.disconnect()
    // Delay stopping mic tracks — stopping immediately causes iOS to revert
    // AVAudioSession from .playAndRecord to .ambient, which suspends the TTS
    // AudioContext before it can start playing. Keeping tracks alive for 3 s
    // gives TTS time to start within the same .playAndRecord session.
    const streamToStop = this.stream
    setTimeout(() => streamToStop?.getTracks().forEach((t) => t.stop()), 3000)
    // Delay AudioContext.close() by 3 s.
    // On iOS/macOS Safari the ASR context holds AVAudioSession in
    // .playAndRecord mode. Closing it immediately causes the OS to
    // reconfigure the session, which suspends the TTS AudioContext before
    // it can start playing (the LLM response still takes 1-3 s to arrive).
    // Keeping the context alive until TTS is underway avoids the interruption.
    const ctxToClose = this.audioCtx
    setTimeout(() => ctxToClose?.close().catch(() => undefined), 3000)
    this.ws = null
    this.processor = null
    this.source = null
    this.audioCtx = null
    this.stream = null
  }

  get active(): boolean {
    return this.started
  }
}
