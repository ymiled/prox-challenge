import { useCallback, useEffect, useRef, useState } from 'react'
import Message from './components/Message'
import { apiUrl, fetchHealth, streamChat, synthesizeSpeech, validateAnthropicKey } from './lib/api'
import { appendStreamText, repairConcatenatedWords } from './lib/joinStreamText'
import { parseArtifacts } from './lib/parseArtifacts'
import type { Artifact, ArtifactEvent, Message as MessageType, PageRef } from './lib/types'

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor
    webkitSpeechRecognition?: SpeechRecognitionConstructor
  }
}

interface SpeechRecognitionEventLike extends Event {
  results: SpeechRecognitionResultList
}

interface SpeechRecognitionErrorEventLike extends Event {
  error: string
}

interface SpeechRecognitionLike extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null
  onend: (() => void) | null
}

interface SpeechRecognitionConstructor {
  new (): SpeechRecognitionLike
}

const SUGGESTED_QUESTIONS = [
  "What's the duty cycle for MIG welding at 200A on 240V?",
  'What polarity setup do I need for TIG welding? Which socket does the ground clamp go in?',
  "I'm getting porosity in my flux-cored welds. What should I check?",
  'Which welding process should I use for thin sheet steel at home?',
]

const VOICE_COMMAND_HINTS = [
  'Say: "show me the page"',
  'Say: "open the diagram"',
  'Say: "read that again"',
  'Say: "stop"',
]

let messageCounter = 0
function newId() {
  return `msg-${++messageCounter}-${Date.now()}`
}

function getSpeechRecognitionCtor(): SpeechRecognitionConstructor | null {
  if (typeof window === 'undefined') return null
  return window.SpeechRecognition ?? window.webkitSpeechRecognition ?? null
}

function createSpeechUtterance(text: string): SpeechSynthesisUtterance {
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = 1
  utterance.pitch = 1
  utterance.volume = 1
  utterance.lang = 'en-US'
  return utterance
}

function normalizeVoiceCommand(text: string): string {
  return text.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim()
}

function getFriendlyVoiceError(error: string): { message: string; status: string; showError: boolean } {
  switch (error) {
    case 'no-speech':
      return {
        message: '',
        status: 'Nothing heard — tap mic to try again',
        showError: false,
      }
    case 'audio-capture':
      return {
        message: 'I could not access your microphone. Check that a microphone is connected and available.',
        status: 'Microphone unavailable',
        showError: true,
      }
    case 'not-allowed':
    case 'service-not-allowed':
      return {
        message: 'Microphone access is blocked. Please allow microphone permission in your browser and try again.',
        status: 'Microphone permission needed',
        showError: true,
      }
    case 'network':
      return {
        message: 'Voice recognition lost its connection. Please try again in a moment.',
        status: 'Voice connection lost',
        showError: true,
      }
    case 'language-not-supported':
      return {
        message: 'This browser does not support voice recognition for the current language setting.',
        status: 'Voice language unsupported',
        showError: true,
      }
    case 'aborted':
      return {
        message: 'Voice capture was cancelled.',
        status: 'Voice capture cancelled',
        showError: false,
      }
    default:
      return {
        message: 'Voice input ran into a problem. Please try again.',
        status: 'Voice input unavailable',
        showError: true,
      }
  }
}

function cleanSpokenText(text: string): string {
  return text
    .replace(/•/g, ' ')
    .replace(/·/g, ' ')
    .replace(/—/g, ', ')
    .replace(/–/g, ', ')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/^\s*[-*]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/\[(.*?)\]\((.*?)\)/g, '$1')
    .replace(/<\/?[^>]+>/g, ' ')
    .replace(/\b([A-Za-z0-9_-]+)\s+p\.(\d+)\b/g, '$1 page $2')
    .replace(/\s+/g, ' ')
    .trim()
}

function buildSpokenText(message: MessageType): string {
  const base = cleanSpokenText(message.spokenText?.trim() || message.text.trim())
  if (!base) return ''

  const extras: string[] = []
  if (message.artifacts.length) {
    const artifact = message.artifacts[0]
    extras.push(`I am also showing ${artifact.title.toLowerCase()} below.`)
  }
  if (message.citations.length) {
    const citation = message.citations[0]
    extras.push(`The supporting manual reference is ${citation.doc} page ${citation.page}.`)
  }
  return [base, ...extras].join(' ').trim()
}

function pickEnglishVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  const normalized = voices.filter((voice) => voice.lang?.toLowerCase().startsWith('en'))
  if (!normalized.length) return null

  return (
    normalized.find((voice) => voice.lang.toLowerCase() === 'en-us') ||
    normalized.find((voice) => voice.lang.toLowerCase().startsWith('en-us')) ||
    normalized.find((voice) => /english|united states|america/i.test(`${voice.name} ${voice.lang}`)) ||
    normalized.find((voice) => voice.default) ||
    normalized[0]
  )
}

export default function App() {
  const [messages, setMessages] = useState<MessageType[]>([])
  const [input, setInput] = useState('')
  const [anthropicApiKey, setAnthropicApiKey] = useState('')
  const [serverHasAnthropicKey, setServerHasAnthropicKey] = useState(false)
  const [healthCheckDone, setHealthCheckDone] = useState(false)
  const [apiKeyDraft, setApiKeyDraft] = useState('')
  const [apiKeyGateError, setApiKeyGateError] = useState<string | null>(null)
  const [isValidatingKey, setIsValidatingKey] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [autoSpeak, setAutoSpeak] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [voiceMissed, setVoiceMissed] = useState(false)
  const [voiceError, setVoiceError] = useState<string | null>(null)
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null)
  const [voiceStatus, setVoiceStatus] = useState('Tap the mic button to talk')
  const [localTtsReady, setLocalTtsReady] = useState(false)
  const [localTtsEnabled, setLocalTtsEnabled] = useState(false)
  const [pendingAutoSpeakMessage, setPendingAutoSpeakMessage] = useState<MessageType | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const transcriptRef = useRef('')
  const pendingAutoSendRef = useRef(false)
  const englishVoiceRef = useRef<SpeechSynthesisVoice | null>(null)
  const speechPrimedRef = useRef(false)
  const streamingTextRef = useRef('')
  const streamingArtifactsRef = useRef<Artifact[]>([])
  const browserSpeechSupported = typeof window !== 'undefined' && 'speechSynthesis' in window
  const speechSupported = true
  const recognitionSupported = typeof window !== 'undefined' && getSpeechRecognitionCtor() !== null
  const voiceEngineLabel = localTtsEnabled && localTtsReady ? 'Using local voice' : browserSpeechSupported ? 'Using browser voice' : 'Voice unavailable'
  const effectiveAnthropicKey = anthropicApiKey.trim()
  const requiresUserKey = !serverHasAnthropicKey
  const showApiKeyGate = healthCheckDone && requiresUserKey && !effectiveAnthropicKey

  useEffect(() => {
    if (typeof window === 'undefined') return
    const savedKey = window.sessionStorage.getItem('vulcan_omnipro_assistant_anthropic_api_key')
    if (savedKey) {
      setAnthropicApiKey(savedKey)
      setApiKeyDraft(savedKey)
    }
  }, [])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  useEffect(() => {
    let active = true
    fetchHealth()
      .then((payload) => {
        if (!active) return
        setLocalTtsEnabled(Boolean(payload.local_tts_enabled))
        setLocalTtsReady(Boolean(payload.local_tts_ready))
        setServerHasAnthropicKey(Boolean((payload as { anthropic_enabled?: boolean }).anthropic_enabled))
      })
      .catch(() => {
        if (!active) return
        setLocalTtsEnabled(false)
        setLocalTtsReady(false)
        setServerHasAnthropicKey(false)
      })
      .finally(() => {
        if (active) setHealthCheckDone(true)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    return () => {
      recognitionRef.current?.abort()
      audioRef.current?.pause()
    }
  }, [])

  // Auto-reset transient voice status messages back to idle after a short delay
  useEffect(() => {
    const transientStatuses = [
      'Nothing heard — tap mic to try again',
      'Voice capture cancelled',
      'Question sent',
    ]
    if (!transientStatuses.includes(voiceStatus)) return
    const id = setTimeout(() => setVoiceStatus('Tap the mic button to talk'), 3500)
    return () => clearTimeout(id)
  }, [voiceStatus])

  useEffect(() => {
    if (!browserSpeechSupported) return

    const assignVoice = () => {
      englishVoiceRef.current = pickEnglishVoice(window.speechSynthesis.getVoices())
    }

    assignVoice()
    window.speechSynthesis.onvoiceschanged = assignVoice
    return () => {
      window.speechSynthesis.onvoiceschanged = null
    }
  }, [browserSpeechSupported])

  const primeBrowserSpeech = useCallback(() => {
    if (!browserSpeechSupported || speechPrimedRef.current) return
    try {
      const utterance = createSpeechUtterance(' ')
      utterance.volume = 0
      utterance.onend = () => {
        speechPrimedRef.current = true
      }
      utterance.onerror = () => {
        speechPrimedRef.current = true
      }
      window.speechSynthesis.cancel()
      window.speechSynthesis.resume()
      window.speechSynthesis.speak(utterance)
      speechPrimedRef.current = true
    } catch {
      // Best-effort browser unlock only.
    }
  }, [browserSpeechSupported])

  useEffect(() => {
    if (!browserSpeechSupported) return

    const handleActivation = () => {
      primeBrowserSpeech()
    }

    window.addEventListener('pointerdown', handleActivation, { passive: true })
    window.addEventListener('keydown', handleActivation)
    return () => {
      window.removeEventListener('pointerdown', handleActivation)
      window.removeEventListener('keydown', handleActivation)
    }
  }, [browserSpeechSupported, primeBrowserSpeech])

  const resetApiKeyGate = useCallback((message: string) => {
    setAnthropicApiKey('')
    setApiKeyDraft('')
    setApiKeyGateError(message)
    if (typeof window !== 'undefined') {
      window.sessionStorage.removeItem('vulcan_omnipro_assistant_anthropic_api_key')
    }
  }, [])

  const isApiKeyError = useCallback((message: string): boolean => {
    const lowered = message.toLowerCase()
    return (
      lowered.includes('api key') ||
      lowered.includes('x-api-key') ||
      lowered.includes('authentication') ||
      lowered.includes('unauthorized') ||
      lowered.includes('invalid api') ||
      lowered.includes('invalid x-api-key')
    )
  }, [])

  const stopSpeaking = useCallback(() => {
    audioRef.current?.pause()
    audioRef.current = null
    if (browserSpeechSupported) {
      window.speechSynthesis.cancel()
    }
    setSpeakingMessageId(null)
  }, [browserSpeechSupported])

  const latestAssistantMessage = useCallback((): MessageType | null => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index].role === 'assistant' && messages[index].text.trim()) return messages[index]
    }
    return null
  }, [messages])

  const speakMessage = useCallback(
    async (message: MessageType) => {
      const spokenText = buildSpokenText(message)
      if (!spokenText) return
      stopSpeaking()
      const speakWithBrowser = () => {
        primeBrowserSpeech()
        const utterance = createSpeechUtterance(spokenText)
        utterance.voice = englishVoiceRef.current
        if (englishVoiceRef.current?.lang) {
          utterance.lang = englishVoiceRef.current.lang
        }
        setSpeakingMessageId(message.id)
        setVoiceStatus('Using browser voice')
        utterance.onstart = () => {
          setVoiceStatus('Speaking answer...')
        }
        utterance.onend = () => {
          setSpeakingMessageId((current) => (current === message.id ? null : current))
          setVoiceStatus('Answer ready')
        }
        utterance.onerror = (e: Event) => {
          const errorType = (e as SpeechSynthesisErrorEvent).error
          setSpeakingMessageId((current) => (current === message.id ? null : current))
          if (errorType === 'interrupted' || errorType === 'canceled') {
            setVoiceStatus('Answer ready')
            return
          }
          setVoiceStatus('Browser voice playback failed')
          setVoiceError('Speech playback failed. Tap Speak once to re-enable browser voice.')
        }
        window.speechSynthesis.cancel()
        window.speechSynthesis.resume()
        window.speechSynthesis.speak(utterance)
      }

      if (!localTtsEnabled || !localTtsReady) {
        if (browserSpeechSupported) {
          setVoiceStatus('Using browser voice')
          speakWithBrowser()
        } else {
          setVoiceError('No speech engine is available in this browser.')
        }
        return
      }

      try {
        const url = await synthesizeSpeech(spokenText)
        const audio = new Audio(url)
        audioRef.current = audio
        setSpeakingMessageId(message.id)
        setVoiceStatus('Using local voice')
        audio.onplay = () => {
          setVoiceStatus('Speaking answer...')
        }
        audio.onended = () => {
          setSpeakingMessageId((current) => (current === message.id ? null : current))
          if (audioRef.current === audio) audioRef.current = null
          setVoiceStatus('Answer ready')
        }
        audio.onerror = () => {
          setSpeakingMessageId((current) => (current === message.id ? null : current))
          if (audioRef.current === audio) audioRef.current = null
          setVoiceStatus('Local speech playback failed')
          setVoiceError('Local speech playback failed.')
        }
        await audio.play()
      } catch (error) {
        setLocalTtsReady(false)
        setLocalTtsEnabled(false)
        if (!browserSpeechSupported) {
          setSpeakingMessageId(null)
          setVoiceError(error instanceof Error ? error.message : 'Local speech playback failed.')
          return
        }
        setVoiceStatus('Using browser voice')
        speakWithBrowser()
      }
    },
    [browserSpeechSupported, localTtsEnabled, localTtsReady, primeBrowserSpeech, stopSpeaking]
  )

  useEffect(() => {
    if (!pendingAutoSpeakMessage || !autoSpeak || isLoading) return
    let cancelled = false
    const timeoutId = window.setTimeout(() => {
      if (cancelled) return
      speakMessage(pendingAutoSpeakMessage).finally(() => {
        setPendingAutoSpeakMessage((current) => (current?.id === pendingAutoSpeakMessage.id ? null : current))
      })
    }, 50)
    return () => {
      cancelled = true
      window.clearTimeout(timeoutId)
    }
  }, [autoSpeak, isLoading, pendingAutoSpeakMessage, speakMessage])

  const openLatestPage = useCallback(() => {
    const assistant = latestAssistantMessage()
    const page = assistant?.citations?.[0]
    if (!page) {
      setVoiceError('No cited manual page is available yet.')
      return
    }
    window.open(apiUrl(`/pages/${page.doc}/${page.page}`), '_blank', 'noopener,noreferrer')
    setVoiceStatus(`Opening ${page.doc} page ${page.page}`)
  }, [latestAssistantMessage])

  const openLatestArtifact = useCallback(() => {
    const assistant = latestAssistantMessage()
    const artifact = assistant?.artifacts?.[0]
    if (!artifact) {
      setVoiceError('No diagram or artifact is available yet.')
      return
    }
    window.open(artifact.url, '_blank', 'noopener,noreferrer')
    setVoiceStatus(`Opening ${artifact.title}`)
  }, [latestAssistantMessage])

  const processVoiceCommand = useCallback(
    (spoken: string): boolean => {
      const normalized = normalizeVoiceCommand(spoken)
      if (!normalized) return true

      if (normalized === 'stop' || normalized === 'stop talking' || normalized === 'stop speaking') {
        stopSpeaking()
        setVoiceStatus('Stopped playback')
        return true
      }

      if (
        normalized.includes('read that again') ||
        normalized.includes('repeat that') ||
        normalized.includes('say that again') ||
        normalized.includes('read it again')
      ) {
        const assistant = latestAssistantMessage()
        if (assistant) {
          speakMessage(assistant)
          setVoiceStatus('Reading the last answer again')
        } else {
          setVoiceError('There is no assistant answer to read yet.')
        }
        return true
      }

      if (
        normalized.includes('show me the page') ||
        normalized.includes('open the page') ||
        normalized.includes('show the page')
      ) {
        openLatestPage()
        return true
      }

      if (
        normalized.includes('open the diagram') ||
        normalized.includes('show me the diagram') ||
        normalized.includes('open the artifact') ||
        normalized.includes('show me the visual')
      ) {
        openLatestArtifact()
        return true
      }

      return false
    },
    [latestAssistantMessage, openLatestArtifact, openLatestPage, speakMessage, stopSpeaking]
  )

  const handleSubmit = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || isLoading) return
      if (requiresUserKey && !effectiveAnthropicKey) {
        setApiKeyGateError('Enter your Anthropic API key to continue.')
        return
      }

      setInput('')
      setIsLoading(true)
      setVoiceError(null)
      primeBrowserSpeech()
      setVoiceStatus('Question sent')
      if (textareaRef.current) textareaRef.current.style.height = 'auto'

      const userMsg: MessageType = {
        id: newId(),
        role: 'user',
        text: trimmed,
        images: [],
        artifacts: [],
        citations: [],
        isStreaming: false,
      }

      const assistantId = newId()
      const assistantMsg: MessageType = {
        id: assistantId,
        role: 'assistant',
        text: '',
        spokenText: '',
        images: [],
        artifacts: [],
        citations: [],
        isStreaming: true,
      }

      streamingTextRef.current = ''
      streamingArtifactsRef.current = []
      setMessages((prev) => [...prev, userMsg, assistantMsg])
      const historySnapshot = messages.map((m) => ({ role: m.role, content: m.text }))

      try {
        await streamChat(
          trimmed,
          historySnapshot,
          {
            onTextDelta: (event) => {
              const parsed = parseArtifacts(event.content)
              streamingTextRef.current = appendStreamText(streamingTextRef.current, parsed.text)
              streamingArtifactsRef.current = dedupeArtifacts([...streamingArtifactsRef.current, ...parsed.artifacts])
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        text: appendStreamText(m.text, parsed.text),
                        artifacts: dedupeArtifacts([...m.artifacts, ...parsed.artifacts]),
                      }
                    : m
                )
              )
            },
            onImage: (event) => {
              const image = {
                doc: event.doc,
                page: event.page,
                caption: event.caption,
                url: apiUrl(event.url),
                src: event.src,
              }
              setMessages((prev) =>
                prev.map((m) => {
                  if (m.id !== assistantId) return m
                  if (m.images.some((i) => i.doc === image.doc && i.page === image.page)) return m
                  return { ...m, images: [...m.images, image] }
                })
              )
            },
            onArtifact: (event) => {
              const artifact = artifactFromEvent(event)
              streamingArtifactsRef.current = mergeAgentArtifact(streamingArtifactsRef.current, artifact)
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, artifacts: mergeAgentArtifact(m.artifacts, artifact) }
                    : m
                )
              )
            },
            onDone: (event) => {
              const citations: PageRef[] = dedupeCitations(event.citations ?? [])
              const repairedText = repairConcatenatedWords(streamingTextRef.current)
              const completedArtifacts = streamingArtifactsRef.current
              setMessages((prev) =>
                prev.map((m) => {
                  if (m.id !== assistantId) return m
                  return {
                    ...m,
                    isStreaming: false,
                    citations,
                    text: repairedText,
                    spokenText: repairedText,
                    artifacts: completedArtifacts,
                  }
                })
              )
              streamingTextRef.current = ''
              streamingArtifactsRef.current = []
              setIsLoading(false)
              setVoiceStatus(
                completedArtifacts.length || citations.length
                  ? 'Answer ready. You can say "show me the page" or "open the diagram".'
                  : 'Answer ready'
              )
              if (autoSpeak && repairedText.trim()) {
                setPendingAutoSpeakMessage({
                  id: assistantId,
                  role: 'assistant',
                  text: repairedText,
                  spokenText: repairedText,
                  images: [],
                  artifacts: completedArtifacts,
                  citations,
                  isStreaming: false,
                })
              }
            },
            onError: (event) => {
              streamingTextRef.current = ''
              streamingArtifactsRef.current = []
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false, error: event.message } : m))
              )
              setIsLoading(false)
              if (isApiKeyError(event.message)) {
                resetApiKeyGate('Enter a valid Anthropic API key.')
              }
              setVoiceError(event.message)
            },
          },
          { voiceMode: true, anthropicApiKey: effectiveAnthropicKey }
        )
      } catch (err) {
        streamingTextRef.current = ''
        streamingArtifactsRef.current = []
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, isStreaming: false, error: err instanceof Error ? err.message : 'Connection failed' }
              : m
          )
        )
        setIsLoading(false)
        if (err instanceof Error && isApiKeyError(err.message)) {
          resetApiKeyGate('Enter a valid Anthropic API key.')
        }
        setVoiceError(err instanceof Error ? err.message : 'Connection failed')
      }
    },
    [autoSpeak, effectiveAnthropicKey, isApiKeyError, isLoading, messages, primeBrowserSpeech, requiresUserKey, resetApiKeyGate]
  )

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop()
    setIsListening(false)
  }, [])

  const finishVoiceCapture = useCallback(() => {
    const transcript = transcriptRef.current.trim()
    pendingAutoSendRef.current = false

    if (!transcript) {
      setVoiceStatus('No speech captured')
      return
    }

    if (processVoiceCommand(transcript)) {
      setInput('')
      transcriptRef.current = ''
      return
    }

    setInput(transcript)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
    transcriptRef.current = ''
    handleSubmit(transcript)
  }, [handleSubmit, processVoiceCommand])

  const startListening = useCallback(() => {
    if (!recognitionSupported || isLoading) return
    stopSpeaking()
    primeBrowserSpeech()
    setVoiceError(null)
    setVoiceStatus('Listening...')

    const Ctor = getSpeechRecognitionCtor()
    if (!Ctor) return

    if (!recognitionRef.current) {
      const recognition = new Ctor()
      recognition.continuous = false
      recognition.interimResults = true
      recognition.lang = 'en-US'
      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((result) => result[0]?.transcript ?? '')
          .join(' ')
          .trim()
        transcriptRef.current = transcript
        setInput(transcript)
        if (textareaRef.current) {
          textareaRef.current.style.height = 'auto'
          textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
        }
      }
      recognition.onerror = (event) => {
        const friendly = getFriendlyVoiceError(event.error)
        if (friendly.showError) {
          setVoiceError(friendly.message)
        } else {
          setVoiceError(null)
        }
        setVoiceStatus(friendly.status)
        if (event.error === 'no-speech') {
          setVoiceMissed(true)
          setTimeout(() => setVoiceMissed(false), 1500)
        }
      }
      recognition.onend = () => {
        setIsListening(false)
        if (pendingAutoSendRef.current) {
          finishVoiceCapture()
        }
      }
      recognitionRef.current = recognition
    }

    try {
      recognitionRef.current.start()
      setIsListening(true)
    } catch {
      setVoiceError('Voice input is already active.')
    }
  }, [finishVoiceCapture, isLoading, primeBrowserSpeech, recognitionSupported, stopSpeaking])

  const beginVoiceCapture = useCallback(() => {
    if (!recognitionSupported || isLoading) return
    transcriptRef.current = ''
    pendingAutoSendRef.current = false
    startListening()
  }, [isLoading, recognitionSupported, startListening])

  const endVoiceCapture = useCallback(() => {
    if (!isListening) return
    pendingAutoSendRef.current = true
    stopListening()
    setVoiceStatus('Processing your voice...')
  }, [isListening, stopListening])

  const cancelHoldToTalk = useCallback(() => {
    pendingAutoSendRef.current = false
    transcriptRef.current = ''
    stopListening()
    setInput('')
    setVoiceStatus('Voice capture cancelled')
  }, [stopListening])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(input)
    }
  }

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
  }

  const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const nextValue = e.target.value
    setApiKeyDraft(nextValue)
    if (apiKeyGateError) setApiKeyGateError(null)
  }

  const handleApiKeySave = async () => {
    const nextValue = apiKeyDraft.trim()
    if (!nextValue) {
      setApiKeyGateError('Enter your Anthropic API key to continue.')
      return
    }
    if (!nextValue.startsWith('sk-ant-') || nextValue.length < 40) {
      setApiKeyGateError("That doesn't look like a valid Anthropic API key. It should start with sk-ant-")
      return
    }
    setIsValidatingKey(true)
    setApiKeyGateError(null)
    try {
      const result = await validateAnthropicKey(nextValue)
      if (!result.valid) {
        setApiKeyGateError(result.error ?? 'API key rejected. Double-check it and try again.')
        return
      }
    } catch (e) {
      setApiKeyGateError(e instanceof Error ? e.message : 'Could not reach the server. Try again.')
      return
    } finally {
      setIsValidatingKey(false)
    }
    setAnthropicApiKey(nextValue)
    setApiKeyDraft('')
    setApiKeyGateError(null)
    setVoiceError(null)
    if (typeof window !== 'undefined') {
      window.sessionStorage.setItem('vulcan_omnipro_assistant_anthropic_api_key', nextValue)
    }
  }

  const handleApiKeyKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleApiKeySave()
    }
  }

  const hasMessages = messages.length > 0

  return (
    <div className="relative flex flex-col h-screen bg-slate-50 overflow-hidden">
      {!healthCheckDone && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-slate-950/30 backdrop-blur-md">
          <span className="w-8 h-8 border-4 border-white/30 border-t-white rounded-full animate-spin" />
        </div>
      )}
      {showApiKeyGate && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/30 backdrop-blur-md px-4">
          <div className="w-full max-w-md rounded-3xl border border-white/60 bg-white/90 shadow-2xl p-6 text-center">
            <div className="w-14 h-14 bg-orange-100 rounded-2xl flex items-center justify-center mx-auto mb-4 text-2xl font-semibold text-orange-600">
              P
            </div>
            <h2 className="text-xl font-semibold text-slate-900 mb-2">Enter your Anthropic API key</h2>
            <p className="text-sm text-slate-500 mb-4">
              Your key is hidden in the UI and stored only in this browser tab for the current session.
            </p>
            <input
              type="password"
              value={apiKeyDraft}
              onChange={handleApiKeyChange}
              onKeyDown={handleApiKeyKeyDown}
              placeholder="sk-ant-..."
              autoComplete="off"
              spellCheck={false}
              disabled={isValidatingKey}
              className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-orange-400 focus:ring-2 focus:ring-orange-200 focus:outline-none disabled:bg-slate-50 disabled:text-slate-400"
            />
            {apiKeyGateError && <p className="mt-3 text-sm text-red-600">{apiKeyGateError}</p>}
            <button
              type="button"
              onClick={handleApiKeySave}
              disabled={isValidatingKey}
              className="mt-4 w-full rounded-xl bg-orange-500 hover:bg-orange-600 disabled:bg-orange-300 text-white px-4 py-3 text-sm font-semibold transition-colors shadow-sm flex items-center justify-center gap-2"
            >
              {isValidatingKey ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Checking key...
                </>
              ) : (
                'Continue'
              )}
            </button>
          </div>
        </div>
      )}
      <header className={`flex-shrink-0 bg-slate-900 text-white px-4 py-2 flex items-center justify-between gap-4 shadow-lg z-10 transition-all ${showApiKeyGate ? 'blur-sm pointer-events-none select-none' : ''}`}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center font-bold text-sm shadow-inner">
            V
          </div>
          <div>
            <h1 className="font-bold text-base leading-none tracking-wide">Vulcan OmniPro Assistant</h1>
            <p className="text-slate-400 text-xs mt-0.5">Voice-first help for the Vulcan OmniPro 220</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <button
            type="button"
            onClick={() => {
              if (!speechSupported) return
              if (autoSpeak) stopSpeaking()
              setAutoSpeak((value) => !value)
            }}
            disabled={!speechSupported}
            className={`rounded-full px-3 py-1.5 border transition-colors ${
              autoSpeak && speechSupported ? 'bg-white text-slate-900 border-white' : 'bg-slate-800 border-slate-700 text-slate-300'
            } disabled:opacity-50`}
          >
            Auto speak {autoSpeak ? 'on' : 'off'}
          </button>
        </div>
      </header>

      <div className={`flex flex-1 overflow-hidden transition-all ${showApiKeyGate ? 'blur-sm pointer-events-none select-none' : ''}`}>
        <div className="flex flex-col w-full min-w-0">
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4 chat-scroll">
            {!hasMessages && (
              <div className="flex flex-col items-center justify-center h-full text-center px-4 py-6">
                <div className="w-12 h-12 bg-orange-100 rounded-2xl flex items-center justify-center mb-3 text-lg font-semibold text-orange-600">
                  P
                </div>
                <h2 className="text-base font-semibold text-slate-700 mb-1">Vulcan OmniPro Assistant</h2>
                <p className="text-xs text-slate-400 max-w-sm mb-5">
                  Ask anything about setup, troubleshooting, or settings. Tap the mic to talk with Vulcan OmniPro Assistant.
                </p>
                <div className="grid grid-cols-1 gap-2 w-full max-w-md">
                  {SUGGESTED_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => handleSubmit(q)}
                      className="text-left text-sm text-slate-600 bg-white border border-slate-200 hover:border-orange-300 hover:bg-orange-50 rounded-xl px-4 py-2.5 transition-colors shadow-sm"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <Message
                key={msg.id}
                message={msg}
                canSpeak={speechSupported}
                isSpeaking={speakingMessageId === msg.id}
                onSpeak={speakMessage}
                onStopSpeaking={stopSpeaking}
              />
            ))}
          </div>

          <div className="flex-shrink-0 border-t border-slate-200 bg-white px-4 py-3">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handleSubmit(input)
              }}
              className="space-y-2"
            >
              <div className="flex items-end gap-2">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={handleTextareaChange}
                  onKeyDown={handleKeyDown}
                  placeholder={isListening ? 'Listening...' : 'Ask about the Vulcan OmniPro 220...'}
                  rows={1}
                  disabled={isLoading || (requiresUserKey && !effectiveAnthropicKey)}
                  className="flex-1 resize-none rounded-xl border border-slate-300 focus:border-orange-400 focus:ring-2 focus:ring-orange-200 focus:outline-none px-4 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 disabled:bg-slate-50 disabled:text-slate-400 transition-colors"
                  style={{ minHeight: '44px', maxHeight: '160px' }}
                />
                <button
                  type="button"
                  onClick={() => {
                    if (isListening) endVoiceCapture()
                    else beginVoiceCapture()
                  }}
                  onContextMenu={(e) => e.preventDefault()}
                  disabled={!recognitionSupported || isLoading || (requiresUserKey && !effectiveAnthropicKey)}
                  className={`flex-shrink-0 rounded-xl px-4 py-2.5 text-sm font-semibold transition-colors shadow-sm border ${
                    isListening
                      ? 'bg-red-50 text-red-700 border-red-200'
                      : voiceMissed
                      ? 'bg-amber-50 text-amber-700 border-amber-300'
                      : 'bg-white text-slate-700 border-slate-300 hover:border-orange-300 hover:text-orange-600'
                  } disabled:bg-slate-100 disabled:text-slate-400 disabled:border-slate-200`}
                  style={{ height: '44px' }}
                >
                  {isListening ? 'Tap to send' : voiceMissed ? 'Try again' : 'Tap to talk'}
                </button>
                <button
                  type="submit"
                  disabled={isLoading || !input.trim() || (requiresUserKey && !effectiveAnthropicKey)}
                  className="flex-shrink-0 bg-orange-500 hover:bg-orange-600 disabled:bg-slate-200 disabled:text-slate-400 text-white rounded-xl px-5 py-2.5 text-sm font-semibold transition-colors shadow-sm"
                  style={{ height: '44px' }}
                >
                  {isLoading ? (
                    <span className="flex items-center gap-1.5">
                      <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Wait</span>
                    </span>
                  ) : (
                    'Send'
                  )}
                </button>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 pl-1">
                <span>Enter to send · Shift+Enter for new line</span>
                <div className="flex flex-wrap items-center gap-3">
                  <span>{recognitionSupported ? voiceStatus : 'Voice input not supported in this browser'}</span>
                  <span>{speechSupported ? voiceEngineLabel : 'Speech playback not supported'}</span>
                </div>
              </div>
              {recognitionSupported && (
                <div className="flex flex-wrap gap-2 pl-1">
                  {VOICE_COMMAND_HINTS.map((hint) => (
                    <button
                      key={hint}
                      type="button"
                      onClick={() => setInput(hint.replace('Say: "', '').replace('"', ''))}
                      className="text-[11px] text-slate-500 bg-slate-100 hover:bg-orange-50 hover:text-orange-600 rounded-full px-2.5 py-1 transition-colors"
                    >
                      {hint}
                    </button>
                  ))}
                </div>
              )}
              {isListening && (
                <button
                  type="button"
                  onClick={cancelHoldToTalk}
                  className="text-xs text-slate-500 hover:text-red-600 transition-colors pl-1"
                >
                  Cancel voice capture
                </button>
              )}
              {voiceError && <p className="text-xs text-red-500 pl-1">{voiceError}</p>}
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

function dedupeArtifacts(artifacts: Artifact[]): Artifact[] {
  const seen = new Map<string, Artifact>()
  for (const artifact of artifacts) seen.set(artifact.id, artifact)
  return Array.from(seen.values())
}

const ARTIFACT_TYPE_TO_MIME: Record<Artifact['type'], string> = {
  react: 'application/vnd.ant.react',
  svg: 'image/svg+xml',
  html: 'text/html',
  code: 'application/vnd.ant.code',
  markdown: 'text/markdown',
  mermaid: 'application/vnd.ant.mermaid',
  json: 'application/json',
}

function artifactFromEvent(event: ArtifactEvent): Artifact {
  const { artifact_id, artifact_type, title, content } = event.artifact
  return {
    id: artifact_id,
    type: artifact_type,
    mimeType: ARTIFACT_TYPE_TO_MIME[artifact_type] ?? 'application/json',
    title,
    content,
    url: apiUrl(`/artifacts/${artifact_id}`),
  }
}

/**
 * Merge an agent-generated artifact into the list. If an inline artifact with
 * the same title already exists (parsed from <antArtifact> in text), replace it
 * so the agent version (purpose-built by the artifact agent) wins.
 */
function mergeAgentArtifact(existing: Artifact[], incoming: Artifact): Artifact[] {
  const replaced = existing.map((a) => (a.title === incoming.title ? incoming : a))
  if (replaced.some((a) => a.id === incoming.id || a.title === incoming.title)) return replaced
  return [...existing, incoming]
}

function dedupeCitations(citations: PageRef[]): PageRef[] {
  const seen = new Set<string>()
  const out: PageRef[] = []
  for (const citation of citations) {
    const key = `${citation.doc}:${citation.page}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(citation)
  }
  return out
}
