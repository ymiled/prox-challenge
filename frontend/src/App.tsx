import { useCallback, useEffect, useRef, useState } from 'react'
import Message from './components/Message'
import {
  apiUrl,
  deleteAnthropicCredential,
  deleteDeepgramCredential,
  fetchAuthMe,
  fetchCredentialStatus,
  fetchHealth,
  login,
  logout,
  saveAnthropicCredential,
  saveDeepgramCredential,
  signup,
  streamChat,
  synthesizeSpeech,
} from './lib/api'
import { appendStreamText, repairConcatenatedWords } from './lib/joinStreamText'
import { DeepgramASR } from './lib/deepgram'
import { TTSPlayer, drainSentences } from './lib/tts'
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

interface AuthUser {
  username: string
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

function idleVoiceStatus(conversationModeEnabled: boolean, deepgramEnabled: boolean): string {
  if (conversationModeEnabled && deepgramEnabled) return 'Listening...'
  return 'Tap the mic button to talk'
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
    .replace(/•/g, '\n- ')
    .replace(/·/g, '\n- ')
    .replace(/—/g, ', ')
    .replace(/–/g, ', ')
    .replace(/```html[\s\S]*?```/gi, ' I included an HTML example below. ')
    .replace(/```(?:tsx|jsx|ts|js|json|css|bash|sh|mermaid|svg|xml)?[\s\S]*?```/gi, ' I included a code example below. ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/^\s*[-*]\s+/gm, 'Then, ')
    .replace(/^\s*(\d+)\.\s+/gm, 'Step $1: ')
    .replace(/\[(.*?)\]\((.*?)\)/g, '$1')
    .replace(/<\/?[^>]+>/g, ' ')
    .replace(/\b[a-z][\w:-]*="[^"]*"/g, ' ')
    .replace(/\b[a-z][\w:-]*='[^']*'/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\b([A-Za-z0-9_-]+)\s+p\.(\d+)\b/g, '$1 page $2')
    .replace(/https?:\/\/\S+/gi, ' ')
    .replace(/\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b/gi, ' ')
    .replace(/\{[\s\S]{80,}\}/g, ' I included structured data below. ')
    .replace(/\[[\s\S]{80,}\]/g, ' I included a detailed list below. ')
    .replace(/([A-Za-z])\/([A-Za-z])/g, '$1 or $2')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\s*\n\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function truncateForSpeech(text: string, maxLength = 550): string {
  if (text.length <= maxLength) return text
  const truncated = text.slice(0, maxLength)
  const lastSentenceEnd = Math.max(truncated.lastIndexOf('.'), truncated.lastIndexOf('!'), truncated.lastIndexOf('?'))
  if (lastSentenceEnd > maxLength * 0.6) {
    return `${truncated.slice(0, lastSentenceEnd + 1).trim()}`
  }
  const lastSpace = truncated.lastIndexOf(' ')
  const safe = lastSpace > 0 ? truncated.slice(0, lastSpace) : truncated
  return `${safe.trim()}.`
}

function buildArtifactSpeech(artifacts: Artifact[]): string {
  if (!artifacts.length) return ''
  if (artifacts.length === 1) {
    const artifact = artifacts[0]
    const artifactKind = artifact.type === 'html' || artifact.type === 'react'
      ? 'an interactive'
      : artifact.type === 'svg' || artifact.type === 'mermaid'
        ? 'a visual'
        : artifact.type === 'code' || artifact.type === 'json'
          ? 'a reference'
          : 'an artifact'
    return `I am also showing ${artifactKind} ${artifact.title.toLowerCase()} below.`
  }
  const artifactNames = artifacts
    .slice(0, 2)
    .map((artifact) => artifact.title.toLowerCase())
    .join(' and ')
  const remainder = artifacts.length - 2
  return remainder > 0
    ? `I am also showing ${artifactNames}, plus ${remainder} more items below.`
    : `I am also showing ${artifactNames} below.`
}

function buildCitationSpeech(citations: PageRef[]): string {
  if (!citations.length) return ''
  const refs = citations.slice(0, 2).map((citation) => `${citation.doc} page ${citation.page}`)
  if (refs.length === 1) {
    return `The supporting manual reference is ${refs[0]}.`
  }
  return `The supporting manual references are ${refs.join(' and ')}.`
}

function buildSpokenText(message: MessageType): string {
  const sourceText = message.spokenText?.trim() || message.text.trim()
  const structureHeavy =
    /<\/?[a-z][^>]*>/i.test(sourceText) ||
    /```[\s\S]*?```/.test(sourceText) ||
    /<antArtifact[\s\S]*?<\/antArtifact>/.test(sourceText) ||
    sourceText.split('\n').filter((line) => /^\s*(?:<|{|}|\[|\]|function |\w+\(|const |let |var )/.test(line)).length >= 4

  const base = structureHeavy
    ? 'I prepared the answer on screen and kept the structured content in the visual output below.'
    : truncateForSpeech(cleanSpokenText(sourceText))
  if (!base) return ''

  const extras: string[] = []
  const artifactSpeech = buildArtifactSpeech(message.artifacts)
  const citationSpeech = buildCitationSpeech(message.citations)
  if (artifactSpeech) extras.push(artifactSpeech)
  if (citationSpeech) extras.push(citationSpeech)
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
  const [serverHasAnthropicKey, setServerHasAnthropicKey] = useState(false)
  const [anthropicKeySource, setAnthropicKeySource] = useState<'env' | 'stored' | null>(null)
  const [serverHasDeepgramKey, setServerHasDeepgramKey] = useState(false)
  const [deepgramKeySource, setDeepgramKeySource] = useState<'env' | 'stored' | null>(null)
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login')
  const [authUsername, setAuthUsername] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authError, setAuthError] = useState<string | null>(null)
  const [isAuthenticating, setIsAuthenticating] = useState(false)
  const [isLoggingOut, setIsLoggingOut] = useState(false)
  const [disableServerDeepgramForSession, setDisableServerDeepgramForSession] = useState(false)
  const [forceApiKeyOverride, setForceApiKeyOverride] = useState(false)
  const [forceDeepgramKeyOverride, setForceDeepgramKeyOverride] = useState(false)
  const [healthCheckDone, setHealthCheckDone] = useState(false)
  const [apiKeyDraft, setApiKeyDraft] = useState('')
  const [deepgramKeyDraft, setDeepgramKeyDraft] = useState('')
  const [apiKeyGateError, setApiKeyGateError] = useState<string | null>(null)
  const [deepgramKeyGateError, setDeepgramKeyGateError] = useState<string | null>(null)
  const [isValidatingKey, setIsValidatingKey] = useState(false)
  const [isValidatingDeepgramKey, setIsValidatingDeepgramKey] = useState(false)
  const [isDeletingAnthropicCredential, setIsDeletingAnthropicCredential] = useState(false)
  const [isDeletingDeepgramCredential, setIsDeletingDeepgramCredential] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [settingsError, setSettingsError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [autoSpeak, setAutoSpeak] = useState(() => localStorage.getItem('autoSpeak') === 'true')
  const [conversationModeEnabled, setConversationModeEnabled] = useState(() => localStorage.getItem('conversationMode') === 'true')
  const [isListening, setIsListening] = useState(false)
  const [isWakeListening, setIsWakeListening] = useState(false)
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
  const startConversationListeningRef = useRef<(() => void) | null>(null)
  const finishVoiceCaptureRef = useRef<(() => void) | null>(null)
  const streamingTextRef = useRef('')
  const streamingArtifactsRef = useRef<Artifact[]>([])
  const deepgramAsrRef = useRef<DeepgramASR | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const streamAbortControllerRef = useRef<AbortController | null>(null)
  const deepgramModeRef = useRef<'idle' | 'manual' | 'wake' | 'conversation'>('idle')
  const conversationArmedRef = useRef(false)
  const conversationCommitTimeoutRef = useRef<number | null>(null)
  const interruptionGuardRef = useRef('')
  const assistantSpeakingRef = useRef(false)
  const deepgramTtsPlayerRef = useRef<TTSPlayer | null>(null)
  const ttsBufferRef = useRef('')
  const browserSpeechSupported = typeof window !== 'undefined' && 'speechSynthesis' in window
  const speechSupported = true
  const voiceLoopEnabled = autoSpeak || conversationModeEnabled
  const deepgramEnabled =
    !forceDeepgramKeyOverride && !disableServerDeepgramForSession && serverHasDeepgramKey
  const recognitionSupported = deepgramEnabled || (typeof window !== 'undefined' && getSpeechRecognitionCtor() !== null)
  const requiresUserKey = !serverHasAnthropicKey || forceApiKeyOverride
  const showAuthGate = authChecked && currentUser === null
  const showApiKeyGate = healthCheckDone && authChecked && currentUser !== null && requiresUserKey
  const showDeepgramKeyGate = healthCheckDone && authChecked && currentUser !== null && forceDeepgramKeyOverride

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const refreshCredentialStatus = useCallback(async () => {
    const [health, credentials] = await Promise.all([fetchHealth(), fetchCredentialStatus()])
    setLocalTtsEnabled(Boolean(health.local_tts_enabled))
    setLocalTtsReady(Boolean(health.local_tts_ready))
    setServerHasDeepgramKey(Boolean(credentials.deepgram_configured))
    setAnthropicKeySource(credentials.anthropic_source ?? null)
    setDeepgramKeySource(credentials.deepgram_source ?? null)
    setServerHasAnthropicKey(Boolean(credentials.anthropic_configured))
  }, [])

  useEffect(() => {
    let active = true
    Promise.allSettled([fetchHealth(), fetchAuthMe()])
      .then(async ([healthResult, authResult]) => {
        if (!active) return
        if (healthResult.status === 'fulfilled') {
          setLocalTtsEnabled(Boolean(healthResult.value.local_tts_enabled))
          setLocalTtsReady(Boolean(healthResult.value.local_tts_ready))
        } else {
          setLocalTtsEnabled(false)
          setLocalTtsReady(false)
        }

        if (authResult.status === 'fulfilled' && authResult.value.authenticated && authResult.value.user) {
          setCurrentUser({ username: authResult.value.user.username })
          await refreshCredentialStatus().catch(() => {
            if (!active) return
            setServerHasDeepgramKey(false)
            setAnthropicKeySource(null)
            setDeepgramKeySource(null)
            setServerHasAnthropicKey(false)
          })
        } else {
          setCurrentUser(null)
          setServerHasDeepgramKey(false)
          setAnthropicKeySource(null)
          setDeepgramKeySource(null)
          setServerHasAnthropicKey(false)
        }
      })
      .catch(() => {
        if (!active) return
        setLocalTtsEnabled(false)
        setLocalTtsReady(false)
        setServerHasDeepgramKey(false)
        setAnthropicKeySource(null)
        setDeepgramKeySource(null)
        setServerHasAnthropicKey(false)
      })
      .finally(() => {
        if (active) {
          setAuthChecked(true)
          setHealthCheckDone(true)
        }
      })
    return () => {
      active = false
    }
  }, [refreshCredentialStatus])

  useEffect(() => {
    return () => {
      audioCtxRef.current?.close().catch(() => undefined)
      deepgramAsrRef.current?.stop()
      deepgramTtsPlayerRef.current?.stop()
      recognitionRef.current?.abort()
      audioRef.current?.pause()
    }
  }, [])

  // Unlock the AudioContext on every user gesture so TTSPlayer can play
  // audio in async callbacks without triggering browser autoplay restrictions.
  //
  // Safari requires audio to be *started synchronously* inside the gesture
  // handler — calling start() inside a .then() callback loses the activation.
  // So we always call src.start(0) synchronously first (it queues up on a
  // suspended context and fires once resume() resolves), then call resume().
  // This re-pings on every gesture so neither Safari nor Chrome idle-suspends
  // the context in the seconds between the user pressing Send and TTS firing.
  useEffect(() => {
    const unlock = () => {
      if (!audioCtxRef.current) {
        try { audioCtxRef.current = new AudioContext() } catch { return }
      }
      const ctx = audioCtxRef.current
      // Always play the silent buffer synchronously within the gesture.
      try {
        const buf = ctx.createBuffer(1, 1, ctx.sampleRate)
        const src = ctx.createBufferSource()
        src.buffer = buf
        src.connect(ctx.destination)
        src.start(0)
      } catch { /* ignore */ }
      if (ctx.state === 'suspended') {
        ctx.resume().catch(() => undefined)
      }
    }
    window.addEventListener('pointerdown', unlock, { passive: true })
    window.addEventListener('keydown', unlock)
    return () => {
      window.removeEventListener('pointerdown', unlock)
      window.removeEventListener('keydown', unlock)
    }
  }, [])

  // Ensure autoSpeak is enabled whenever conversationModeEnabled is true
  useEffect(() => {
    if (conversationModeEnabled && !autoSpeak) {
      setAutoSpeak(true)
      localStorage.setItem('autoSpeak', 'true')
    }
  }, [conversationModeEnabled, autoSpeak])

  // Auto-reset transient voice status messages back to idle after a short delay
  useEffect(() => {
    const transientStatuses = [
      'Nothing heard — tap mic to try again',
      'Voice capture cancelled',
      'Question sent',
    ]
    if (!transientStatuses.includes(voiceStatus)) return
    const id = setTimeout(() => setVoiceStatus(idleVoiceStatus(conversationModeEnabled, deepgramEnabled)), 3500)
    return () => clearTimeout(id)
  }, [conversationModeEnabled, deepgramEnabled, voiceStatus])

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

  const resetApiKeyGate = useCallback((message: string, options?: { preserveDraft?: boolean }) => {
    setForceApiKeyOverride(true)
    setApiKeyDraft(options?.preserveDraft ? apiKeyDraft.trim() : '')
    setApiKeyGateError(message)
  }, [apiKeyDraft])

  const shouldReopenApiKeyGate = useCallback((message: string): boolean => {
    const lowered = message.toLowerCase()
    return (
      lowered.includes('api key') ||
      lowered.includes('x-api-key') ||
      lowered.includes('authentication') ||
      lowered.includes('unauthorized') ||
      lowered.includes('invalid api') ||
      lowered.includes('invalid x-api-key') ||
      lowered.includes('insufficient balance') ||
      lowered.includes('credit balance') ||
      lowered.includes('billing') ||
      lowered.includes('quota')
    )
  }, [])

  const resetDeepgramKeyGate = useCallback((message: string, options?: { preserveDraft?: boolean }) => {
    setForceDeepgramKeyOverride(true)
    setDeepgramKeyDraft(options?.preserveDraft ? deepgramKeyDraft.trim() : '')
    setDeepgramKeyGateError(message)
  }, [deepgramKeyDraft])

  const shouldReopenDeepgramKeyGate = useCallback((message: string): boolean => {
    const lowered = message.toLowerCase()
    return (
      lowered.includes('deepgram') ||
      lowered.includes('token') ||
      lowered.includes('transcribe') ||
      lowered.includes('tts') ||
      lowered.includes('authorization') ||
      lowered.includes('invalid deepgram api key') ||
      lowered.includes('no available balance') ||
      lowered.includes('out of quota') ||
      lowered.includes('not configured') ||
      lowered.includes('rate limited')
    )
  }, [])

  const stopSpeaking = useCallback(() => {
    assistantSpeakingRef.current = false
    deepgramTtsPlayerRef.current?.stop()
    deepgramTtsPlayerRef.current = null
    audioRef.current?.pause()
    audioRef.current = null
    if (browserSpeechSupported) {
      window.speechSynthesis.cancel()
    }
    setSpeakingMessageId(null)
  }, [browserSpeechSupported])

  const interruptAssistantTurn = useCallback(() => {
    streamAbortControllerRef.current?.abort()
    streamAbortControllerRef.current = null
    setPendingAutoSpeakMessage(null)
    ttsBufferRef.current = ''
    stopSpeaking()
  }, [stopSpeaking])

  const stopDeepgramSession = useCallback(() => {
    if (conversationCommitTimeoutRef.current !== null) {
      window.clearTimeout(conversationCommitTimeoutRef.current)
      conversationCommitTimeoutRef.current = null
    }
    deepgramAsrRef.current?.stop()
    deepgramAsrRef.current = null
    deepgramModeRef.current = 'idle'
    interruptionGuardRef.current = ''
    setIsWakeListening(false)
  }, [])

  const pauseListeningWhileAssistantSpeaks = useCallback(() => {
    stopDeepgramSession()
    recognitionRef.current?.stop()
    setIsListening(false)
  }, [stopDeepgramSession])

  const latestAssistantMessage = useCallback((): MessageType | null => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index].role === 'assistant' && messages[index].text.trim()) return messages[index]
    }
    return null
  }, [messages])

  const speakText = useCallback(
    async (
      spokenText: string,
      options?: {
        messageId?: string | null
        endStatus?: string | null
        onEnd?: () => void
        resumeConversation?: boolean
      },
    ) => {
      const trimmed = spokenText.trim()
      if (!trimmed) {
        options?.onEnd?.()
        return
      }

      const messageId = options?.messageId ?? null
      const endStatus = options?.endStatus ?? 'Answer ready'
      interruptionGuardRef.current = conversationArmedRef.current ? normalizeVoiceCommand(trimmed).slice(0, 180) : ''
      const resumeConversation = Boolean(options?.resumeConversation && conversationModeEnabled && deepgramEnabled)
      const finishPlayback = () => {
        assistantSpeakingRef.current = false
        if (messageId) {
          setSpeakingMessageId((current: string | null) => (current === messageId ? null : current))
        } else {
          setSpeakingMessageId(null)
        }
        if (endStatus) setVoiceStatus(endStatus)
        options?.onEnd?.()
        if (options?.resumeConversation && conversationArmedRef.current && deepgramModeRef.current !== 'conversation') {
          window.setTimeout(() => {
            startConversationListeningRef.current?.()
          }, 150)
        }
        interruptionGuardRef.current = ''
      }

      interruptAssistantTurn()
      pauseListeningWhileAssistantSpeaks()
      if (resumeConversation) stopDeepgramSession()

      if (deepgramEnabled) {
        assistantSpeakingRef.current = true
        if (messageId) setSpeakingMessageId(messageId)
        setVoiceStatus('Speaking answer...')
        const player = new TTSPlayer(
          () => {
            assistantSpeakingRef.current = true
            pauseListeningWhileAssistantSpeaks()
            setVoiceStatus('Speaking answer...')
          },
          finishPlayback,
          (message) => {
            if (shouldReopenDeepgramKeyGate(message)) {
              resetDeepgramKeyGate('This Deepgram API key cannot be used right now. Enter another Deepgram API key.')
              setVoiceStatus('Deepgram key needs attention')
            }
            setVoiceError(message)
          },
          undefined,
          () => audioCtxRef.current,
        )
        deepgramTtsPlayerRef.current = player
        player.enqueue(trimmed)
        return
      }

      const speakWithBrowser = () => {
        primeBrowserSpeech()
        const utterance = createSpeechUtterance(trimmed)
        utterance.voice = englishVoiceRef.current
        if (englishVoiceRef.current?.lang) {
          utterance.lang = englishVoiceRef.current.lang
        }
        if (messageId) setSpeakingMessageId(messageId)
        setVoiceStatus('Using browser voice')
        utterance.onstart = () => {
          assistantSpeakingRef.current = true
          pauseListeningWhileAssistantSpeaks()
          setVoiceStatus('Speaking answer...')
        }
        utterance.onend = finishPlayback
        utterance.onerror = (e: Event) => {
          const errorType = (e as SpeechSynthesisErrorEvent).error
          if (messageId) {
            setSpeakingMessageId((current: string | null) => (current === messageId ? null : current))
          } else {
            setSpeakingMessageId(null)
          }
          if (errorType === 'interrupted' || errorType === 'canceled') {
            if (endStatus) setVoiceStatus(endStatus)
            options?.onEnd?.()
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
        const url = await synthesizeSpeech(trimmed)
        const audio = new Audio(url)
        audioRef.current = audio
        if (messageId) setSpeakingMessageId(messageId)
        setVoiceStatus('Using local voice')
        audio.onplay = () => {
          assistantSpeakingRef.current = true
          pauseListeningWhileAssistantSpeaks()
          setVoiceStatus('Speaking answer...')
        }
        audio.onended = () => {
          if (audioRef.current === audio) audioRef.current = null
          finishPlayback()
        }
        audio.onerror = () => {
          if (messageId) {
            setSpeakingMessageId((current: string | null) => (current === messageId ? null : current))
          } else {
            setSpeakingMessageId(null)
          }
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
    [browserSpeechSupported, conversationModeEnabled, deepgramEnabled, interruptAssistantTurn, localTtsEnabled, localTtsReady, pauseListeningWhileAssistantSpeaks, primeBrowserSpeech, resetDeepgramKeyGate, shouldReopenDeepgramKeyGate, stopDeepgramSession]
  )

  const speakMessage = useCallback(
    async (message: MessageType) => {
      const spokenText = buildSpokenText(message)
      if (!spokenText) return
      await speakText(spokenText, {
        messageId: message.id,
        endStatus: 'Answer ready',
        resumeConversation: conversationModeEnabled,
      })
    },
    [conversationModeEnabled, speakText]
  )

  useEffect(() => {
    if (!pendingAutoSpeakMessage || !voiceLoopEnabled || isLoading) return
    let cancelled = false
    const timeoutId = window.setTimeout(() => {
      if (cancelled) return
      speakMessage(pendingAutoSpeakMessage).finally(() => {
        setPendingAutoSpeakMessage((current: MessageType | null) => (current?.id === pendingAutoSpeakMessage.id ? null : current))
      })
    }, 50)
    return () => {
      cancelled = true
      window.clearTimeout(timeoutId)
    }
  }, [isLoading, pendingAutoSpeakMessage, speakMessage, voiceLoopEnabled])

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
      if (!trimmed) return
      if (requiresUserKey) {
        setApiKeyGateError('Enter your Anthropic API key to continue.')
        return
      }

      // Any new submit should interrupt the current assistant turn first.
      interruptAssistantTurn()
      pauseListeningWhileAssistantSpeaks()

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
      ttsBufferRef.current = ''
      setMessages((prev) => [...prev, userMsg, assistantMsg])
      const historySnapshot = messages.map((m) => ({ role: m.role, content: m.text }))

      if (voiceLoopEnabled && deepgramEnabled) {
        const player = new TTSPlayer(
          () => {
            assistantSpeakingRef.current = true
            setSpeakingMessageId(assistantId)
            setVoiceStatus('Speaking answer...')
          },
          () => {
            assistantSpeakingRef.current = false
            setSpeakingMessageId((prev: string | null) => (prev === assistantId ? null : prev))
            setVoiceStatus('Answer ready')
            if (conversationModeEnabled) {
              window.setTimeout(() => {
                startConversationListeningRef.current?.()
              }, 150)
            }
          },
          (message) => {
            if (shouldReopenDeepgramKeyGate(message)) {
              resetDeepgramKeyGate('This Deepgram API key cannot be used right now. Enter another Deepgram API key.')
              setVoiceStatus('Deepgram key needs attention')
            }
            setVoiceError(message)
          },
          undefined,
          () => audioCtxRef.current,
        )
        deepgramTtsPlayerRef.current = player
      }

      const streamController = new AbortController()
      streamAbortControllerRef.current = streamController

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
              if (voiceLoopEnabled && deepgramEnabled && deepgramTtsPlayerRef.current) {
                ttsBufferRef.current += parsed.text
                const { sentences, remaining } = drainSentences(ttsBufferRef.current)
                ttsBufferRef.current = remaining
                for (const s of sentences) {
                  deepgramTtsPlayerRef.current.enqueue(cleanSpokenText(s))
                }
              }
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
              streamAbortControllerRef.current = null
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
              if (voiceLoopEnabled && deepgramEnabled && deepgramTtsPlayerRef.current) {
                const remaining = ttsBufferRef.current.trim()
                if (remaining) {
                  deepgramTtsPlayerRef.current.enqueue(cleanSpokenText(remaining))
                }
                ttsBufferRef.current = ''
              } else if (voiceLoopEnabled && repairedText.trim()) {
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
              streamAbortControllerRef.current = null
              streamingTextRef.current = ''
              streamingArtifactsRef.current = []
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false, error: event.message } : m))
              )
              setIsLoading(false)
              if (shouldReopenApiKeyGate(event.message)) {
                resetApiKeyGate('This API key cannot be used for requests right now. Enter another Anthropic API key.', {
                  preserveDraft: false,
                })
              }
              setVoiceError(event.message)
            },
          },
          { voiceMode: true, signal: streamController.signal }
        )
      } catch (err) {
        streamAbortControllerRef.current = null
        if (err instanceof DOMException && err.name === 'AbortError') {
          const repairedText = repairConcatenatedWords(streamingTextRef.current)
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, isStreaming: false, text: repairedText || m.text, spokenText: repairedText || m.text }
                : m
            )
          )
          streamingTextRef.current = ''
          streamingArtifactsRef.current = []
          ttsBufferRef.current = ''
          setIsLoading(false)
          return
        }
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
        if (err instanceof Error && shouldReopenApiKeyGate(err.message)) {
          resetApiKeyGate('This API key cannot be used for requests right now. Enter another Anthropic API key.', {
            preserveDraft: false,
          })
        }
        setVoiceError(err instanceof Error ? err.message : 'Connection failed')
      }
    },
    [conversationModeEnabled, deepgramEnabled, interruptAssistantTurn, messages, pauseListeningWhileAssistantSpeaks, primeBrowserSpeech, requiresUserKey, resetApiKeyGate, resetDeepgramKeyGate, shouldReopenApiKeyGate, shouldReopenDeepgramKeyGate, voiceLoopEnabled]
  )

  const stopListening = useCallback(() => {
    stopDeepgramSession()
    recognitionRef.current?.stop()
    setIsListening(false)
  }, [stopDeepgramSession])

  const finishVoiceCapture = useCallback(() => {
    const transcript = transcriptRef.current.trim()
    pendingAutoSendRef.current = false
    setIsListening(false)

    if (!transcript) {
      transcriptRef.current = ''
      setInput('')
      if (conversationModeEnabled && deepgramEnabled) {
        conversationArmedRef.current = true
        setVoiceStatus(idleVoiceStatus(true, true))
        window.setTimeout(() => {
          startConversationListeningRef.current?.()
        }, 150)
        return
      }
      setVoiceStatus('No speech captured')
      return
    }

    if (processVoiceCommand(transcript)) {
      setInput('')
      transcriptRef.current = ''
      if (conversationModeEnabled && conversationArmedRef.current) {
        window.setTimeout(() => {
          startConversationListeningRef.current?.()
        }, 150)
      }
      return
    }

    setInput(transcript)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
    transcriptRef.current = ''
    handleSubmit(transcript)
  }, [conversationModeEnabled, deepgramEnabled, handleSubmit, processVoiceCommand, speakText])

  finishVoiceCaptureRef.current = finishVoiceCapture

  const startConversationListening = useCallback((force = false) => {
    if ((!force && !conversationModeEnabled) || !deepgramEnabled || isLoading) return
    if (deepgramModeRef.current === 'conversation' && deepgramAsrRef.current) return
    if (assistantSpeakingRef.current || speakingMessageId !== null || deepgramTtsPlayerRef.current?.isSpeaking) return

    conversationArmedRef.current = true
    setVoiceError(null)
    transcriptRef.current = ''
    deepgramModeRef.current = 'conversation'
    setIsWakeListening(false)
    setIsListening(true)
    setVoiceStatus('Listening...')

    const asr = new DeepgramASR()
    deepgramAsrRef.current = asr
    asr.start(
      (result) => {
        const transcript = result.transcript.trim()
        if (!transcript) {
          if (result.speechFinal) {
            stopDeepgramSession()
            setVoiceStatus('Listening...')
            finishVoiceCaptureRef.current?.()
          }
          return
        }

        const normalizedTranscript = normalizeVoiceCommand(transcript)
        const interruptionGuard = interruptionGuardRef.current
        const shortTranscript = normalizedTranscript.length <= 32
        const likelyEcho =
          Boolean(interruptionGuard) &&
          normalizedTranscript.length > 0 &&
          shortTranscript &&
          interruptionGuard.includes(normalizedTranscript)

        if (likelyEcho && assistantSpeakingRef.current) {
          return
        }

        if (assistantSpeakingRef.current) {
          streamAbortControllerRef.current?.abort()
          streamAbortControllerRef.current = null
          interruptAssistantTurn()
        }

        transcriptRef.current = transcript
        setInput(transcript)
        if (textareaRef.current) {
          textareaRef.current.style.height = 'auto'
          textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
        }

        if (result.speechFinal) {
          stopDeepgramSession()
          setVoiceStatus('Processing your voice...')
          finishVoiceCaptureRef.current?.()
          return
        }

        if (conversationCommitTimeoutRef.current !== null) {
          window.clearTimeout(conversationCommitTimeoutRef.current)
          conversationCommitTimeoutRef.current = null
        }

        if (result.isFinal) {
          conversationCommitTimeoutRef.current = window.setTimeout(() => {
            conversationCommitTimeoutRef.current = null
            if (!transcriptRef.current.trim()) return
            stopDeepgramSession()
            setVoiceStatus('Processing your voice...')
            finishVoiceCaptureRef.current?.()
          }, 900)
        }
      },
      (error) => {
        stopDeepgramSession()
        setIsListening(false)
        if (shouldReopenDeepgramKeyGate(error)) {
          resetDeepgramKeyGate('This Deepgram API key cannot be used right now. Enter another Deepgram API key.')
        }
        setVoiceError(error)
        setVoiceStatus('Conversation listening unavailable')
      },
    )
  }, [conversationModeEnabled, deepgramEnabled, finishVoiceCapture, interruptAssistantTurn, isLoading, resetDeepgramKeyGate, shouldReopenDeepgramKeyGate, speakingMessageId, stopDeepgramSession])

  const startListening = useCallback(() => {
    if (!recognitionSupported || isLoading) return
    stopDeepgramSession()
    conversationArmedRef.current = false
    interruptAssistantTurn()
    primeBrowserSpeech()
    setVoiceError(null)
    setVoiceStatus('Listening...')

    if (deepgramEnabled) {
      stopDeepgramSession()
      deepgramModeRef.current = 'manual'
      const asr = new DeepgramASR()
      deepgramAsrRef.current = asr
      asr.start(
        (result) => {
          if (result.transcript) {
            transcriptRef.current = result.transcript
            setInput(result.transcript)
            if (textareaRef.current) {
              textareaRef.current.style.height = 'auto'
              textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
            }
          }
        },
        (error) => {
          stopDeepgramSession()
          if (shouldReopenDeepgramKeyGate(error)) {
            resetDeepgramKeyGate('This Deepgram API key cannot be used right now. Enter another Deepgram API key.')
          }
          setVoiceError(error)
          setVoiceStatus('Voice error — try again')
          setIsListening(false)
        },
      )
      setIsListening(true)
      return
    }

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
  }, [deepgramEnabled, finishVoiceCapture, interruptAssistantTurn, isLoading, primeBrowserSpeech, recognitionSupported, resetDeepgramKeyGate, shouldReopenDeepgramKeyGate, stopDeepgramSession])

  startConversationListeningRef.current = startConversationListening

  const beginVoiceCapture = useCallback(() => {
    if (!recognitionSupported || isLoading) return
    transcriptRef.current = ''
    pendingAutoSendRef.current = false
    startListening()
  }, [isLoading, recognitionSupported, startListening])

  const endVoiceCapture = useCallback(() => {
    if (!isListening) return
    if (deepgramAsrRef.current) {
      stopDeepgramSession()
      setIsListening(false)
      setVoiceStatus('Processing your voice...')
      finishVoiceCapture()
      return
    }
    pendingAutoSendRef.current = true
    stopListening()
    setVoiceStatus('Processing your voice...')
  }, [finishVoiceCapture, isListening, stopDeepgramSession, stopListening])

  const cancelHoldToTalk = useCallback(() => {
    pendingAutoSendRef.current = false
    transcriptRef.current = ''
    stopListening()
    setInput('')
    setVoiceStatus('Voice capture cancelled')
  }, [stopListening])

  useEffect(() => {
    if (!conversationModeEnabled || !deepgramEnabled || showApiKeyGate || isLoading || speakingMessageId !== null) {
      conversationArmedRef.current = false
      if (deepgramModeRef.current === 'conversation') {
        stopDeepgramSession()
        setIsListening(false)
      }
      return
    }
    if (deepgramModeRef.current === 'idle') {
      startConversationListening()
    }
  }, [conversationModeEnabled, deepgramEnabled, isLoading, showApiKeyGate, speakingMessageId, startConversationListening, stopDeepgramSession])

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
    setSettingsError(null)
    try {
      const result = await saveAnthropicCredential(nextValue)
      if (!result.saved) {
        const message = result.error ?? 'API key rejected. Double-check it and try again.'
        setApiKeyGateError(message)
        setSettingsError(message)
        return
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Could not reach the server. Try again.'
      setApiKeyGateError(message)
      setSettingsError(message)
      return
    } finally {
      setIsValidatingKey(false)
    }
    setForceApiKeyOverride(false)
    setApiKeyDraft('')
    setApiKeyGateError(null)
    setSettingsError(null)
    setVoiceError(null)
    await refreshCredentialStatus()
  }

  const handleApiKeyKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleApiKeySave()
    }
  }

  const handleDeepgramKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const nextValue = e.target.value
    setDeepgramKeyDraft(nextValue)
    if (deepgramKeyGateError) setDeepgramKeyGateError(null)
  }

  const handleDeepgramKeySave = async () => {
    const nextValue = deepgramKeyDraft.trim()
    if (!nextValue) {
      setDeepgramKeyGateError('Enter your Deepgram API key to continue.')
      return
    }
    setIsValidatingDeepgramKey(true)
    setDeepgramKeyGateError(null)
    setSettingsError(null)
    try {
      const result = await saveDeepgramCredential(nextValue)
      if (!result.saved) {
        const message = result.error ?? 'Deepgram API key rejected. Double-check it and try again.'
        setDeepgramKeyGateError(message)
        setSettingsError(message)
        return
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Could not reach the server. Try again.'
      setDeepgramKeyGateError(message)
      setSettingsError(message)
      return
    } finally {
      setIsValidatingDeepgramKey(false)
    }
    setDisableServerDeepgramForSession(false)
    setForceDeepgramKeyOverride(false)
    setDeepgramKeyDraft('')
    setDeepgramKeyGateError(null)
    setSettingsError(null)
    setVoiceError(null)
    await refreshCredentialStatus()
  }

  const handleDeepgramKeyKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleDeepgramKeySave()
    }
  }

  const handleAnthropicDelete = async () => {
    setIsDeletingAnthropicCredential(true)
    setSettingsError(null)
    try {
      await deleteAnthropicCredential()
      setForceApiKeyOverride(true)
      setApiKeyDraft('')
      await refreshCredentialStatus()
    } catch (e) {
      setSettingsError(e instanceof Error ? e.message : 'Could not remove the Anthropic key.')
    } finally {
      setIsDeletingAnthropicCredential(false)
    }
  }

  const handleDeepgramDelete = async () => {
    setIsDeletingDeepgramCredential(true)
    setSettingsError(null)
    try {
      await deleteDeepgramCredential()
      setDisableServerDeepgramForSession(true)
      setForceDeepgramKeyOverride(false)
      setDeepgramKeyDraft('')
      await refreshCredentialStatus()
    } catch (e) {
      setSettingsError(e instanceof Error ? e.message : 'Could not remove the Deepgram key.')
    } finally {
      setIsDeletingDeepgramCredential(false)
    }
  }

  const handleAuthSubmit = async () => {
    const username = authUsername.trim()
    const password = authPassword
    if (!username || !password) {
      setAuthError('Enter your username and password to continue.')
      return
    }
    setIsAuthenticating(true)
    setAuthError(null)
    try {
      const result = authMode === 'signup'
        ? await signup(username, password)
        : await login(username, password)
      if (!result.user) {
        setAuthError(result.error ?? 'Authentication failed.')
        return
      }
      setCurrentUser({ username: result.user.username })
      setAuthPassword('')
      setApiKeyGateError(null)
      setDeepgramKeyGateError(null)
      await refreshCredentialStatus()
    } catch (e) {
      setAuthError(e instanceof Error ? e.message : 'Authentication failed.')
    } finally {
      setIsAuthenticating(false)
      setAuthChecked(true)
      setHealthCheckDone(true)
    }
  }

  const handleLogout = async () => {
    setIsLoggingOut(true)
    try {
      await logout()
    } finally {
      setCurrentUser(null)
      setAuthUsername('')
      setAuthPassword('')
      setAuthError(null)
      setMessages([])
      setInput('')
      setServerHasAnthropicKey(false)
      setServerHasDeepgramKey(false)
      setAnthropicKeySource(null)
      setDeepgramKeySource(null)
      setIsSettingsOpen(false)
      setForceApiKeyOverride(false)
      setForceDeepgramKeyOverride(false)
      setDisableServerDeepgramForSession(false)
      setIsLoggingOut(false)
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
      {showAuthGate && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-slate-950/35 backdrop-blur-md px-4">
          <div className="w-full max-w-lg rounded-3xl border border-white/60 bg-white/95 shadow-2xl p-6">
            <div className="w-14 h-14 bg-slate-900 rounded-2xl flex items-center justify-center mb-4 text-2xl font-semibold text-white">
              V
            </div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Account</p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-900">
              {authMode === 'signup' ? 'Create your account' : 'Sign in to your workspace'}
            </h2>
            <p className="mt-3 text-sm leading-6 text-slate-500">
              Your API keys are saved per account in the database, so you only need to enter them once after signing in.
            </p>
            <div className="mt-6 grid gap-4">
              <div>
                <label className="block text-sm font-semibold text-slate-800 mb-2">Username</label>
                <input
                  type="text"
                  value={authUsername}
                  onChange={(e) => {
                    setAuthUsername(e.target.value)
                    if (authError) setAuthError(null)
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleAuthSubmit()
                    }
                  }}
                  autoComplete="username"
                  placeholder="your-name"
                  disabled={isAuthenticating}
                  className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-slate-500 focus:ring-2 focus:ring-slate-200 focus:outline-none disabled:bg-slate-50 disabled:text-slate-400"
                />
                {authMode === 'signup' && (
                  <p className="mt-2 text-xs text-slate-500">Use 3-32 characters: letters, numbers, dots, dashes, or underscores.</p>
                )}
              </div>
              <div>
                <label className="block text-sm font-semibold text-slate-800 mb-2">Password</label>
                <input
                  type="password"
                  value={authPassword}
                  onChange={(e) => {
                    setAuthPassword(e.target.value)
                    if (authError) setAuthError(null)
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleAuthSubmit()
                    }
                  }}
                  autoComplete={authMode === 'signup' ? 'new-password' : 'current-password'}
                  disabled={isAuthenticating}
                  className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-slate-500 focus:ring-2 focus:ring-slate-200 focus:outline-none disabled:bg-slate-50 disabled:text-slate-400"
                />
                {authMode === 'signup' && (
                  <p className="mt-2 text-xs text-slate-500">Use at least 8 characters.</p>
                )}
              </div>
            </div>
            {authError && <p className="mt-4 text-sm text-red-600">{authError}</p>}
            <button
              type="button"
              onClick={handleAuthSubmit}
              disabled={isAuthenticating}
              className="mt-5 w-full rounded-xl bg-slate-900 hover:bg-slate-800 disabled:bg-slate-400 text-white px-4 py-3 text-sm font-semibold transition-colors shadow-sm"
            >
              {isAuthenticating ? 'Please wait...' : authMode === 'signup' ? 'Create account' : 'Sign in'}
            </button>
            <div className="mt-4 flex items-center justify-center gap-2 text-sm text-slate-500">
              <span>{authMode === 'signup' ? 'Already have an account?' : 'New here?'}</span>
              <button
                type="button"
                onClick={() => {
                  setAuthMode((current) => (current === 'signup' ? 'login' : 'signup'))
                  setAuthError(null)
                }}
                className="font-semibold text-slate-900 hover:text-slate-700"
              >
                {authMode === 'signup' ? 'Sign in' : 'Create account'}
              </button>
            </div>
          </div>
        </div>
      )}
      {showApiKeyGate && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/30 backdrop-blur-md px-4">
          <div className="w-full max-w-lg rounded-3xl border border-white/60 bg-white/95 shadow-2xl p-6">
            <div className="w-14 h-14 bg-orange-100 rounded-2xl flex items-center justify-center mb-4 text-2xl font-semibold text-orange-600">
              V
            </div>
            <p className="text-xs font-semibold uppercase tracking-wide text-orange-600">Before you start</p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-900">Connect Anthropic to continue</h2>
            <p className="mt-3 text-sm leading-6 text-slate-500">
              This assistant needs an Anthropic API key to answer questions. Add your key once and the app will validate it
              before saving it in this backend's settings.
            </p>
            <div className="mt-5 rounded-2xl border border-orange-100 bg-orange-50/70 p-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-orange-700">What this does</p>
                  <p className="mt-1 text-sm text-slate-600">Enables chat, retrieval, and generated answers in the assistant.</p>
                </div>
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-orange-700">How it is handled</p>
                  <p className="mt-1 text-sm text-slate-600">The key stays hidden in the UI and is saved only after validation succeeds.</p>
                </div>
              </div>
            </div>
            <div className="mt-6">
              <label className="block text-sm font-semibold text-slate-800 mb-2">Anthropic API key</label>
              <p className="mb-3 text-xs text-slate-500">Paste a key that starts with `sk-ant-`.</p>
            </div>
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
                'Save key and continue'
              )}
            </button>
          </div>
        </div>
      )}
      {showDeepgramKeyGate && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/20 backdrop-blur-sm px-4">
          <div className="w-full max-w-lg rounded-3xl border border-white/60 bg-white/95 shadow-2xl p-6">
            <div className="w-14 h-14 bg-sky-100 rounded-2xl flex items-center justify-center mb-4 text-2xl font-semibold text-sky-600">
              D
            </div>
            <p className="text-xs font-semibold uppercase tracking-wide text-sky-600">Optional voice setup</p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-900">Add Deepgram for voice features</h2>
            <p className="mt-3 text-sm leading-6 text-slate-500">
              Deepgram powers voice input and spoken replies. You can add the key now, or keep using text chat and come
              back to voice later from API settings.
            </p>
            <div className="mt-5 rounded-2xl border border-sky-100 bg-sky-50/70 p-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-sky-700">Unlocks</p>
                  <p className="mt-1 text-sm text-slate-600">Tap-to-talk, conversation mode, and spoken assistant playback.</p>
                </div>
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-sky-700">Fallback</p>
                  <p className="mt-1 text-sm text-slate-600">Skip this for now and the app will stay available in text-only mode.</p>
                </div>
              </div>
            </div>
            <div className="mt-6">
              <label className="block text-sm font-semibold text-slate-800 mb-2">Deepgram API key</label>
              <p className="mb-3 text-xs text-slate-500">Add a valid Deepgram key to enable voice features.</p>
            </div>
            <input
              type="password"
              value={deepgramKeyDraft}
              onChange={handleDeepgramKeyChange}
              onKeyDown={handleDeepgramKeyKeyDown}
              placeholder="Deepgram API key"
              autoComplete="off"
              spellCheck={false}
              disabled={isValidatingDeepgramKey}
              className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-sky-400 focus:ring-2 focus:ring-sky-200 focus:outline-none disabled:bg-slate-50 disabled:text-slate-400"
            />
            {deepgramKeyGateError && <p className="mt-3 text-sm text-red-600">{deepgramKeyGateError}</p>}
            <div className="mt-4 flex gap-2">
              <button
                type="button"
                onClick={() => {
                  setDisableServerDeepgramForSession(true)
                  setForceDeepgramKeyOverride(false)
                  setDeepgramKeyDraft('')
                  setDeepgramKeyGateError(null)
                }}
                disabled={isValidatingDeepgramKey}
                className="flex-1 rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 disabled:text-slate-400"
              >
                Keep text only
              </button>
              <button
                type="button"
                onClick={handleDeepgramKeySave}
                disabled={isValidatingDeepgramKey}
                className="flex-1 rounded-xl bg-sky-500 hover:bg-sky-600 disabled:bg-sky-300 text-white px-4 py-3 text-sm font-semibold transition-colors shadow-sm flex items-center justify-center gap-2"
              >
                {isValidatingDeepgramKey ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Checking key...
                  </>
                ) : (
                  'Save key and enable voice'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
      {isSettingsOpen && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/30 backdrop-blur-md px-4">
          <div className="w-full max-w-2xl rounded-3xl border border-white/60 bg-white/95 shadow-2xl p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-slate-900">API Settings</h2>
                <p className="mt-1 text-sm text-slate-500">
                  Keys are stored for your account in the backend database, not in browser session storage.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setIsSettingsOpen(false)}
                className="rounded-xl border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50"
              >
                Close
              </button>
            </div>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="rounded-2xl border border-slate-200 bg-white p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">Anthropic</h3>
                    {/* <p className="mt-1 text-xs text-slate-500">
                      {serverHasAnthropicKey
                        ? `Configured via ${anthropicKeySource === 'env' ? '.env' : 'backend storage'}.`
                        : 'Not configured.'}
                    </p> */}
                  </div>
                  {/* <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${serverHasAnthropicKey ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500'}`}> */}
                    {/* {serverHasAnthropicKey ? (anthropicKeySource === 'env' ? '.env key' : 'Stored key') : 'Missing'} */}
                  {/* </span> */}
                </div>
                <input
                  type="password"
                  value={apiKeyDraft}
                  onChange={handleApiKeyChange}
                  onKeyDown={handleApiKeyKeyDown}
                  placeholder="sk-ant-..."
                  autoComplete="off"
                  spellCheck={false}
                  disabled={isValidatingKey}
                  className="mt-4 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-orange-400 focus:ring-2 focus:ring-orange-200 focus:outline-none disabled:bg-slate-50 disabled:text-slate-400"
                />
                {apiKeyGateError && <p className="mt-3 text-sm text-red-600">{apiKeyGateError}</p>}
                <div className="mt-4 flex gap-2">
                  <button
                    type="button"
                    onClick={handleApiKeySave}
                    disabled={isValidatingKey}
                    className="flex-1 rounded-xl bg-orange-500 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-orange-600 disabled:bg-orange-300"
                  >
                    {isValidatingKey ? 'Checking...' : serverHasAnthropicKey ? 'Replace key' : 'Save key'}
                  </button>
                  {anthropicKeySource === 'stored' && (
                    <button
                      type="button"
                      onClick={handleAnthropicDelete}
                      disabled={isDeletingAnthropicCredential}
                      className="rounded-xl border border-slate-300 px-4 py-3 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 disabled:text-slate-400"
                    >
                      {isDeletingAnthropicCredential ? 'Removing...' : 'Remove'}
                    </button>
                  )}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">Deepgram</h3>
                    {/* <p className="mt-1 text-xs text-slate-500">
                      {serverHasDeepgramKey
                        ? `Configured via ${deepgramKeySource === 'env' ? '.env' : 'backend storage'}.`
                        : 'Not configured.'}
                    </p> */}
                  </div>
                  {/* <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${serverHasDeepgramKey ? 'bg-sky-100 text-sky-700' : 'bg-slate-100 text-slate-500'}`}> */}
                    {/* {serverHasDeepgramKey ? (deepgramKeySource === 'env' ? '.env key' : 'Stored key') : 'Missing'} */}
                  {/* </span> */}
                </div>
                <input
                  type="password"
                  value={deepgramKeyDraft}
                  onChange={handleDeepgramKeyChange}
                  onKeyDown={handleDeepgramKeyKeyDown}
                  placeholder="Deepgram API key"
                  autoComplete="off"
                  spellCheck={false}
                  disabled={isValidatingDeepgramKey}
                  className="mt-4 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-sky-400 focus:ring-2 focus:ring-sky-200 focus:outline-none disabled:bg-slate-50 disabled:text-slate-400"
                />
                {deepgramKeyGateError && <p className="mt-3 text-sm text-red-600">{deepgramKeyGateError}</p>}
                <div className="mt-4 flex gap-2">
                  <button
                    type="button"
                    onClick={handleDeepgramKeySave}
                    disabled={isValidatingDeepgramKey}
                    className="flex-1 rounded-xl bg-sky-500 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-sky-600 disabled:bg-sky-300"
                  >
                    {isValidatingDeepgramKey ? 'Checking...' : serverHasDeepgramKey ? 'Replace key' : 'Save key'}
                  </button>
                  {deepgramKeySource === 'stored' && (
                    <button
                      type="button"
                      onClick={handleDeepgramDelete}
                      disabled={isDeletingDeepgramCredential}
                      className="rounded-xl border border-slate-300 px-4 py-3 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 disabled:text-slate-400"
                    >
                      {isDeletingDeepgramCredential ? 'Removing...' : 'Remove'}
                    </button>
                  )}
                </div>
              </div>
            </div>
            {settingsError && <p className="mt-4 text-sm text-red-600">{settingsError}</p>}
          </div>
        </div>
      )}
      <header className={`flex-shrink-0 bg-slate-900 text-white px-4 py-2 flex items-center justify-between gap-4 shadow-lg z-10 transition-all ${showApiKeyGate || showAuthGate ? 'blur-sm pointer-events-none select-none' : ''}`}>
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
          {currentUser && (
            <span className="hidden sm:inline text-slate-400 truncate max-w-[220px]">{currentUser.username}</span>
          )}
          <button
            type="button"
            onClick={() => {
              setSettingsError(null)
              setIsSettingsOpen(true)
            }}
            className="rounded-full px-3 py-1.5 border border-slate-700 bg-slate-800 text-slate-200 transition-colors hover:bg-slate-700"
          >
            API settings
          </button>
          <button
            type="button"
            onClick={() => {
              if (!deepgramEnabled) return
              setConversationModeEnabled((value) => {
                const next = !value
                localStorage.setItem('conversationMode', String(next))
                if (next) {
                  setAutoSpeak(true)
                  localStorage.setItem('autoSpeak', 'true')
                  startConversationListening(true)
                }
                if (!next) {
                  conversationArmedRef.current = false
                  stopListening()
                  setVoiceStatus(idleVoiceStatus(false, deepgramEnabled))
                }
                return next
              })
            }}
            disabled={!deepgramEnabled || requiresUserKey}
            className={`rounded-full px-3 py-1.5 border transition-colors ${
              conversationModeEnabled && deepgramEnabled ? 'bg-white text-slate-900 border-white' : 'bg-slate-800 border-slate-700 text-slate-300'
            } disabled:opacity-50`}
          >
            Conversation {conversationModeEnabled ? 'on' : 'off'}
          </button>
          <button
            type="button"
            onClick={() => {
              if (!speechSupported) return
              if (autoSpeak) stopSpeaking()
              setAutoSpeak((value) => {
                const next = !value
                localStorage.setItem('autoSpeak', String(next))
                return next
              })
            }}
            disabled={!speechSupported}
            className={`rounded-full px-3 py-1.5 border transition-colors ${
              (autoSpeak || conversationModeEnabled) && speechSupported ? 'bg-white text-slate-900 border-white' : 'bg-slate-800 border-slate-700 text-slate-300'
            } disabled:opacity-50`}
          >
            Auto speak {autoSpeak || conversationModeEnabled ? 'on' : 'off'}
          </button>
          {currentUser && (
            <button
              type="button"
              onClick={handleLogout}
              disabled={isLoggingOut}
              className="rounded-full px-3 py-1.5 border border-slate-700 bg-slate-800 text-slate-200 transition-colors hover:bg-slate-700 disabled:opacity-50"
            >
              {isLoggingOut ? 'Signing out...' : 'Sign out'}
            </button>
          )}
        </div>
      </header>

      <div className={`flex flex-1 overflow-hidden transition-all ${showApiKeyGate || showAuthGate ? 'blur-sm pointer-events-none select-none' : ''}`}>
        <div className="flex flex-col w-full min-w-0">
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4 chat-scroll">
            {!hasMessages && (
              <div className="flex flex-col items-center justify-center h-full text-center px-4 py-6">
                <div className="w-12 h-12 bg-orange-100 rounded-2xl flex items-center justify-center mb-3 text-lg font-semibold text-orange-600">
                  P
                </div>
                <h2 className="text-base font-semibold text-slate-700 mb-1">Vulcan OmniPro Assistant</h2>
                <p className="text-xs text-slate-400 max-w-sm mb-5">
                  Ask anything about setup, troubleshooting, or settings. Tap the mic, or turn on conversation mode and just talk.
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
                  // placeholder={isListening || isWakeListening ? 'Listening...' : 'Ask about the Vulcan OmniPro 220...'}
                  rows={1}
                  disabled={requiresUserKey}
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
                  disabled={!recognitionSupported || isLoading || requiresUserKey}
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
                  disabled={!input.trim() || requiresUserKey}
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
                  {/* <span>{speechSupported ? voiceEngineLabel : 'Speech playback not supported'}</span> */}
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
              {(isListening || isWakeListening) && (
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
