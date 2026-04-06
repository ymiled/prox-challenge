export type MessageRole = 'user' | 'assistant'

export interface PageRef {
  doc: string
  page: number
  score?: number
  section?: string
  excerpt?: string
}

export interface ManualImage {
  doc: string
  page: number
  caption: string
  url: string
  src?: string
}

export interface Artifact {
  id: string
  mimeType: string
  type: 'react' | 'svg' | 'json' | 'html' | 'code' | 'markdown' | 'mermaid'
  title: string
  content: string
  url: string
}

export interface Message {
  id: string
  role: MessageRole
  text: string
  images: ManualImage[]
  artifacts: Artifact[]
  citations: PageRef[]
  isStreaming: boolean
  spokenText?: string
  error?: string
}

// SSE event shapes from the backend
export interface TextDeltaEvent {
  type: 'text_delta'
  content: string
}

export interface ImageEvent {
  type: 'image'
  doc: string
  page: number
  caption: string
  url: string
  src?: string
}

export interface ArtifactEvent {
  type: 'artifact'
  id: string
  artifact_type: 'react' | 'svg' | 'json' | 'html' | 'code' | 'markdown' | 'mermaid'
  title: string
  content: string
  url: string
}

export interface DoneEvent {
  type: 'done'
  citations: PageRef[]
  debug?: Record<string, unknown>
}

export interface ErrorEvent {
  type: 'error'
  message: string
  traceback?: string
}

export type SSEEvent = TextDeltaEvent | ImageEvent | ArtifactEvent | DoneEvent | ErrorEvent
