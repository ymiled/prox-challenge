import ArtifactRenderer from './ArtifactRenderer'
import type { Artifact, ManualImage, Message as MessageType, PageRef } from '../lib/types'
import { apiUrl } from '../lib/api'

interface Props {
  message: MessageType
  canSpeak?: boolean
  isSpeaking?: boolean
  onSpeak?: (message: MessageType) => void
  onStopSpeaking?: () => void
}

function artifactIcon(type: string) {
  if (type === 'react') return 'UI'
  if (type === 'svg') return 'SVG'
  if (type === 'html') return 'HTML'
  if (type === 'markdown') return 'MD'
  if (type === 'mermaid') return 'MM'
  if (type === 'code') return 'CODE'
  return 'DATA'
}

function artifactFilename(artifact: Artifact): string {
  const slug =
    artifact.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || artifact.id

  const extByType: Record<Artifact['type'], string> = {
    react: 'txt',
    svg: 'svg',
    html: 'html',
    json: 'json',
    code: 'txt',
    markdown: 'md',
    mermaid: 'mmd',
  }

  return `${slug}.${extByType[artifact.type]}`
}

function CitationPills({ citations }: { citations: PageRef[] }) {
  if (!citations.length) return null
  const seen = new Set<string>()
  const unique = citations.filter((c) => {
    const key = `${c.doc}:${c.page}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {unique.slice(0, 5).map((c) => (
        <a
          key={`${c.doc}-${c.page}`}
          href={apiUrl(`/pages/${c.doc}/${c.page}`)}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-slate-400 bg-slate-100 hover:bg-slate-200 px-2 py-0.5 rounded-full transition-colors"
          title={c.excerpt ?? ''}
        >
          {c.doc} p.{c.page}
        </a>
      ))}
    </div>
  )
}

function ManualImages({ images }: { images: ManualImage[] }) {
  if (!images.length) return null
  return (
    <div className="flex gap-2 flex-wrap mt-2">
      {images.map((img, index) => (
        <a
          key={index}
          href={img.url}
          target="_blank"
          rel="noopener noreferrer"
          className="group block border border-slate-200 rounded-lg overflow-hidden hover:border-orange-400 hover:shadow-md transition-all"
          style={{ width: '120px' }}
        >
          <img src={img.src ?? img.url} alt={img.caption} className="w-full object-cover" style={{ height: '84px' }} />
          <p className="text-xs text-slate-500 group-hover:text-orange-600 px-2 py-1 truncate bg-white">{img.caption}</p>
        </a>
      ))}
    </div>
  )
}

function InlineArtifactPreviews({ artifacts }: { artifacts: Artifact[] }) {
  if (!artifacts.length) return null
  return (
    <div className="mt-3 space-y-3 w-full min-w-0">
      {artifacts.map((artifact) => (
        <div
          key={artifact.id}
          className="rounded-xl border border-slate-200 bg-white overflow-hidden shadow-sm ring-1 ring-slate-100"
        >
          <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-slate-100 bg-slate-50/90">
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-[10px] font-semibold text-slate-500 bg-white border border-slate-200 rounded px-1.5 py-0.5 flex-shrink-0">
                {artifactIcon(artifact.type)}
              </span>
              <span className="text-xs font-semibold text-slate-800 truncate">{artifact.title}</span>
              <span className="text-[10px] uppercase tracking-wide text-slate-400 flex-shrink-0 hidden sm:inline">
                {artifact.type}
              </span>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0">
              <a
                href={artifact.url}
                download={artifactFilename(artifact)}
                className="text-[11px] font-medium text-slate-500 hover:text-orange-600 transition-colors"
              >
                Download
              </a>
            </div>
          </div>
          <div className="bg-slate-50">
            <ArtifactRenderer artifact={artifact} variant="inline" />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function Message({
  message,
  canSpeak = false,
  isSpeaking = false,
  onSpeak,
  onStopSpeaking,
}: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-orange-500 text-white rounded-2xl rounded-tr-md px-4 py-2.5 max-w-[78%] text-sm leading-relaxed shadow-sm">
          {message.text}
        </div>
      </div>
    )
  }

  const hasArtifacts = message.artifacts.length > 0
  return (
    <div className="flex justify-start w-full min-w-0">
      <div className={hasArtifacts ? 'max-w-full min-w-0 flex-1 space-y-2' : 'max-w-[85%] space-y-1'}>
        <div className="flex items-start gap-2 min-w-0">
          <div
            className="mt-1 w-7 h-7 rounded-full bg-orange-100 border border-orange-200/80 flex items-center justify-center flex-shrink-0 text-[11px] font-semibold leading-none text-orange-700"
            title="Assistant"
            aria-hidden
          >
            AI
          </div>
          <div className="bg-white rounded-2xl rounded-tl-md px-4 py-3 shadow-sm border border-slate-100 flex-1 min-w-0">
            <div className="flex items-start justify-between gap-3">
              <p className="text-sm text-slate-800 whitespace-pre-wrap leading-relaxed flex-1">
                {message.text || (message.isStreaming ? '' : '-')}
                {message.isStreaming && (
                  <span className="inline-block w-0.5 h-4 bg-slate-400 ml-0.5 animate-pulse align-middle" />
                )}
              </p>
              {canSpeak && message.text && !message.isStreaming && (
                <button
                  type="button"
                  onClick={() => (isSpeaking ? onStopSpeaking?.() : onSpeak?.(message))}
                  className="text-xs font-medium text-slate-500 hover:text-orange-600 transition-colors flex-shrink-0"
                >
                  {isSpeaking ? 'Stop' : 'Speak'}
                </button>
              )}
            </div>
            {message.error && (
              <p className="text-xs text-red-500 mt-2 bg-red-50 rounded px-2 py-1">
                Error: {message.error}
              </p>
            )}
          </div>
        </div>

        <div className="pl-8 w-full min-w-0">
          <ManualImages images={message.images} />
          <InlineArtifactPreviews artifacts={message.artifacts} />
          {!message.isStreaming && <CitationPills citations={message.citations} />}
        </div>
      </div>
    </div>
  )
}
