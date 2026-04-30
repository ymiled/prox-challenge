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
    <div className="mt-3 flex flex-wrap gap-2">
      {unique.slice(0, 5).map((c) => (
        <a
          key={`${c.doc}-${c.page}`}
          href={apiUrl(`/pages/${c.doc}/${c.page}`)}
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 font-mono text-[11px] text-slate-400 transition-all hover:border-white/20 hover:bg-white/[0.06] hover:text-slate-200"
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
    <div className="mt-4 flex flex-wrap gap-3">
      {images.map((img, index) => (
        <a
          key={index}
          href={img.url}
          target="_blank"
          rel="noopener noreferrer"
          className="group block overflow-hidden rounded-2xl border border-white/10 bg-[#11141a] shadow-[0_10px_30px_rgba(0,0,0,0.18)] transition-all duration-150 hover:-translate-y-0.5 hover:border-white/20 hover:bg-[#141821]"
          style={{ width: '148px' }}
        >
          <img src={img.src ?? img.url} alt={img.caption} className="h-[96px] w-full object-cover" />
          <p className="truncate border-t border-white/10 bg-[#11141a] px-3 py-2 text-xs text-slate-400 group-hover:text-slate-200">{img.caption}</p>
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
          className="overflow-hidden rounded-2xl border border-white/10 bg-[#0f1218] shadow-[0_12px_32px_rgba(0,0,0,0.2)]"
        >
          <div className="flex items-center justify-between gap-2 border-b border-white/10 bg-white/[0.03] px-4 py-3">
            <div className="flex items-center gap-2 min-w-0">
              <span className="flex-shrink-0 rounded-md border border-white/10 bg-white/[0.04] px-1.5 py-0.5 font-mono text-[10px] font-semibold text-slate-400">
                {artifactIcon(artifact.type)}
              </span>
              <span className="truncate text-xs font-semibold text-slate-100">{artifact.title}</span>
              <span className="hidden flex-shrink-0 font-mono text-[10px] uppercase tracking-wide text-slate-500 sm:inline">
                {artifact.type}
              </span>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0">
              <a
                href={artifact.url}
                download={artifactFilename(artifact)}
                className="font-mono text-[11px] font-medium text-slate-400 transition-colors hover:text-white"
              >
                Download
              </a>
            </div>
          </div>
          <div className="bg-[#0b0e13]">
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
        <div className="max-w-[80%] rounded-[20px] rounded-br-md border border-white/10 bg-[#171b23] px-4 py-3 text-sm leading-6 text-slate-100 shadow-[0_14px_34px_rgba(0,0,0,0.18)]">
          <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.12em] text-slate-500">You</div>
          <div>{message.text}</div>
        </div>
      </div>
    )
  }

  const hasArtifacts = message.artifacts.length > 0
  return (
    <div className="flex justify-start w-full min-w-0">
      <div className={hasArtifacts ? 'max-w-full min-w-0 flex-1 space-y-3' : 'max-w-[88%] space-y-2'}>
        <div className="flex items-start gap-3 min-w-0">
          <div
            className="mt-1 flex h-8 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] px-2.5 font-mono text-[11px] font-semibold leading-none text-slate-300"
            title="Assistant"
            aria-hidden
          >
            Vulcan
          </div>
          <div className="min-w-0 flex-1 rounded-[20px] rounded-tl-md border border-white/10 bg-[#11151c] px-4 py-3.5 shadow-[0_14px_34px_rgba(0,0,0,0.18)]">
            <div className="mb-2 flex items-center justify-between gap-3">
              <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-slate-500">Assistant</div>
              {canSpeak && message.text && !message.isStreaming && (
                <button
                  type="button"
                  onClick={() => (isSpeaking ? onStopSpeaking?.() : onSpeak?.(message))}
                  className="rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 font-mono text-[11px] font-medium text-slate-400 transition-all hover:border-white/20 hover:bg-white/[0.06] hover:text-white"
                >
                  {isSpeaking ? 'Stop' : 'Speak'}
                </button>
              )}
            </div>
            <div className="flex items-start justify-between gap-3">
              <p className="flex-1 whitespace-pre-wrap text-sm leading-7 text-slate-100">
                {message.text || (message.isStreaming ? '' : '-')}
                {message.isStreaming && (
                  <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse align-middle bg-slate-500" />
                )}
              </p>
            </div>
            {message.error && (
              <p className="mt-3 rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                Error: {message.error}
              </p>
            )}
          </div>
        </div>

        <div className="w-full min-w-0 pl-11">
          <ManualImages images={message.images} />
          <InlineArtifactPreviews artifacts={message.artifacts} />
          {!message.isStreaming && <CitationPills citations={message.citations} />}
        </div>
      </div>
    </div>
  )
}
