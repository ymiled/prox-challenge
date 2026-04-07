import type { Artifact } from './types'
import { apiUrl } from './api'

interface ParseResult {
  text: string
  artifacts: Artifact[]
}

function mapArtifactType(mimeType: string): Artifact['type'] {
  if (mimeType === 'application/vnd.ant.react') return 'react'
  if (mimeType === 'image/svg+xml') return 'svg'
  if (mimeType === 'text/html') return 'html'
  if (mimeType === 'application/vnd.ant.code') return 'code'
  if (mimeType === 'text/markdown') return 'markdown'
  if (mimeType === 'application/vnd.ant.mermaid') return 'mermaid'
  return 'json'
}

export function parseArtifacts(content: string): ParseResult {
  const artifacts: Artifact[] = []
  const pattern =
    /<antArtifact\s+identifier="([^"]+)"\s+type="([^"]+)"\s+title="([^"]+)">([\s\S]*?)<\/antArtifact>/g

  let text = content
  let match: RegExpExecArray | null
  while ((match = pattern.exec(content)) !== null) {
    const [, id, mimeType, title, artifactContent] = match
    artifacts.push({
      id,
      mimeType,
      type: mapArtifactType(mimeType),
      title,
      content: artifactContent.trim(),
      url: apiUrl(`/artifacts/${id}`),
    })
  }

  // Use a space when stripping artifacts so "is </antArtifact>Based" does not become "isBased"
  pattern.lastIndex = 0
  text = text.replace(pattern, ' ')
  text = text.replace(/\n{3,}/g, '\n\n')
  text = text.replace(/[ \t]{2,}/g, ' ')
  text = text.replace(/\n[ \t]+/g, '\n')
  // Preserve leading whitespace on this chunk (needed when joining stream deltas)
  text = text.trimEnd()
  return { text, artifacts }
}
