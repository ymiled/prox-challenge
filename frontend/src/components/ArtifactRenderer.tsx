import type { Artifact } from '../lib/types'

export type ArtifactRendererVariant = 'panel' | 'inline'

interface Props {
  artifact: Artifact
  /** `inline` = shorter iframe for embedding inside the chat column. */
  variant?: ArtifactRendererVariant
}

function buildReactSrcdoc(code: string): string {
  const cleanCode = code
    .replace(/export\s+default\s+function\s+\w+\s*\(/, 'function App(')
    .replace(/export\s+default\s+/g, '')

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <script src="https://unpkg.com/react@18/umd/react.development.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { margin: 0; background: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
${cleanCode}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
  </script>
</body>
</html>`
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function buildMermaidSrcdoc(code: string): string {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true, theme: 'default' });
  </script>
  <style>
    body { margin: 0; padding: 16px; background: #ffffff; font-family: system-ui, -apple-system, sans-serif; }
  </style>
</head>
<body>
  <pre class="mermaid">${escapeHtml(code)}</pre>
</body>
</html>`
}

function buildMarkdownSrcdoc(markdown: string): string {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; padding: 20px; background: #ffffff; color: #0f172a; font-family: system-ui, -apple-system, sans-serif; }
    pre { white-space: pre-wrap; line-height: 1.5; }
  </style>
</head>
<body>
  <pre>${escapeHtml(markdown)}</pre>
</body>
</html>`
}

export default function ArtifactRenderer({ artifact, variant = 'panel' }: Props) {
  const iframeHeight = variant === 'inline' ? '380px' : '480px'
  const svgMinH = variant === 'inline' ? 'min-h-[240px]' : 'min-h-[320px]'

  if (artifact.type === 'react') {
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={buildReactSrcdoc(artifact.content)}
        title={artifact.title}
        className="w-full border-0"
        style={{ height: iframeHeight }}
      />
    )
  }

  if (artifact.type === 'svg') {
    return (
      <div className={`p-4 flex items-center justify-center bg-white ${svgMinH}`}>
        <div className="max-w-full overflow-auto" dangerouslySetInnerHTML={{ __html: artifact.content }} />
      </div>
    )
  }

  if (artifact.type === 'html') {
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={artifact.content}
        title={artifact.title}
        className="w-full border-0"
        style={{ height: iframeHeight }}
      />
    )
  }

  if (artifact.type === 'mermaid') {
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={buildMermaidSrcdoc(artifact.content)}
        title={artifact.title}
        className="w-full border-0"
        style={{ height: iframeHeight }}
      />
    )
  }

  if (artifact.type === 'markdown') {
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={buildMarkdownSrcdoc(artifact.content)}
        title={artifact.title}
        className="w-full border-0"
        style={{ height: iframeHeight }}
      />
    )
  }

  if (artifact.type === 'code') {
    return (
      <div className="p-4 overflow-auto">
        <pre className="text-xs text-slate-700 bg-slate-50 rounded-lg p-4 overflow-auto whitespace-pre-wrap">
          {artifact.content}
        </pre>
      </div>
    )
  }

  let pretty = artifact.content
  try {
    pretty = JSON.stringify(JSON.parse(artifact.content), null, 2)
  } catch {
    // show as-is when not valid JSON
  }

  return (
    <div className="p-4 overflow-auto">
      <pre className="text-xs text-slate-700 bg-slate-50 rounded-lg p-4 overflow-auto whitespace-pre-wrap">
        {pretty}
      </pre>
    </div>
  )
}
