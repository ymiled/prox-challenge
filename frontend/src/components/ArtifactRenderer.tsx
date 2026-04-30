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
    body { margin: 0; background: #0b0e13; color: #e5e7eb; font-family: Inter, system-ui, -apple-system, sans-serif; }
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
    mermaid.initialize({ startOnLoad: true, theme: 'dark' });
  </script>
  <style>
    body { margin: 0; padding: 16px; background: #0b0e13; color: #e5e7eb; font-family: Inter, system-ui, -apple-system, sans-serif; }
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
    body { margin: 0; padding: 20px; background: #0b0e13; color: #e5e7eb; font-family: Inter, system-ui, -apple-system, sans-serif; }
    pre { white-space: pre-wrap; line-height: 1.5; }
  </style>
</head>
<body>
  <pre>${escapeHtml(markdown)}</pre>
</body>
</html>`
}

function buildSvgDataUrl(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

export default function ArtifactRenderer({ artifact, variant = 'panel' }: Props) {
  const iframeHeight = variant === 'inline' ? '400px' : '500px'
  const svgMinH = variant === 'inline' ? 'min-h-[260px]' : 'min-h-[340px]'

  if (artifact.type === 'react') {
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={buildReactSrcdoc(artifact.content)}
        title={artifact.title}
        className="w-full border-0 bg-[#0b0e13]"
        style={{ height: iframeHeight }}
      />
    )
  }

  if (artifact.type === 'svg') {
    return (
      <div className={`flex items-center justify-center bg-[#0b0e13] p-5 ${svgMinH}`}>
        <img
          src={buildSvgDataUrl(artifact.content)}
          alt={artifact.title}
          className="block max-w-full h-auto"
          style={{ maxHeight: variant === 'inline' ? '320px' : '420px' }}
        />
      </div>
    )
  }

  if (artifact.type === 'html') {
    return (
      <iframe
        sandbox="allow-scripts"
        srcDoc={artifact.content}
        title={artifact.title}
        className="w-full border-0 bg-[#0b0e13]"
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
        className="w-full border-0 bg-[#0b0e13]"
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
        className="w-full border-0 bg-[#0b0e13]"
        style={{ height: iframeHeight }}
      />
    )
  }

  if (artifact.type === 'code') {
    return (
      <div className="overflow-auto p-4">
        <pre className="overflow-auto whitespace-pre-wrap rounded-xl border border-white/10 bg-white/[0.03] p-4 font-mono text-xs leading-6 text-slate-300">
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
    <div className="overflow-auto p-4">
      <pre className="overflow-auto whitespace-pre-wrap rounded-xl border border-white/10 bg-white/[0.03] p-4 font-mono text-xs leading-6 text-slate-300">
        {pretty}
      </pre>
    </div>
  )
}
