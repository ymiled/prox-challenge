/**
 * Merge streamed assistant text chunks. The Anthropic stream sometimes splits
 * between tokens without a leading space on the next chunk ("The" + "manual" →
 * "Themanual"). Insert a space only when both sides look like they should be
 * separate words/tokens.
 */
export function appendStreamText(prev: string, next: string): string {
  if (!next) return prev
  if (!prev) return next

  const last = prev.charAt(prev.length - 1)
  const first = next.charAt(0)
  if (!last || !first) return prev + next
  if (/\s/.test(last) || /\s/.test(first)) return prev + next

  const lastLo = /[a-z]/.test(last)
  const lastAl = /[a-zA-Z]/.test(last)
  const lastDig = /[0-9]/.test(last)
  const firstLo = /[a-z]/.test(first)
  const firstAl = /[a-zA-Z]/.test(first)
  const firstDig = /[0-9]/.test(first)
  const firstUp = /[A-Z]/.test(first)

  if (lastLo && firstLo) return `${prev} ${next}` // The + manual, is + available
  if (lastDig && firstAl) return `${prev} ${next}` // 43 + references
  if (lastAl && firstDig) return `${prev} ${next}` // page + 37
  if (lastLo && firstUp) return `${prev} ${next}` // is + Based

  return prev + next
}

/**
 * Fixes words concatenated inside a single model chunk (or missed stream joins).
 * Patterns are conservative to avoid breaking product names like OmniPro where possible.
 */
export function repairConcatenatedWords(text: string): string {
  if (!text || text.length < 4) return text

  const rules: Array<[RegExp, string]> = [
    [/what isavailable/gi, 'what is available'],
    [/What isavailable/g, 'What is available'],
    [/isavailable/gi, 'is available'],
    [/Themanual/g, 'The manual'],
    [/themanual/gi, 'the manual'],
    [/iswhat/gi, 'is what'],
    [/based onwhat/gi, 'based on what'],
    [/Based onwhat/g, 'Based on what'],
    [/inthe\b/gi, 'in the'],
    [/onthe\b/gi, 'on the'],
    [/forthe\b/gi, 'for the'],
    [/tothe\b/gi, 'to the'],
    [/withthe\b/gi, 'with the'],
    [/fromthe\b/gi, 'from the'],
    [/andthe\b/gi, 'and the'],
    [/ofthe\b/gi, 'of the'],
    [/atthe\b/gi, 'at the'],
    [/datapulled/gi, 'data pulled'],
    [/diagnosticdata/gi, 'diagnostic data'],
    [/(\d{1,3})(references)\b/gi, '$1 $2'],
    [/(\d{1,3})(mentions)\b/gi, '$1 $2'],
    // Single-letter stream splits (token boundary glitches)
    [/\bs ockets\b/gi, 'sockets'],
    [/\bs ocket\b/gi, 'socket'],
  ]

  let t = text
  for (const [re, rep] of rules) {
    t = t.replace(re, rep)
  }

  t = t.replace(/\bpage(\d{1,3})\b/gi, 'page $1')

  t = t.replace(/[ \t]{2,}/g, ' ')
  return t
}
