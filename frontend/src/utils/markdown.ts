import { marked } from 'marked'
import hljs from 'highlight.js'

// marked v12: renderer functions receive plain string arguments
marked.use({
  renderer: {
    image(href: string, _title: string | null, text: string) {
      return `<img src="${href ?? ''}" alt="${text ?? ''}" class="lesson-image" />`
    },
    // 配图标题：**配图N：xxx** 单独成段 → 居中
    paragraph(text: string) {
      if (/^<strong>配图\d+[：:]/.test(text.trim())) {
        return `<p class="lesson-image-title">${text}</p>\n`
      }
      return `<p>${text}</p>\n`
    },
    // 配图来源：> 来源：xxx → 居中
    blockquote(quote: string) {
      if (quote.includes('来源：') || quote.includes('来源:')) {
        return `<blockquote class="lesson-image-caption">${quote}</blockquote>\n`
      }
      return `<blockquote>${quote}</blockquote>\n`
    },
  },
  breaks: true,
  gfm: true,
})

marked.use({
  highlight: (code: string, lang: string) => {
    if (lang && hljs.getLanguage(lang)) {
      try { return hljs.highlight(code, { language: lang }).value } catch { /* ignore */ }
    }
    return code
  },
} as any)

export function renderMarkdown(content: string): string {
  return marked(content) as string
}
