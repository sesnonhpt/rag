import { useState, useRef, useEffect } from 'react'
import Button from './ui/Button'
import { renderMarkdown } from '@/utils/markdown'

interface Props {
  filename: string
  onApplySkeleton?: (markdown: string) => void
  getEditorContent?: () => string
  applyToEditor?: (html: string) => void
}

const STAGES = [
  { key: 'basic',          title: '基础信息', icon: '📊', marker: '## 📊 基础信息' },
  { key: 'structure',      title: '结构分析', icon: '📝', marker: '## 📝 结构分析' },
  { key: 'design_intent',  title: '设计意图', icon: '💡', marker: '## 💡 设计意图' },
  { key: 'improvement',    title: '改进建议', icon: '✨', marker: '## ✨ 改进建议' },
] as const

type StageKey = typeof STAGES[number]['key']

// ── 工具函数 ──────────────────────────────────────────────────────────────────

function splitByStages(fullText: string): Partial<Record<StageKey, string>> {
  const result: Partial<Record<StageKey, string>> = {}
  for (let i = 0; i < STAGES.length; i++) {
    const { key, marker } = STAGES[i]
    const start = fullText.indexOf(marker)
    if (start === -1) continue
    const contentStart = start + marker.length
    let end = fullText.length
    for (let j = i + 1; j < STAGES.length; j++) {
      const next = fullText.indexOf(STAGES[j].marker)
      if (next !== -1) { end = next; break }
    }
    result[key] = fullText.slice(contentStart, end).trim()
  }
  return result
}

function detectActiveStage(fullText: string): StageKey | '' {
  let last: StageKey | '' = ''
  for (const s of STAGES) if (fullText.includes(s.marker)) last = s.key
  return last
}

// ── 打字机 hook ───────────────────────────────────────────────────────────────

function useTypewriter(rawText: string) {
  const [display, setDisplay] = useState('')
  const [caughtUp, setCaughtUp] = useState(false)
  const idxRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (rawText.length <= idxRef.current) return
    if (timerRef.current) return
    timerRef.current = setInterval(() => {
      if (idxRef.current >= rawText.length) {
        clearInterval(timerRef.current!); timerRef.current = null
        setCaughtUp(true)
        return
      }
      const next = Math.min(idxRef.current + 5, rawText.length)
      idxRef.current = next
      setDisplay(rawText.slice(0, next))
      if (next >= rawText.length) {
        clearInterval(timerRef.current!); timerRef.current = null
        setCaughtUp(true)
      }
    }, 16)
  }, [rawText])

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current) }, [])

  return { display, caughtUp }
}

// ── Step 状态类型 ─────────────────────────────────────────────────────────────

type StepStatus = 'waiting' | 'active' | 'done'

// ── 单个 Step 行 ──────────────────────────────────────────────────────────────

function StepRow({
  stage,
  index,
  isLast,
  status,
  rawText,
}: {
  stage: typeof STAGES[number]
  index: number
  isLast: boolean
  status: StepStatus
  rawText: string
}) {
  const { display, caughtUp } = useTypewriter(rawText)
  const useMarkdown = status === 'done' && caughtUp
  const hasContent = rawText.length > 0

  return (
    <div className="flex gap-4">
      {/* 左侧：圆点 + 竖线 */}
      <div className="flex flex-col items-center">
        {/* 圆点 */}
        <div
          className={`relative flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2 text-sm font-bold transition-all duration-300 ${
            status === 'done'
              ? 'border-slate-500 bg-slate-500 text-white'
              : status === 'active'
              ? 'border-sky-400 bg-white text-sky-600'
              : 'border-slate-300 bg-white text-slate-400'
          }`}
        >
          {status === 'done' ? (
            <svg className="h-4 w-4" viewBox="0 0 16 16" fill="none">
              <path d="M3 8l3.5 3.5L13 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : status === 'active' ? (
            <>
              <span className="absolute inset-0 animate-ping rounded-full bg-sky-300 opacity-50" />
              <span className="text-xs">{stage.icon}</span>
            </>
          ) : (
            <span className="text-xs text-slate-400">{index + 1}</span>
          )}
        </div>

        {/* 竖线 */}
        {!isLast && (
          <div
            className={`mt-1 w-0.5 flex-1 min-h-[1.5rem] rounded-full transition-all duration-500 ${
              status === 'done' ? 'bg-slate-400' : 'bg-slate-200'
            }`}
          />
        )}
      </div>

      {/* 右侧：标题 + 内容 */}
      <div className={`pb-6 flex-1 min-w-0 ${isLast ? 'pb-0' : ''}`}>
        {/* 标题行 */}
        <div className="flex items-center gap-2 h-8">
          <span
            className={`text-sm font-semibold transition-colors duration-300 ${
              status === 'done'
                ? 'text-slate-600'
                : status === 'active'
                ? 'text-slate-900'
                : 'text-slate-400'
            }`}
          >
            {status !== 'waiting' && <span className="mr-1">{stage.icon}</span>}
            {stage.title}
          </span>
          {status === 'active' && (
            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[10px] font-medium text-sky-600 animate-pulse">
              分析中
            </span>
          )}
          {/* {status === 'done' && (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500">
              完成
            </span>
          )} */}
        </div>

        {/* 内容区 */}
        {hasContent && (
          <div className="mt-2">
            {useMarkdown ? (
              <div
                className="prose prose-sm max-w-none rounded-xl bg-slate-50 px-4 py-3 [&_*]:!text-xs [&_*]:!leading-5
                  [&_p]:my-0.5 [&_p]:text-slate-700
                  [&_li]:my-0 [&_li]:text-slate-700
                  [&_h1]:font-semibold [&_h1]:text-slate-800
                  [&_h2]:font-semibold [&_h2]:text-slate-800
                  [&_h3]:font-semibold [&_h3]:text-slate-800
                  [&_strong]:text-slate-800"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(rawText) }}
              />
            ) : (
              <div className="rounded-xl bg-slate-50 px-4 py-3 text-xs leading-6 text-slate-700 whitespace-pre-wrap break-words">
                {display}
                <span className="ml-0.5 inline-block h-[1em] w-0.5 translate-y-[2px] animate-blink bg-slate-500" />
              </div>
            )}
          </div>
        )}

        {/* waiting 占位：三个跳动小点 */}
        {status === 'waiting' && (
          <div className="mt-2 flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-300 animate-bounce [animation-delay:0ms]" />
            <span className="h-1.5 w-1.5 rounded-full bg-slate-300 animate-bounce [animation-delay:150ms]" />
            <span className="h-1.5 w-1.5 rounded-full bg-slate-300 animate-bounce [animation-delay:300ms]" />
          </div>
        )}
      </div>
    </div>
  )
}

// ── 解析改进建议条目 ──────────────────────────────────────────────────────────

function parseImprovementItems(text: string): string[] {
  if (!text) return []
  return text
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.match(/^[-•*\d]/))
    .map(l => l.replace(/^[-•*]\s*/, '').replace(/^\d+[.)]\s*/, '').trim())
    .filter(Boolean)
}

// ── 单条改进建议 ──────────────────────────────────────────────────────────────

function SuggestionItem({
  suggestion,
  getEditorContent,
  applyToEditor,
}: {
  suggestion: string
  getEditorContent?: () => string
  applyToEditor?: (html: string) => void
}) {
  const [state, setState] = useState<'idle' | 'loading' | 'done'>('idle')
  const [result, setResult] = useState('')
  const [error, setError] = useState('')

  const handleApply = async () => {
    const editorHtml = getEditorContent?.() || ''
    if (!editorHtml.trim()) {
      setError('编辑器内容为空，请先加载导学案')
      return
    }
    // 剥离 HTML 标签，只传纯文本给 LLM
    const tmp = document.createElement('div')
    tmp.innerHTML = editorHtml
    const plainText = tmp.innerText || tmp.textContent || ''

    setState('loading')
    setError('')
    setResult('')
    try {
      const { templateApi } = await import('@/api/template')
      const res = await templateApi.aiModify(
        plainText,
        `请根据以下改进建议，对整份导学案内容进行针对性修改，只改与建议相关的部分，其余保持不变：\n${suggestion}`
      )
      setResult(res.modified_text)
      setState('done')
    } catch (e: any) {
      setError(e.message || '生成失败')
      setState('idle')
    }
  }

  const [copied, setCopied] = useState(false)

  const handleAccept = async () => {
    if (!result) return
    try {
      await navigator.clipboard.writeText(result)
    } catch {
      const el = document.createElement('textarea')
      el.value = result
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleReject = () => {
    setState('idle')
    setResult('')
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <div
        className="prose prose-xs max-w-none [&_*]:!text-xs [&_*]:!leading-5
          [&_p]:my-0 [&_p]:text-slate-700 [&_strong]:text-slate-800 [&_strong]:font-semibold"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(suggestion) }}
      />

      {state === 'idle' && (
        <button
          onClick={handleApply}
          className="mt-2 rounded-lg bg-sky-50 px-3 py-1 text-[11px] font-medium text-sky-700 hover:bg-sky-100 transition-colors"
        >
          按此建议修改
        </button>
      )}

      {state === 'loading' && (
        <div className="mt-2 flex items-center gap-1.5 text-[11px] text-slate-400">
          <span className="h-3 w-3 animate-spin rounded-full border border-slate-300 border-t-sky-500" />
          AI 正在修改...
        </div>
      )}

      {error && (
        <p className="mt-1 text-[11px] text-red-500">{error}</p>
      )}

      {state === 'done' && result && (
        <div className="mt-3 space-y-2">
          <div
            className="max-h-40 overflow-y-auto rounded-lg border border-amber-200 bg-amber-50 px-3 py-2
              prose prose-xs max-w-none [&_*]:!text-xs [&_*]:!leading-5
              [&_p]:my-0.5 [&_p]:text-slate-700
              [&_li]:my-0 [&_li]:text-slate-700
              [&_strong]:text-slate-800 [&_strong]:font-semibold"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(result) }}
          />
          <div className="flex gap-2">
            <button
              onClick={handleAccept}
              className="flex-1 rounded-lg bg-slate-800 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-slate-700 transition-colors"
            >
              {copied ? '已复制 ✓' : '复制'}
            </button>
            <button
              onClick={handleReject}
              className="flex-1 rounded-lg border border-slate-200 px-3 py-1.5 text-[11px] text-slate-500 hover:bg-slate-50 transition-colors"
            >
              取消
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────────────────

export default function LessonDeepAnalysisPanel({ filename, getEditorContent, applyToEditor }: Props) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [fullText, setFullText] = useState('')
  const [doneStages, setDoneStages] = useState<Set<StageKey>>(new Set())
  const [runId, setRunId] = useState(0)
  const abortRef = useRef<(() => void) | null>(null)

  const handleAnalyze = async () => {
    if (!filename) return
    setLoading(true)
    setError('')
    setFullText('')
    setDoneStages(new Set())
    setRunId(id => id + 1)

    const { templateApi } = await import('@/api/template')
    let currentStageRef: StageKey | '' = ''

    const abort = await templateApi.analyzeLessonStream(
      filename,
      (event) => {
        const { event: eventType, data } = event
        if (eventType === 'stage_start') {
          currentStageRef = data.stage as StageKey
        } else if (eventType === 'content') {
          setFullText(prev => prev + (data.token || ''))
        } else if (eventType === 'stage_complete') {
          if (currentStageRef) setDoneStages(prev => new Set([...prev, currentStageRef as StageKey]))
        } else if (eventType === 'complete') {
          setDoneStages(new Set(STAGES.map(s => s.key)))
        } else if (eventType === 'error') {
          setError(data.message || '分析失败')
        }
      },
      () => setLoading(false),
      (err) => { setError(err); setLoading(false) }
    )
    abortRef.current = abort
  }

  const handleStop = () => {
    abortRef.current?.()
    abortRef.current = null
    setLoading(false)
  }

  const stageContents = splitByStages(fullText)
  const activeStage = loading ? detectActiveStage(fullText) : ''
  const started = fullText.length > 0 || loading

  // 改进建议条目（分析完成后解析）
  const improvementItems = doneStages.has('improvement')
    ? parseImprovementItems(stageContents['improvement'] ?? '')
    : []

  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
      {/* 标题 */}
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">导学案分析</p>

      {/* 按钮 */}
      <div className="mt-5">
        {!loading ? (
          <Button className="w-full" onClick={handleAnalyze} disabled={!filename}>
             开始分析
          </Button>
        ) : (
          <Button className="w-full" variant="secondary" onClick={handleStop}>
            停止
          </Button>
        )}
      </div>

      {/* 错误 */}
      {error && (
        <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Steps */}
      {started && (
        <div className="mt-6">
          {STAGES.map((stage, index) => {
            const isDone = doneStages.has(stage.key)
            const isActive = activeStage === stage.key
            const status: StepStatus = isDone ? 'done' : isActive ? 'active' : 'waiting'
            const content = stageContents[stage.key] ?? ''

            return (
              <StepRow
                key={`${runId}-${stage.key}`}
                stage={stage}
                index={index}
                isLast={index === STAGES.length - 1}
                status={status}
                rawText={content}
              />
            )
          })}
        </div>
      )}

      {/* 改进建议 - 逐条可操作 */}
      {improvementItems.length > 0 && !loading && (
        <div className="mt-6 border-t border-slate-100 pt-5">
          <p className="mb-3 text-xs font-semibold text-slate-500 uppercase tracking-widest">逐条应用改进建议</p>
          <div className="space-y-2">
            {improvementItems.map((item, i) => (
              <SuggestionItem
                key={`${runId}-suggestion-${i}`}
                suggestion={item}
                getEditorContent={getEditorContent}
                applyToEditor={applyToEditor}
              />
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
