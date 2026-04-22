import { useState, useEffect } from 'react'
import { useLesson } from '@/hooks/useLesson'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import Select from '@/components/ui/Select'
import Button from '@/components/ui/Button'
import Loading from '@/components/ui/Loading'
import { renderMarkdown } from '@/utils/markdown'
import type { LessonPlanResponse } from '@/types/lesson'

// ── 常量 ──────────────────────────────────────────────────────────────────────

const HISTORY_KEY = 'lesson_plan_history_records'

const STAGE_LABELS: Record<string, string> = {
  queued: '正在建立流式连接...',
  internal_start: '正在分析主题与准备生成流程...',
  started: '正在分析主题与准备生成流程...',
  planner_done: '规划完成，准备检索资料...',
  tools_done: '工具调用完成...',
  retriever_done: '已检索完成，正在撰写正文...',
  writer_done: '正文已完成，正在整理结果...',
  completed: '内容生成完成，正在渲染页面...',
}

const IMAGE_PROMPT_TEMPLATES: Record<string, { style: string; build: (p: { topic: string; notes: string }) => string }> = {
  concept_diagram: {
    style: 'diagram_clean',
    build: ({ topic, notes }) =>
      `为"${topic}"生成一张高中物理中文教学概念示意图。必须围绕该主题的物理概念本身展开，突出核心概念之间的关系，使用清晰箭头、模块或标注，中文标签准确，适合直接插入教案或课件。不要出现动物、植物、风景、人物写真、纯装饰背景或与主题无关的物体。${notes ? `补充要求：${notes}` : ''}`.trim(),
  },
  process_diagram: {
    style: 'diagram_clean',
    build: ({ topic, notes }) =>
      `为"${topic}"生成一张高中物理中文教学流程图。按步骤展示关键物理过程、变化顺序或推导逻辑，使用箭头和模块表达，层次分明，标签简洁。不要出现与主题无关的生物、自然风景、纯抽象纹理或装饰画面。${notes ? `补充要求：${notes}` : ''}`.trim(),
  },
  structure_diagram: {
    style: 'diagram_clean',
    build: ({ topic, notes }) =>
      `为"${topic}"生成一张高中物理中文教学结构图。突出组成部分、层级关系和相互连接，版面清晰，适合学生快速理解整体结构。只呈现和主题相关的物理对象、图示元素与中文标签，不要加入无关素材。${notes ? `补充要求：${notes}` : ''}`.trim(),
  },
  comparison_infographic: {
    style: 'minimal_infographic',
    build: ({ topic, notes }) =>
      `为"${topic}"生成一张高中物理中文教学对比信息图。突出关键差异、对应关系或易错辨析，信息排布整齐，重点醒目，适合课件展示。图中只保留与主题相关的物理概念和中文说明，不要加入装饰性主体或无关图像。${notes ? `补充要求：${notes}` : ''}`.trim(),
  },
  classroom_illustration: {
    style: 'education_illustration',
    build: ({ topic, notes }) =>
      `为"${topic}"生成一张适合高中物理课堂导入的中文教学插画。画面友好、不过度写实，必须围绕该主题的物理场景或实验情境展开，并配有清晰中文标签。不要生成与主题无关的动物、植物、风景或随意插画元素。${notes ? `补充要求：${notes}` : ''}`.trim(),
  },
}

function buildPrompt(type: string, topic: string, notes: string) {
  const tpl = IMAGE_PROMPT_TEMPLATES[type] ?? IMAGE_PROMPT_TEMPLATES.concept_diagram
  let prompt = tpl.build({ topic: topic || '当前教学主题', notes })
  if (/(速度|速率|位移|时间|加速度)/.test(topic)) {
    prompt += ' 优先使用速度-时间、位移-时间、运动轨迹、刻度尺、计时器、小车等与运动学相关的物理图示元素，不要生成生活无关插画。'
  }
  return { prompt, style: tpl.style }
}

// ── 类型 ──────────────────────────────────────────────────────────────────────

interface HistoryRecord {
  topic: string
  notes: string
  templateCategory: string
  templateLabel: string
  subject?: string
  createdAt: string
  lessonContent: string
  planningMode?: string
}

// ── 选项 ──────────────────────────────────────────────────────────────────────

const templateOptions = [
  { value: 'comprehensive', label: '综合模版' },
  { value: 'teaching_design', label: '教学设计' },
  { value: 'guide', label: '导学案模板' },
  { value: 'ppt', label: 'PPT课件' },
]

const modelOptions = [
  { value: '', label: '默认模型 (gemini-2.0-flash)' },
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite' },
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  { value: 'gpt-4o', label: 'GPT-4o' },
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
  { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
  { value: 'coding-glm-5-free', label: 'Coding GLM-5 Free' },
]

const imagePromptTypeOptions = [
  { value: 'concept_diagram', label: '概念示意图' },
  { value: 'process_diagram', label: '流程图' },
  { value: 'structure_diagram', label: '结构图' },
  { value: 'comparison_infographic', label: '对比信息图' },
  { value: 'classroom_illustration', label: '课堂插画' },
]

// ── 结果展示 ──────────────────────────────────────────────────────────────────

function LessonResult({
  result,
  onExport,
}: {
  result: LessonPlanResponse
  onExport: (html: string, title: string) => void
}) {
  const html = renderMarkdown(result.lesson_content ?? '')
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-lg font-bold">生成内容</h2>
        <Button size="sm" disabled={!result.lesson_content} onClick={() => onExport(html, result.topic)}>
          导出 DOCX
        </Button>
      </div>

      <div className="lesson-markdown" dangerouslySetInnerHTML={{ __html: html }} />

      {result.additional_resources && result.additional_resources.length > 0 && (
        <div className="mt-8 pt-6 border-t border-gray-200">
          <h3 className="text-base font-bold mb-4">参考文档</h3>
          <div className="space-y-3">
            {result.additional_resources.map((r, i) => (
              <div key={i} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex justify-between items-start mb-1">
                  <span className="text-sm font-semibold text-gray-900">{r.source}</span>
                  <span className="text-xs text-gray-400">{(r.score * 100).toFixed(1)}%</span>
                </div>
                <p className="text-sm text-gray-600 line-clamp-2">{r.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── 主页面 ────────────────────────────────────────────────────────────────────

export default function LessonPage() {
  const [topic, setTopic] = useState('')
  const [notes, setNotes] = useState('')
  const [templateCategory, setTemplateCategory] = useState('comprehensive')
  const [model, setModel] = useState('')
  const collection = 'embedding_v4_test'

  // AI 生图
  const [aiVisualEnabled, setAiVisualEnabled] = useState(false)
  const [imagePrompt, setImagePrompt] = useState('')
  const [imagePromptType, setImagePromptType] = useState('concept_diagram')
  const [imagePromptStyle, setImagePromptStyle] = useState('diagram_clean')
  const [promptCustomized, setPromptCustomized] = useState(false)

  // 历史（localStorage 持久化）
  const [history, setHistory] = useState<HistoryRecord[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    } catch {
      return []
    }
  })

  // 当前展示的结果（可以是最新生成，也可以是历史详情）
  const [displayResult, setDisplayResult] = useState<LessonPlanResponse | null>(null)

  const { loading, error, result, progressEvent, generateStream, exportDocx } = useLesson()

  // 最新生成完成后更新展示
  useEffect(() => {
    if (result) setDisplayResult(result)
  }, [result])

  // 持久化历史
  const persistHistory = (records: HistoryRecord[]) => {
    setHistory(records)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(records))
  }

  // ── AI 生图 ──

  const applyTemplate = (type: string, force = false) => {
    if (!force && promptCustomized) return
    const { prompt, style } = buildPrompt(type, topic, notes)
    setImagePrompt(prompt)
    setImagePromptStyle(style)
    setPromptCustomized(false)
  }

  const handleAiVisualToggle = (checked: boolean) => {
    setAiVisualEnabled(checked)
    if (checked) applyTemplate(imagePromptType, true)
  }

  const handleImagePromptTypeChange = (type: string) => {
    setImagePromptType(type)
    if (aiVisualEnabled) applyTemplate(type, true)
  }

  // topic/notes 变化时同步更新 prompt（未自定义时）
  useEffect(() => { applyTemplate(imagePromptType) }, [topic, notes]) // eslint-disable-line

  // ── 生成 ──

  const handleGenerate = () => {
    if (!topic.trim()) return
    setDisplayResult(null)

    generateStream(
      {
        topic,
        notes: notes || undefined,
        collection,
        template_category: templateCategory as any,
        model: model || undefined,
        ai_visual_enabled: aiVisualEnabled,
        ai_visual_prompt: aiVisualEnabled ? (imagePrompt.trim() || undefined) : undefined,
        ai_visual_style: aiVisualEnabled ? imagePromptStyle : undefined,
      },
      (res) => {
        // 保存到历史
        const templateLabel = templateOptions.find(t => t.value === templateCategory)?.label ?? ''
        const record: HistoryRecord = {
          topic,
          notes,
          templateCategory,
          templateLabel,
          subject: res.subject ?? '',
          createdAt: new Date().toISOString(),
          lessonContent: res.lesson_content ?? '',
          planningMode: res.planning_mode ?? 'context_first',
        }
        const next = [
          record,
          ...history.filter(h => !(h.topic === record.topic && h.templateCategory === record.templateCategory)),
        ].slice(0, 8)
        persistHistory(next)
      }
    )
  }

  // ── 历史操作 ──

  const handleViewHistory = (record: HistoryRecord) => {
    if (!record.lessonContent) {
      alert('这条历史记录暂无完整教案内容，可点击"继续"重新生成。')
      return
    }
    setTopic(record.topic)
    setNotes(record.notes)
    setTemplateCategory(record.templateCategory)
    setDisplayResult({
      topic: record.topic,
      lesson_content: record.lessonContent,
      additional_resources: [],
      image_resources: [],
      planning_mode: record.planningMode,
    } as LessonPlanResponse)
  }

  const handleRestoreHistory = (record: HistoryRecord) => {
    setTopic(record.topic)
    setNotes(record.notes)
    setTemplateCategory(record.templateCategory)
  }

  const handleDeleteHistory = (index: number) => {
    const next = [...history]
    next.splice(index, 1)
    persistHistory(next)
  }

  // ── 导出 ──

  const handleExport = (html: string, title: string) => {
    exportDocx(html, title)
  }

  // ── 进度文案 ──

  const statusMessage = progressEvent
    ? (() => {
        const base = STAGE_LABELS[progressEvent.stage] ?? `当前阶段：${progressEvent.stage}`
        if (progressEvent.stage === 'planner_done' && progressEvent.search_query_count) {
          return `规划完成，准备检索资料（查询 ${progressEvent.search_query_count} 路）...`
        }
        if (progressEvent.stage === 'retriever_done') {
          return `已检索到 ${progressEvent.relevant_result_count ?? 0} 条相关内容，正在撰写正文...`
        }
        return base
      })()
    : '准备中...'

  // ── 渲染 ──

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── 左侧表单 ── */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-bold mb-4">模板设置</h2>
            <div className="space-y-4">
              <Input
                label="内容主题"
                placeholder="例如：法拉第与电磁感应"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
              />
              <Textarea
                label="备注"
                placeholder="例如：希望更偏实验探究；增加生活案例"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
              />
              <Select label="模板类别" options={templateOptions} value={templateCategory}
                onChange={(e) => setTemplateCategory(e.target.value)} />
              <Select label="LLM模型" options={modelOptions} value={model}
                onChange={(e) => setModel(e.target.value)} />

              {/* AI 生图 */}
              <div className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <input type="checkbox" id="ai-visual" checked={aiVisualEnabled}
                    onChange={(e) => handleAiVisualToggle(e.target.checked)} className="w-4 h-4" />
                  <label htmlFor="ai-visual" className="text-sm font-semibold cursor-pointer">
                    使用 AI 生图补充
                  </label>
                </div>
                {aiVisualEnabled && (
                  <div className="space-y-3 mt-3">
                    <Select label="Prompt 类型" options={imagePromptTypeOptions} value={imagePromptType}
                      onChange={(e) => handleImagePromptTypeChange(e.target.value)} />
                    <Textarea label="生图 Prompt" placeholder="勾选后自动填入，可修改" value={imagePrompt}
                      onChange={(e) => { setImagePrompt(e.target.value); setPromptCustomized(true) }} rows={4} />
                  </div>
                )}
              </div>

              <div className="flex gap-3">
                <Button onClick={handleGenerate} disabled={loading || !topic.trim()} className="flex-1">
                  生成内容
                </Button>
                <Button variant="secondary" onClick={() => {
                  setTopic(''); setNotes(''); setAiVisualEnabled(false)
                  setImagePrompt(''); setImagePromptStyle('diagram_clean'); setPromptCustomized(false)
                  setDisplayResult(null)
                }}>
                  重置
                </Button>
              </div>
            </div>

            {/* 进度 */}
            {loading && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse flex-shrink-0" />
                  <p className="text-sm text-blue-700 font-medium">{statusMessage}</p>
                </div>
              </div>
            )}
            {error && (
              <div className="mt-4 p-3 bg-red-50 rounded-lg border border-red-200">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}
          </div>

          {/* ── 历史记录 ── */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-base font-bold mb-3">最近生成</h2>
            {history.length === 0 ? (
              <p className="text-sm text-gray-400">这里会保留最近生成的主题，方便继续修改或快速回看。</p>
            ) : (
              <div className="divide-y divide-gray-100">
                {/* 表头 */}
                <div className="grid grid-cols-[1fr_auto_auto] gap-2 pb-2 text-xs font-bold text-gray-400 uppercase tracking-wide">
                  <div>主题</div>
                  <div>时间</div>
                  <div className="text-right">操作</div>
                </div>
                {history.map((record, index) => (
                  <div key={index} className="grid grid-cols-[1fr_auto_auto] gap-2 py-3 items-start">
                    <div className="min-w-0">
                      <div className="text-xs text-gray-400">{record.templateLabel}</div>
                      <div className="text-sm font-semibold text-gray-900 truncate">{record.topic}</div>
                      <div className="text-xs text-gray-500 mt-0.5 line-clamp-1">
                        {record.subject ? `${record.subject} · ` : ''}
                        {record.planningMode === 'autonomous' ? '自主规划' : '上下文优先'}
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 whitespace-nowrap pt-1">
                      {new Date(record.createdAt).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div className="flex flex-col gap-1 items-end">
                      <button
                        className="text-xs px-2 py-1 rounded bg-primary text-white hover:bg-primary-hover"
                        onClick={() => handleViewHistory(record)}
                      >详情</button>
                      <button
                        className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-700 hover:bg-gray-200"
                        onClick={() => handleRestoreHistory(record)}
                      >继续</button>
                      <button
                        className="text-xs px-2 py-1 rounded text-gray-400 hover:text-red-500"
                        onClick={() => handleDeleteHistory(index)}
                      >删除</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── 右侧结果 ── */}
        <div className="lg:col-span-2">
          {displayResult
            ? <LessonResult result={displayResult} onExport={handleExport} />
            : (
              <div className="bg-white rounded-lg border border-gray-200 p-12 text-center text-gray-400">
                {loading ? statusMessage : '请填写表单并点击"生成内容"开始'}
              </div>
            )
          }
        </div>
      </div>

      {loading && <Loading message={statusMessage} />}
    </div>
  )
}
