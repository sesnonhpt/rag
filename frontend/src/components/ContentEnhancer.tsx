import { useState } from 'react'
import Button from '@/components/ui/Button'

interface ContentEnhancerProps {
  selectedText: string
  instruction: string
  onInstructionChange: (instruction: string) => void
  onEnhance: () => Promise<void>
  isLoading: boolean
  result: string | null
  onApply: () => void
  onCancel: () => void
}

const commonPrompts = [
  { label: '老师口吻', prompt: '把这段文字改成老师上课时更自然、更顺口的表达。' },
  { label: '更易懂', prompt: '把这段内容改得更容易让学生听懂，少一些书面腔。' },
  { label: '补例子', prompt: '围绕这段内容补一个贴近课堂的例子，便于讲解。' },
  { label: '改提问', prompt: '把这段内容改成老师课堂提问的表达方式。' },
  { label: '压缩精简', prompt: '保留原意，把这段话压缩得更短、更清楚。' },
  { label: '活动引导', prompt: '把这段内容改写成适合课堂活动引导的话术。' },
]

export default function ContentEnhancer({
  selectedText,
  instruction,
  onInstructionChange,
  onEnhance,
  isLoading,
  result,
  onApply,
  onCancel,
}: ContentEnhancerProps) {
  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">AI 辅助工具</p>
          <h3 className="mt-2 text-xl font-bold text-slate-900">选中文字后加强内容</h3>
        </div>
      </div>

      {/* 当前选中内容 */}
      <div className="rounded-3xl border border-slate-200 bg-slate-50 px-4 py-4">
        <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">当前选中内容</div>
        <div className="mt-2 max-h-28 overflow-y-auto text-sm leading-6 text-slate-700">
          {selectedText || '先在左侧正文中选中一段内容，这里会自动显示。'}
        </div>
      </div>

      {selectedText && (
        <>
          {/* 输入指令 */}
          <div className="mt-4">
            <label className="block text-sm font-semibold text-slate-700">希望如何加强</label>
            <textarea
              className="mt-2 w-full rounded-2xl border border-slate-300 px-3 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              rows={3}
              placeholder="例如：补充一个例子；改成更易懂的表达；增加课堂提问"
              value={instruction}
              onChange={(e) => onInstructionChange(e.target.value)}
            />
          </div>

          {/* 快速操作 */}
          <div className="mt-4 flex flex-wrap gap-2">
            {commonPrompts.map((item) => (
              <button
                key={item.label}
                onClick={() => onInstructionChange(item.prompt)}
                className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 transition hover:border-sky-300 hover:bg-sky-50 hover:text-sky-700"
              >
                {item.label}
              </button>
            ))}
          </div>

          {/* 生成按钮 */}
          <Button 
            className="mt-5 w-full" 
            disabled={!instruction.trim() || isLoading} 
            onClick={onEnhance}
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></span>
                AI 正在思考...
              </span>
            ) : (
              '生成加强版本'
            )}
          </Button>

          {/* 流式思考过程 */}
          {isLoading && (
            <div className="mt-4 rounded-3xl border border-blue-200 bg-blue-50 px-4 py-4">
              <div className="text-sm font-semibold text-blue-800 mb-3">💭 AI 思考过程</div>
              <div className="space-y-2">
                <div className="flex items-start gap-2 text-sm text-blue-700 animate-pulse">
                  <span>→</span>
                  <span>理解选中内容...</span>
                </div>
                <div className="flex items-start gap-2 text-sm text-blue-700 animate-pulse" style={{ animationDelay: '0.3s' }}>
                  <span>→</span>
                  <span>分析如何加强...</span>
                </div>
                <div className="flex items-start gap-2 text-sm text-blue-700 animate-pulse" style={{ animationDelay: '0.6s' }}>
                  <span>→</span>
                  <span>生成建议...</span>
                </div>
              </div>
            </div>
          )}

          {/* 结果展示 */}
          {result && (
            <div className="mt-5 rounded-3xl border border-emerald-200 bg-emerald-50 px-4 py-4">
              <div className="text-sm font-semibold text-emerald-800">✨ 加强后的版本</div>
              <p className="mt-2 whitespace-pre-wrap text-sm leading-7 text-slate-700">{result}</p>
              <div className="mt-4 flex gap-2">
                <Button size="sm" className="flex-1" onClick={onApply}>应用修改</Button>
                <Button size="sm" variant="secondary" className="flex-1" onClick={onCancel}>暂不采用</Button>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  )
}
