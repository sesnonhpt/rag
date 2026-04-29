import type { LessonAnalysis } from '@/types/template'
import Button from '@/components/ui/Button'

interface LegacyGuideAnalysisPanelProps {
  analysis: LessonAnalysis | null
  loading: boolean
  error?: string
  onApplySkeleton: () => void
}

function renderList(items: string[]) {
  return items.slice(0, 4).map((item, index) => (
    <div key={`${item}-${index}`} className="flex items-start gap-2 text-sm leading-6 text-slate-600">
      <span className="mt-1 text-sky-500">•</span>
      <span>{item}</span>
    </div>
  ))
}

export default function LegacyGuideAnalysisPanel({
  analysis,
  loading,
  error,
  onApplySkeleton,
}: LegacyGuideAnalysisPanelProps) {
  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">旧导学案拆解</p>
      <h3 className="mt-2 text-xl font-bold text-slate-900">先帮老师看懂这份导学案</h3>
      <p className="mt-3 text-sm leading-7 text-slate-500">
        系统会提炼课堂主线、设计逻辑和优先修改点，并生成一版可继续编辑的导学案骨架。
      </p>

      {loading && (
        <div className="mt-5 rounded-3xl border border-sky-200 bg-sky-50 px-4 py-4 text-sm text-sky-800">
          正在拆解旧导学案，整理教学主线和可编辑骨架...
        </div>
      )}

      {error && !loading && (
        <div className="mt-5 rounded-3xl border border-amber-200 bg-amber-50 px-4 py-4 text-sm text-amber-800">
          {error}
        </div>
      )}

      {!loading && analysis && (
        <>
          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <div className="rounded-2xl bg-slate-50 px-4 py-4">
              <div className="text-xs text-slate-400">课题</div>
              <div className="mt-1 text-sm font-semibold text-slate-900">{analysis.topic}</div>
            </div>
            <div className="rounded-2xl bg-slate-50 px-4 py-4">
              <div className="text-xs text-slate-400">学科 / 年级</div>
              <div className="mt-1 text-sm font-semibold text-slate-900">
                {analysis.subject} · {analysis.grade}
              </div>
            </div>
            <div className="rounded-2xl bg-slate-50 px-4 py-4">
              <div className="text-xs text-slate-400">难度判断</div>
              <div className="mt-1 text-sm font-semibold text-slate-900">{analysis.difficulty}</div>
            </div>
            <div className="rounded-2xl bg-slate-50 px-4 py-4">
              <div className="text-xs text-slate-400">建议动作</div>
              <div className="mt-1 text-sm font-semibold text-slate-900">先写入骨架，再局部修改</div>
            </div>
          </div>

          <div className="mt-5 rounded-3xl border border-emerald-200 bg-emerald-50 px-4 py-4">
            <div className="text-sm font-semibold text-emerald-800">课堂主线</div>
            <p className="mt-2 text-sm leading-7 text-slate-700">{analysis.lesson_mainline}</p>
          </div>

          <div className="mt-5 space-y-4">
            <div>
              <div className="text-sm font-semibold text-slate-900">设计逻辑</div>
              <div className="mt-2 space-y-2">{renderList(analysis.design_logic)}</div>
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-900">老师可直接借鉴</div>
              <div className="mt-2 space-y-2">{renderList(analysis.teacher_moves)}</div>
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-900">优先修改点</div>
              <div className="mt-2 space-y-2">{renderList(analysis.edit_priorities)}</div>
            </div>
          </div>

          <Button className="mt-5 w-full" onClick={onApplySkeleton}>
            把可编辑骨架写入正文
          </Button>
        </>
      )}
    </section>
  )
}
