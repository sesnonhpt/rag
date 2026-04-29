import { useState } from 'react'
import type { TeachingThought } from '@/types/template'
import { renderMarkdown } from '@/utils/markdown'

interface TeachingThoughtsTimelineProps {
  thoughts: TeachingThought[]
}

// 把后端维度 key / title 映射成更自然的老师口语
const DISPLAY_NAMES: Record<string, string> = {
  what_to_teach:   '讲什么',
  how_to_open:     '怎么开场',
  design_logic:    '为何这样排',
  hard_points:     '学生卡哪',
  how_to_close:    '怎么收尾',
  // 兼容旧 title
  '教什么':   '讲什么',
  '怎么铺垫': '怎么开场',
  '主线逻辑': '为何这样排',
  '难在哪':   '学生卡哪',
  '怎么收束': '怎么收尾',
}

function displayName(thought: TeachingThought) {
  return DISPLAY_NAMES[thought.dimension] || DISPLAY_NAMES[thought.title] || thought.title
}

export default function TeachingThoughtsTimeline({ thoughts }: TeachingThoughtsTimelineProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0)

  if (thoughts.length === 0) return null

  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900">备课思路</h3>
        <button
          className="text-xs text-slate-400 hover:text-slate-600"
          onClick={() => setExpandedIndex(expandedIndex === null ? 0 : null)}
        >
          {expandedIndex === null ? '展开' : '收起'}
        </button>
      </div>

      {/* Tab 式导航 */}
      <div className="flex items-center gap-2 flex-wrap">
        {thoughts.map((thought, index) => (
          <button
            key={index}
            onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap
              ${expandedIndex === index
                ? 'bg-slate-800 text-white'
                : 'bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700'
              }`}
          >
            {displayName(thought)}
          </button>
        ))}
      </div>

      {/* 展开内容 */}
      {expandedIndex !== null && thoughts[expandedIndex] && (
        <div className="mt-3 rounded-xl bg-slate-50 px-3 py-3">
          {/* 一句话概括 */}
          <p className="text-xs font-medium text-slate-700 leading-5">
            {thoughts[expandedIndex].content}
          </p>

          {/* 要点列表 */}
          {thoughts[expandedIndex].key_points && thoughts[expandedIndex].key_points!.length > 0 && (
            <ul className="mt-2 space-y-1">
              {thoughts[expandedIndex].key_points!.map((point, idx) => (
                <li key={idx} className="flex items-start gap-1.5 text-xs text-slate-500 leading-5">
                  <span className="mt-0.5 h-1 w-1 shrink-0 rounded-full bg-slate-400" />
                  <span dangerouslySetInnerHTML={{ __html: renderMarkdown(point).replace(/^<p>|<\/p>\n?$/g, '') }} />
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  )
}
