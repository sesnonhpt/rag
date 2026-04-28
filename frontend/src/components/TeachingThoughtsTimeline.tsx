import { useState } from 'react'
import type { TeachingThought } from '@/types/template'

interface TeachingThoughtsTimelineProps {
  thoughts: TeachingThought[]
}

export default function TeachingThoughtsTimeline({ thoughts }: TeachingThoughtsTimelineProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)

  if (thoughts.length === 0) return null

  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900">💡 备课思路</h3>
        <button 
          className="text-xs text-blue-600 hover:text-blue-700"
          onClick={() => setExpandedIndex(expandedIndex === null ? 0 : null)}
        >
          {expandedIndex === null ? '展开细节' : '收起'}
        </button>
      </div>
      
      {/* 步骤式流程 */}
      <div className="flex items-center gap-2">
        {thoughts.map((thought, index) => (
          <div key={index} className="flex items-center">
            {/* 步骤圆圈 */}
            <button
              onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
              className={`
                flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full text-lg
                transition-all cursor-pointer
                ${expandedIndex === index 
                  ? 'bg-blue-600 text-white ring-4 ring-blue-100' 
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }
              `}
              title={thought.title}
            >
              {thought.icon}
            </button>
            
            {/* 连接线 */}
            {index < thoughts.length - 1 && (
              <div className="h-0.5 w-8 bg-slate-200"></div>
            )}
          </div>
        ))}
      </div>

      {/* 展开的详情 */}
      {expandedIndex !== null && thoughts[expandedIndex] && (
        <div className="mt-4 rounded-xl border border-blue-200 bg-blue-50 p-4 animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="mb-2 flex items-center gap-2">
            <span className="text-2xl">{thoughts[expandedIndex].icon}</span>
            <h4 className="text-sm font-semibold text-slate-900">
              {thoughts[expandedIndex].title}
            </h4>
          </div>
          <p className="text-xs leading-relaxed text-slate-700 mb-3">
            {thoughts[expandedIndex].content}
          </p>
          {thoughts[expandedIndex].key_points && thoughts[expandedIndex].key_points!.length > 0 && (
            <div className="space-y-1">
              {thoughts[expandedIndex].key_points!.map((point, idx) => (
                <div key={idx} className="flex items-start gap-2 text-xs text-slate-600">
                  <span className="text-blue-500">•</span>
                  <span>{point}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
