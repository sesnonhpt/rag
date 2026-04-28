import { useState } from 'react'
import type { ThinkingStep } from '@/types/template'

interface ThinkingProcessPanelProps {
  steps: ThinkingStep[]
}

export default function ThinkingProcessPanel({ steps }: ThinkingProcessPanelProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())

  const toggleStep = (stepId: number) => {
    if (steps.find(s => s.id === stepId)?.status === 'completed') {
      setExpandedSteps(prev => {
        const next = new Set(prev)
        if (next.has(stepId)) {
          next.delete(stepId)
        } else {
          next.add(stepId)
        }
        return next
      })
    }
  }

  if (steps.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="text-6xl mb-4">🤖</div>
        <h3 className="text-xl font-bold text-slate-900 mb-2">
          AI 准备就绪
        </h3>
        <p className="text-slate-500">
          上传导学案后，AI 会在这里展示思考过程
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {steps.map((step) => {
        const isExpanded = expandedSteps.has(step.id)
        const canExpand = step.status === 'completed'

        return (
          <div
            key={step.id}
            className={`rounded-3xl border p-5 transition-all ${
              step.status === 'thinking'
                ? 'border-sky-300 bg-sky-50 shadow-lg'
                : step.status === 'completed'
                ? 'border-emerald-300 bg-emerald-50'
                : 'border-slate-200 bg-slate-50 opacity-60'
            }`}
          >
            <div
              className={`flex items-start gap-3 ${canExpand ? 'cursor-pointer' : ''}`}
              onClick={() => canExpand && toggleStep(step.id)}
            >
              <div className="text-3xl flex-shrink-0">{step.icon}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="text-lg font-bold text-slate-900">
                    第 {step.id} 步：{step.title}
                  </h3>
                  {step.status === 'thinking' && (
                    <div className="flex gap-1">
                      <div 
                        className="w-2 h-2 bg-sky-500 rounded-full animate-bounce" 
                        style={{ animationDelay: '0ms' }} 
                      />
                      <div 
                        className="w-2 h-2 bg-sky-500 rounded-full animate-bounce" 
                        style={{ animationDelay: '150ms' }} 
                      />
                      <div 
                        className="w-2 h-2 bg-sky-500 rounded-full animate-bounce" 
                        style={{ animationDelay: '300ms' }} 
                      />
                    </div>
                  )}
                  {step.status === 'completed' && (
                    <span className="text-emerald-600 text-xl">✓</span>
                  )}
                  {canExpand && (
                    <svg
                      className={`h-5 w-5 text-slate-400 transition-transform ml-auto ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  )}
                </div>

                {step.status === 'completed' && !isExpanded && step.summary && (
                  <p className="mt-2 text-sm text-slate-600">{step.summary}</p>
                )}

                {(step.status === 'thinking' || isExpanded) && step.thoughts.length > 0 && (
                  <div className="mt-3 space-y-2">
                    {step.thoughts.map((thought, i) => (
                      <div
                        key={i}
                        className="flex gap-2 text-sm text-slate-700 animate-fade-in"
                      >
                        <span className="text-slate-400 flex-shrink-0">•</span>
                        <span className="flex-1">{thought}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}

      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(-4px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  )
}
