/**
 * Floating Action Buttons Component
 * 
 * Fixed buttons on the right edge of the page for Import and Export actions
 */

import { useState } from 'react'

interface FloatingActionButtonsProps {
  onImport: () => void
  onExportEdited: () => void
  onExportOriginal: () => void
}

export default function FloatingActionButtons({
  onImport,
  onExportEdited,
  onExportOriginal,
}: FloatingActionButtonsProps) {
  const [showExportMenu, setShowExportMenu] = useState(false)

  return (
    <div className="fixed right-0 top-[30%] z-50 flex flex-col gap-2">
      {/* Import Button */}
      <button
        onClick={onImport}
        className="group flex items-center gap-2 bg-sky-600 text-white shadow-lg hover:bg-sky-700 transition-all duration-300 rounded-l-lg pr-3 pl-2 py-2.5"
        title="导入文档"
      >
        <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>
        <span className="font-medium text-sm whitespace-nowrap">导入</span>
      </button>

      {/* Export Button with Menu */}
      <div className="relative">
        <button
          onClick={() => setShowExportMenu(!showExportMenu)}
          className="group flex items-center gap-2 bg-slate-600 text-white shadow-lg hover:bg-slate-700 transition-all duration-300 rounded-l-lg pr-3 pl-2 py-2.5"
          title="导出文档"
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <span className="font-medium text-sm whitespace-nowrap">导出</span>
        </button>

        {/* Export Menu */}
        {showExportMenu && (
          <div className="absolute right-full top-0 mr-2 w-52 rounded-lg border border-slate-200 bg-white shadow-xl">
            <button
              onClick={() => {
                onExportEdited()
                setShowExportMenu(false)
              }}
              className="w-full px-3 py-2.5 text-left hover:bg-slate-50 flex items-center gap-2.5 border-b border-slate-100 rounded-t-lg transition-colors"
            >
              <svg className="h-5 w-5 text-sky-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <div>
                <p className="font-medium text-slate-900 text-sm">导出编辑后文档</p>
                <p className="text-xs text-slate-500 mt-0.5">导出当前编辑器内容</p>
              </div>
            </button>
            <button
              onClick={() => {
                onExportOriginal()
                setShowExportMenu(false)
              }}
              className="w-full px-3 py-2.5 text-left hover:bg-slate-50 flex items-center gap-2.5 rounded-b-lg transition-colors"
            >
              <svg className="h-5 w-5 text-slate-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10"
                />
              </svg>
              <div>
                <p className="font-medium text-slate-900 text-sm">下载原始文档</p>
                <p className="text-xs text-slate-500 mt-0.5">下载未编辑的模板</p>
              </div>
            </button>
          </div>
        )}
      </div>

      {/* Backdrop for closing menu */}
      {showExportMenu && (
        <div
          className="fixed inset-0 z-[-1]"
          onClick={() => setShowExportMenu(false)}
        />
      )}
    </div>
  )
}
