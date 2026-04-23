import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Button from '@/components/ui/Button'
import Loading from '@/components/ui/Loading'
import { templateApi } from '@/api/template'
import type { TemplateContent } from '@/types/template'

// Quill editor will be loaded dynamically
declare global {
  interface Window {
    Quill: any
  }
}

const DOWNLOAD_BASE_URL = import.meta.env.DEV
  ? 'http://localhost:8000'
  : (import.meta.env.VITE_API_BASE_URL || '')

export default function TemplateEditorPage() {
  const { filename } = useParams<{ filename: string }>()
  const navigate = useNavigate()
  
  const [loading, setLoading] = useState(true)
  const [content, setContent] = useState<TemplateContent | null>(null)
  const [aiInstruction, setAiInstruction] = useState('')
  const [aiLoading, setAiLoading] = useState(false)
  const [selectedText, setSelectedText] = useState('')
  const [selectionRange, setSelectionRange] = useState<any>(null)
  const [aiResult, setAiResult] = useState('')
  const [error, setError] = useState('')
  const [showPopover, setShowPopover] = useState(false)
  const [popoverPosition, setPopoverPosition] = useState({ top: 0, left: 0 })
  const [showContextMenu, setShowContextMenu] = useState(false)
  const [contextMenuPosition, setContextMenuPosition] = useState({ x: 0, y: 0 })
  const [imagePrompt, setImagePrompt] = useState('')
  const [imageLoading, setImageLoading] = useState(false)
  const [showImageDialog, setShowImageDialog] = useState(false)
  
  const editorRef = useRef<HTMLDivElement>(null)
  const quillRef = useRef<any>(null)
  const [quillLoaded, setQuillLoaded] = useState(false)
  const popoverRef = useRef<HTMLDivElement>(null)
  const contextMenuRef = useRef<HTMLDivElement>(null)

  // Common prompt templates
  const commonPrompts = [
    { label: '简化语言', prompt: '使语言更简洁明了，保持原意' },
    { label: '增加例子', prompt: '增加具体的例子来说明这个概念' },
    { label: '改为疑问句', prompt: '将这段文字改写为疑问句形式' },
    { label: '扩展内容', prompt: '扩展这段内容，增加更多细节和解释' },
    { label: '改为口语化', prompt: '将这段文字改为更口语化、易懂的表达' },
    { label: '纠正语法', prompt: '检查并纠正语法错误，优化表达' },
  ]

  // Load Quill.js
  useEffect(() => {
    if (window.Quill) {
      setQuillLoaded(true)
      return
    }

    const link = document.createElement('link')
    link.href = 'https://cdn.quilljs.com/1.3.7/quill.snow.css'
    link.rel = 'stylesheet'
    document.head.appendChild(link)

    const script = document.createElement('script')
    script.src = 'https://cdn.quilljs.com/1.3.7/quill.min.js'
    script.onload = () => {
      console.log('Quill.js loaded')
      setQuillLoaded(true)
    }
    document.head.appendChild(script)
  }, [])

  // Initialize editor
  useEffect(() => {
    if (!editorRef.current || !quillLoaded || !window.Quill || quillRef.current) return

    console.log('Initializing Quill editor')
    const quill = new window.Quill(editorRef.current, {
      theme: 'snow',
      modules: {
        toolbar: [
          [{ header: [1, 2, 3, false] }],
          ['bold', 'italic', 'underline', 'strike'],
          [{ list: 'ordered' }, { list: 'bullet' }],
          [{ indent: '-1' }, { indent: '+1' }],
          ['link', 'image'],
          ['clean'],
        ],
      },
      placeholder: '在此编辑模板内容...',
    })

    quillRef.current = quill

    // Load content if available
    if (content && content.content_html) {
      console.log('Setting initial content, length:', content.content_html.length)
      quill.root.innerHTML = content.content_html
    }

    // Handle text selection for AI modification
    quill.on('selection-change', (range: any) => {
      if (range && range.length > 0) {
        const text = quill.getText(range.index, range.length)
        setSelectedText(text.trim())
        setSelectionRange(range)
        
        // Don't auto-show popover, wait for user to right-click
        // Calculate popover position for later use
        const bounds = quill.getBounds(range.index, range.length)
        const editorRect = editorRef.current?.getBoundingClientRect()
        
        if (editorRect) {
          setPopoverPosition({
            top: bounds.top + bounds.height + 10,
            left: bounds.left
          })
        }
      } else {
        // Don't clear selection when typing in popover
        if (!popoverRef.current?.contains(document.activeElement)) {
          // Only clear if not interacting with popover
          setSelectedText('')
          setSelectionRange(null)
        }
      }
    })
  }, [quillLoaded, content])

  // Load template content
  useEffect(() => {
    if (!filename) return
    loadContent()
  }, [filename])

  // Click outside to close popover
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        showPopover &&
        popoverRef.current &&
        !popoverRef.current.contains(event.target as Node) &&
        editorRef.current &&
        !editorRef.current.contains(event.target as Node)
      ) {
        handleClosePopover()
      }
      
      // Close context menu when clicking outside
      if (
        showContextMenu &&
        contextMenuRef.current &&
        !contextMenuRef.current.contains(event.target as Node)
      ) {
        setShowContextMenu(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showPopover, showContextMenu])

  // Handle right-click context menu
  useEffect(() => {
    const handleContextMenu = (e: MouseEvent) => {
      // Only handle right-click inside the editor
      if (editorRef.current && editorRef.current.contains(e.target as Node)) {
        e.preventDefault()
        setContextMenuPosition({ x: e.clientX, y: e.clientY })
        setShowContextMenu(true)
      }
    }

    document.addEventListener('contextmenu', handleContextMenu)
    return () => {
      document.removeEventListener('contextmenu', handleContextMenu)
    }
  }, [])

  const loadContent = async () => {
    if (!filename) return
    
    try {
      setLoading(true)
      setError('')
      const data = await templateApi.getContent(filename)
      console.log('Content loaded:', data.filename, 'HTML length:', data.content_html?.length)
      setContent(data)
      
      // Set content in editor if it's already initialized
      if (quillRef.current && data.content_html) {
        console.log('Setting content in existing editor')
        quillRef.current.root.innerHTML = data.content_html
      }
    } catch (err: any) {
      console.error('Failed to load content:', err)
      setError(err.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  const handleAIModify = async () => {
    if (!selectedText || !aiInstruction.trim()) {
      alert('请选择文本并输入修改指令')
      return
    }
    
    try {
      setAiLoading(true)
      setAiResult('')
      const result = await templateApi.aiModify(selectedText, aiInstruction)
      
      // Show result in the sidebar
      setAiResult(result.modified_text)
    } catch (err: any) {
      alert(`AI 修改失败: ${err.message}`)
    } finally {
      setAiLoading(false)
    }
  }

  const handleApplyAI = () => {
    if (!aiResult || !quillRef.current || !selectionRange) return
    
    // Apply the AI modification
    quillRef.current.deleteText(selectionRange.index, selectionRange.length)
    quillRef.current.insertText(selectionRange.index, aiResult)
    
    // Clear AI state
    setAiResult('')
    setAiInstruction('')
    setSelectedText('')
    setSelectionRange(null)
  }

  const handleCancelAI = () => {
    setAiResult('')
    setAiInstruction('')
    setShowPopover(false)
  }

  const handleClosePopover = () => {
    setShowPopover(false)
    setSelectedText('')
    setSelectionRange(null)
    setAiInstruction('')
    setAiResult('')
  }

  const handleSelectPrompt = (prompt: string) => {
    setAiInstruction(prompt)
  }

  const handleOpenTextEnhance = () => {
    setShowContextMenu(false)
    if (selectedText) {
      // Show the AI editing popover
      setShowPopover(true)
    } else {
      alert('请先选择要增强的文本')
    }
  }

  const handleOpenImageGeneration = () => {
    setShowContextMenu(false)
    setImagePrompt(selectedText || '')
    setShowImageDialog(true)
  }

  const handleCloseImageDialog = () => {
    setShowImageDialog(false)
    setImagePrompt('')
  }

  const generateImage = async (prompt: string) => {
    if (!quillRef.current || !prompt.trim()) return
    
    try {
      setImageLoading(true)
      
      // TODO: Replace with your actual image generation API
      // Example API call:
      // const response = await axios.post('/api/generate-image', { prompt })
      // const imageUrl = response.data.image_url
      
      // For demonstration, we'll use a placeholder
      await new Promise(resolve => setTimeout(resolve, 2000)) // Simulate API call
      
      // Placeholder image URL (replace with actual generated image)
      const imageUrl = `https://via.placeholder.com/400x300?text=${encodeURIComponent(prompt)}`
      
      // Insert image into editor at current cursor position
      const range = quillRef.current.getSelection() || { index: 0 }
      quillRef.current.insertEmbed(range.index, 'image', imageUrl)
      quillRef.current.setSelection(range.index + 1)
      
      setShowImageDialog(false)
      setImagePrompt('')
      
    } catch (err: any) {
      alert(`图片生成失败: ${err.message}`)
    } finally {
      setImageLoading(false)
    }
  }

  const handleExport = async (format: 'docx' | 'pdf' | 'md') => {
    if (!filename || !quillRef.current) return
    
    try {
      setError('')
      
      // Auto-save content before export
      const html = quillRef.current.root.innerHTML
      await templateApi.saveContent(filename, html, false)
      
      // Export
      const result = await templateApi.exportTemplate(filename, format)
      
      // Trigger download
      const downloadUrl = `${DOWNLOAD_BASE_URL}${result.download_url}`
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = ''
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      alert(`导出成功！文件大小: ${(result.file_size / 1024).toFixed(1)} KB`)
    } catch (err: any) {
      alert(`导出失败: ${err.message}`)
    }
  }

  if (loading) {
    return <Loading message="正在加载模板..." />
  }

  if (error && !content) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <p className="text-red-700 mb-4">{error}</p>
          <Button onClick={() => navigate('/templates')}>返回列表</Button>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold mb-2">{content?.filename}</h1>
            <p className="text-sm text-gray-500">
              模板编辑器 - 编辑后直接导出
            </p>
          </div>
          <div className="flex gap-3">
            <Button variant="secondary" onClick={() => navigate('/templates')}>
              返回列表
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="lg:col-span-1 space-y-4">
          {/* Export */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-bold mb-4">导出</h2>
            <p className="text-sm text-gray-600 mb-4">编辑完成后，选择格式导出</p>
            <div className="space-y-2">
              <Button size="sm" variant="secondary" className="w-full" onClick={() => handleExport('docx')}>
                导出 DOCX
              </Button>
              <Button size="sm" variant="secondary" className="w-full" onClick={() => handleExport('pdf')}>
                导出 PDF
              </Button>
              <Button size="sm" variant="secondary" className="w-full" onClick={() => handleExport('md')}>
                导出 Markdown
              </Button>
            </div>
          </div>
        </div>

        {/* Editor */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg border border-gray-200 p-6 relative">
            <div ref={editorRef} style={{ minHeight: '600px' }} />
            
            {/* AI Edit Popover */}
            {showPopover && selectedText && (
              <div
                ref={popoverRef}
                className="absolute z-50 bg-white rounded-lg shadow-xl border border-gray-300 p-4 w-96"
                style={{
                  top: `${popoverPosition.top}px`,
                  left: `${popoverPosition.left}px`,
                  maxWidth: 'calc(100% - 40px)'
                }}
              >
                {/* Close button */}
                <button
                  onClick={handleClosePopover}
                  className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>

                <h3 className="text-sm font-bold mb-3 text-gray-800">AI 编辑</h3>
                
                {/* Selected Text Display */}
                <div className="mb-3">
                  <label className="block text-xs font-medium mb-1 text-gray-600">已选择文本</label>
                  <div className="p-2 bg-blue-50 border border-blue-200 rounded text-xs max-h-20 overflow-y-auto">
                    {selectedText}
                  </div>
                </div>
                
                {/* Common Prompt Tags */}
                <div className="mb-3">
                  <label className="block text-xs font-medium mb-2 text-gray-600">快捷指令</label>
                  <div className="flex flex-wrap gap-2">
                    {commonPrompts.map((item, index) => (
                      <button
                        key={index}
                        onClick={() => handleSelectPrompt(item.prompt)}
                        className="px-2 py-1 text-xs bg-gray-100 hover:bg-blue-100 border border-gray-300 hover:border-blue-400 rounded transition-colors"
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                </div>
                
                {/* AI Instruction Input */}
                <div className="mb-3">
                  <label className="block text-xs font-medium mb-1 text-gray-600">修改指令</label>
                  <textarea
                    className="w-full px-2 py-2 border border-gray-300 rounded text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows={3}
                    placeholder="例如：使语言更简洁、增加例子、改为疑问句..."
                    value={aiInstruction}
                    onChange={(e) => setAiInstruction(e.target.value)}
                    autoFocus
                  />
                </div>

                {/* AI Generate Button */}
                <Button
                  size="sm"
                  className="w-full mb-3"
                  disabled={!aiInstruction.trim() || aiLoading}
                  onClick={handleAIModify}
                >
                  {aiLoading ? 'AI 处理中...' : '生成修改'}
                </Button>

                {/* AI Result Display */}
                {aiResult && (
                  <div className="mb-3">
                    <label className="block text-xs font-medium mb-1 text-gray-600">AI 修改结果</label>
                    <div className="p-2 bg-green-50 border border-green-200 rounded text-xs max-h-32 overflow-y-auto mb-2">
                      {aiResult}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        className="flex-1"
                        onClick={handleApplyAI}
                      >
                        应用修改
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        className="flex-1"
                        onClick={handleCancelAI}
                      >
                        取消
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* Context Menu (Right-click) */}
            {showContextMenu && (
              <div
                ref={contextMenuRef}
                className="fixed z-50 bg-white rounded-lg shadow-xl border border-gray-300 py-2 min-w-[200px]"
                style={{
                  top: `${contextMenuPosition.y}px`,
                  left: `${contextMenuPosition.x}px`,
                }}
              >
                <button
                  onClick={handleOpenTextEnhance}
                  className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 flex items-center gap-2"
                >
                  <span className="text-lg">✨</span>
                  <span>AI 文本编辑</span>
                </button>
                <button
                  onClick={handleOpenImageGeneration}
                  className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 flex items-center gap-2"
                >
                  <span className="text-lg">🎨</span>
                  <span>AI 生图</span>
                </button>
              </div>
            )}
            
            {/* Image Generation Dialog */}
            {showImageDialog && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
                <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md">
                  <h3 className="text-lg font-bold mb-4">AI 生图</h3>
                  
                  <div className="mb-4">
                    <label className="block text-sm font-medium mb-2 text-gray-700">
                      图片描述
                    </label>
                    <textarea
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      rows={4}
                      placeholder="例如：一个可爱的卡通猫咪在看书..."
                      value={imagePrompt}
                      onChange={(e) => setImagePrompt(e.target.value)}
                      autoFocus
                    />
                  </div>
                  
                  <div className="flex gap-3">
                    <Button
                      className="flex-1"
                      onClick={() => generateImage(imagePrompt)}
                      disabled={!imagePrompt.trim() || imageLoading}
                    >
                      {imageLoading ? '生成中...' : '生成图片'}
                    </Button>
                    <Button
                      variant="secondary"
                      className="flex-1"
                      onClick={handleCloseImageDialog}
                      disabled={imageLoading}
                    >
                      取消
                    </Button>
                  </div>
                  
                  {imageLoading && (
                    <div className="mt-4 text-center text-sm text-gray-600">
                      <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mb-2"></div>
                      <p>正在生成图片，请稍候...</p>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            <style>{`
              /* Quill editor custom styles for better Word document rendering */
              .ql-editor {
                font-family: "SimSun", "宋体", serif;
                font-size: 14px;
                line-height: 1.8;
                padding: 40px 60px;
                background: white;
              }
              
              .ql-editor h1 {
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
                font-family: "SimHei", "黑体", sans-serif;
              }
              
              .ql-editor h2 {
                font-size: 18px;
                font-weight: bold;
                margin: 16px 0 8px 0;
                font-family: "SimHei", "黑体", sans-serif;
              }
              
              .ql-editor h3 {
                font-size: 16px;
                font-weight: bold;
                margin: 12px 0 6px 0;
              }
              
              .ql-editor p {
                margin: 8px 0;
                text-indent: 2em;
              }
              
              .ql-editor strong {
                font-weight: bold;
              }
              
              .ql-editor table {
                border-collapse: collapse;
                width: 100%;
                margin: 16px 0;
              }
              
              .ql-editor table td,
              .ql-editor table th {
                border: 1px solid #000;
                padding: 8px 12px;
                text-align: left;
              }
              
              .ql-editor table th {
                background-color: #f5f5f5;
                font-weight: bold;
              }
              
              .ql-editor ul,
              .ql-editor ol {
                padding-left: 2em;
                margin: 8px 0;
              }
              
              .ql-editor li {
                margin: 4px 0;
              }
              
              /* Preserve spacing and indentation */
              .ql-editor .ql-indent-1 {
                padding-left: 3em;
              }
              
              .ql-editor .ql-indent-2 {
                padding-left: 6em;
              }
              
              .ql-editor .ql-indent-3 {
                padding-left: 9em;
              }
              
              /* Better table rendering */
              .ql-editor table {
                font-size: 13px;
              }
              
              /* Preserve underlines and formatting */
              .ql-editor u {
                text-decoration: underline;
              }
              
              .ql-editor s {
                text-decoration: line-through;
              }
              
              /* Image styling */
              .ql-editor img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 16px auto;
                border-radius: 4px;
              }
            `}</style>
          </div>
        </div>
      </div>

      {error && (
        <div className="fixed bottom-4 right-4 bg-red-50 border border-red-200 rounded-lg p-4 max-w-md">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}
    </div>
  )
}
