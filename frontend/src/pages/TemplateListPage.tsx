import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Loading from '@/components/ui/Loading'
import { templateApi } from '@/api/template'
import type { TemplateFileInfo } from '@/types/template'

interface GroupedTemplates {
  [folder: string]: TemplateFileInfo[]
}

export default function TemplateListPage() {
  const navigate = useNavigate()
  const [templates, setTemplates] = useState<TemplateFileInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [activeSearch, setActiveSearch] = useState('') // 实际执行搜索的关键词
  const [error, setError] = useState('')
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['根目录']))

  useEffect(() => {
    loadTemplates()
  }, [activeSearch])

  const loadTemplates = async () => {
    try {
      setLoading(true)
      setError('')
      const data = await templateApi.list(activeSearch || undefined)
      setTemplates(data.templates || [])
    } catch (err: any) {
      console.error('Failed to load templates:', err)
      setError(err.message || '加载失败')
      setTemplates([]) // Ensure templates is always an array
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = () => {
    setActiveSearch(search)
  }

  const handleClearSearch = () => {
    setSearch('')
    setActiveSearch('')
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  // Group templates by folder
  const groupedTemplates: GroupedTemplates = templates.reduce((acc, template) => {
    const parts = template.filename.split('/')
    const folder = parts.length > 1 ? parts[0] : '根目录'
    
    if (!acc[folder]) {
      acc[folder] = []
    }
    acc[folder].push(template)
    return acc
  }, {} as GroupedTemplates)

  const toggleFolder = (folder: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev)
      if (next.has(folder)) {
        next.delete(folder)
      } else {
        next.add(folder)
      }
      return next
    })
  }

  const getFileName = (fullPath: string) => {
    const parts = fullPath.split('/')
    return parts[parts.length - 1]
  }

  const handleEdit = (filename: string) => {
    // Encode each path segment separately to preserve subdirectory structure
    const encodedPath = filename.split('/').map(segment => encodeURIComponent(segment)).join('/')
    navigate(`/templates/edit/${encodedPath}`)
  }

  const handleDownload = (filename: string) => {
    // Encode each path segment separately to preserve subdirectory structure
    const encodedPath = filename.split('/').map(segment => encodeURIComponent(segment)).join('/')
    window.location.href = `/templates/download/${encodedPath}`
  }

  const getFileIcon = (fileType: string) => {
    if (fileType.includes('Word')) return '📄'
    if (fileType.includes('PDF')) return '📕'
    if (fileType.includes('文本')) return '📝'
    return '📎'
  }

  const formatDate = (timestamp: string) => {
    const date = new Date(parseFloat(timestamp) * 1000)
    return date.toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  if (loading && templates.length === 0) {
    return <Loading message="正在加载模板列表..." />
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <div className="mb-4">
          <h1 className="text-2xl font-bold mb-2">📚 导学案模板库</h1>
          <p className="text-gray-600">浏览、编辑和管理导学案模板文件</p>
        </div>
        
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Input
              placeholder="🔍 搜索模板文件名..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            {search && (
              <button
                onClick={handleClearSearch}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                title="清除搜索"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          <Button onClick={handleSearch} disabled={loading}>
            搜索
          </Button>
          <Button onClick={loadTemplates} disabled={loading} variant="secondary">
            {loading ? '加载中...' : '刷新'}
          </Button>
        </div>
      </div>

      {/* Stats */}
      {templates && templates.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
          <div className="flex justify-between items-center">
            <span className="text-gray-600">
              {activeSearch ? '搜索结果：' : '共找到'}
            </span>
            <span className="text-2xl font-bold text-primary">{templates.length}</span>
            <span className="text-gray-600">个模板文件</span>
          </div>
          {activeSearch && (
            <p className="text-sm text-gray-500 mt-2 text-center">
              搜索关键词：<span className="font-semibold">"{activeSearch}"</span>
            </p>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Template Grid by Folder */}
      {templates.length === 0 && !loading ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <div className="text-6xl mb-4 opacity-30">
            {activeSearch ? '🔍' : '📭'}
          </div>
          <p className="text-gray-600">
            {activeSearch 
              ? `未找到包含 "${activeSearch}" 的模板文件` 
              : '暂无模板文件，请将文件放入 data/templates/ 目录'}
          </p>
          {activeSearch && (
            <Button 
              variant="secondary" 
              className="mt-4"
              onClick={handleClearSearch}
            >
              清除搜索
            </Button>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          {Object.entries(groupedTemplates).sort(([a], [b]) => {
            // 根目录排在最前面
            if (a === '根目录') return -1
            if (b === '根目录') return 1
            return a.localeCompare(b, 'zh-CN')
          }).map(([folder, folderTemplates]) => {
            const isExpanded = expandedFolders.has(folder)
            
            return (
              <div key={folder} className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                {/* Folder Header */}
                <div
                  className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200 cursor-pointer hover:bg-gray-100 transition-colors"
                  onClick={() => toggleFolder(folder)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">
                      {isExpanded ? '📂' : '📁'}
                    </span>
                    <div>
                      <h3 className="font-bold text-gray-900">{folder}</h3>
                      <p className="text-sm text-gray-600">{folderTemplates.length} 个文件</p>
                    </div>
                  </div>
                  <button className="text-gray-500 hover:text-gray-700">
                    <svg
                      className={`w-6 h-6 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                </div>

                {/* Folder Content */}
                {isExpanded && (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
                    {folderTemplates.map((template) => (
                      <div
                        key={template.filename}
                        className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-lg transition-shadow"
                      >
                        <div className="flex items-start gap-3 mb-3">
                          <div className="text-3xl">{getFileIcon(template.file_type)}</div>
                          <div className="flex-1 min-w-0">
                            <h4 className="font-semibold text-gray-900 mb-1 truncate text-sm">
                              {getFileName(template.filename)}
                            </h4>
                            <div className="space-y-1">
                              <div className="flex items-center gap-2 text-xs text-gray-600">
                                <span>📦</span>
                                <span>{template.size_display}</span>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-gray-600">
                                <span>📅</span>
                                <span>{formatDate(template.modified_at)}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            className="flex-1"
                            onClick={() => handleEdit(template.filename)}
                          >
                            编辑
                          </Button>
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => handleDownload(template.filename)}
                          >
                            下载
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
