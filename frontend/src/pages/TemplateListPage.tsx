import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Loading from '@/components/ui/Loading'
import { templateApi } from '@/api/template'
import type { TemplateFileInfo } from '@/types/template'

export default function TemplateListPage() {
  const navigate = useNavigate()
  const [templates, setTemplates] = useState<TemplateFileInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    loadTemplates()
  }, [search])

  const loadTemplates = async () => {
    try {
      setLoading(true)
      setError('')
      const data = await templateApi.list(search)
      setTemplates(data.templates)
    } catch (err: any) {
      setError(err.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = (filename: string) => {
    navigate(`/templates/edit/${encodeURIComponent(filename)}`)
  }

  const handleDownload = (filename: string) => {
    window.location.href = `/templates/download/${encodeURIComponent(filename)}`
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
        <h1 className="text-2xl font-bold mb-4">📚 导学案模板库</h1>
        <p className="text-gray-600 mb-4">浏览、编辑和管理导学案模板文件</p>
        
        <div className="flex gap-4">
          <div className="flex-1">
            <Input
              placeholder="🔍 搜索模板文件名..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <Button onClick={loadTemplates}>刷新</Button>
        </div>
      </div>

      {/* Stats */}
      {templates.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
          <div className="flex justify-between items-center">
            <span className="text-gray-600">共找到</span>
            <span className="text-2xl font-bold text-primary">{templates.length}</span>
            <span className="text-gray-600">个模板文件</span>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Template Grid */}
      {templates.length === 0 && !loading ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <div className="text-6xl mb-4 opacity-30">📭</div>
          <p className="text-gray-600">暂无模板文件，请将文件放入 data/templates/ 目录</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {templates.map((template) => (
            <div
              key={template.filename}
              className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-lg transition-shadow"
            >
              <div className="flex items-start gap-4 mb-4">
                <div className="text-4xl">{getFileIcon(template.file_type)}</div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-gray-900 mb-1 truncate">
                    {template.filename}
                  </h3>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <span>📦</span>
                      <span>{template.size_display}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <span>📅</span>
                      <span>{formatDate(template.modified_at)}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <span>📋</span>
                      <span>{template.file_type}</span>
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
}
