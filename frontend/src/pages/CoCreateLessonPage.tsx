import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import Button from '@/components/ui/Button'
import Loading from '@/components/ui/Loading'

interface TeachingThought {
  dimension: string
  icon: string
  title: string
  content: string
  suggestions: string[]
}

export default function CoCreateLessonPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  
  const [topic, setTopic] = useState<string>('')
  const [subject, setSubject] = useState<string>('')
  const [filename, setFilename] = useState<string>('')
  const [contentHtml, setContentHtml] = useState<string>('')
  const [thoughts, setThoughts] = useState<TeachingThought[]>([])
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    // 从 URL 参数获取文件信息
    const fileParam = searchParams.get('file')
    const topicParam = searchParams.get('topic')
    const subjectParam = searchParams.get('subject')

    if (!fileParam) {
      // 如果没有文件参数，返回模板列表
      navigate('/templates')
      return
    }

    setFilename(fileParam)
    setTopic(topicParam || extractTopicFromFilename(fileParam))
    setSubject(subjectParam || '未知学科')

    // 加载内容和备课思路
    loadContentAndThoughts(fileParam, topicParam || extractTopicFromFilename(fileParam), subjectParam || '未知学科')
  }, [searchParams, navigate])

  // 从文件名提取主题
  const extractTopicFromFilename = (filename: string): string => {
    // 移除扩展名和路径
    const name = filename.split('/').pop()?.replace(/\.(docx?|pdf|txt)$/i, '') || ''
    // 移除常见的前缀（如：1.1、第一章等）
    return name.replace(/^[\d.]+\s*/, '').replace(/^第[一二三四五六七八九十]+[章节课]\s*/, '')
  }

  // 加载内容和备课思路
  const loadContentAndThoughts = async (filename: string, topic: string, subject: string) => {
    setIsLoading(true)
    setThoughts([])

    try {
      // 1. 读取文件内容
      const contentResponse = await fetch(`/api/templates/${encodeURIComponent(filename)}/content`)
      if (!contentResponse.ok) {
        throw new Error('无法读取文件内容')
      }
      const contentData = await contentResponse.json()
      setContentHtml(contentData.content_html)

      // 2. 将 HTML 转换为纯文本用于分析
      const textContent = htmlToText(contentData.content_html)

      // 3. 调用后端 API 分析并获取备课思路
      const formData = new FormData()
      formData.append('content', textContent)

      const response = await fetch('/api/templates/co-create-analyze', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('分析失败')
      }

      const data = await response.json()
      
      // 模拟备课思路（实际应该从后端获取）
      // TODO: 后端需要返回 teaching_thoughts
      const mockThoughts: TeachingThought[] = [
        {
          dimension: 'what_to_teach',
          icon: '📚',
          title: '这节课要讲什么',
          content: `${topic}的核心概念、基本原理和应用方法。重点是让学生理解概念的本质，而不是死记硬背定义。`,
          suggestions: [
            '从学生已有的知识经验出发',
            '用生活中的例子引入概念',
            '强调概念之间的联系'
          ]
        },
        {
          dimension: 'how_to_introduce',
          icon: '🎯',
          title: '怎么导入',
          content: '通过生活情境或实验现象引入，激发学生的好奇心和探究欲望。',
          suggestions: [
            '设计一个贴近学生生活的情境',
            '提出一个引发思考的问题',
            '展示一个有趣的实验或现象'
          ]
        },
        {
          dimension: 'common_mistakes',
          icon: '⚠️',
          title: '学生易错点',
          content: '学生容易混淆相关概念，或者在应用时忽略前提条件。需要通过对比和辨析来帮助学生理解。',
          suggestions: [
            '提前预设易错题型',
            '设计对比练习',
            '引导学生自己发现错误'
          ]
        },
        {
          dimension: 'classroom_activities',
          icon: '🎨',
          title: '课堂活动',
          content: '设计动手实验、小组讨论或问题探究活动，让学生在做中学、在讨论中学。',
          suggestions: [
            '实验观察：让学生亲手操作',
            '小组讨论：分享不同的理解',
            '问题探究：引导学生主动思考'
          ]
        },
        {
          dimension: 'check_understanding',
          icon: '✅',
          title: '检验理解',
          content: '通过课堂提问、练习题和小测验来检验学生是否真正理解了核心概念。',
          suggestions: [
            '设计分层练习题',
            '用提问检查理解深度',
            '观察学生的操作过程'
          ]
        }
      ]

      setThoughts(mockThoughts)

    } catch (error: any) {
      console.error('加载失败:', error)
      alert(`加载失败: ${error.message}`)
      navigate('/templates')
    } finally {
      setIsLoading(false)
    }
  }

  // HTML 转纯文本
  const htmlToText = (html: string): string => {
    const div = document.createElement('div')
    div.innerHTML = html
    return div.textContent || div.innerText || ''
  }

  // 开始编辑导学案
  const handleStartEdit = () => {
    navigate(`/templates/edit/${encodeURIComponent(filename)}`)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      {/* 顶部导航 */}
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-[1600px] px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/templates')}
                className="text-slate-600 hover:text-slate-900"
              >
                ← 返回
              </button>
              <div className="h-6 w-px bg-slate-300"></div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">{topic || '导学案'}</h1>
                <p className="text-sm text-slate-500">{subject}</p>
              </div>
            </div>
            <Button
              onClick={handleStartEdit}
              className="bg-blue-600 hover:bg-blue-700"
            >
              编辑导学案
            </Button>
          </div>
        </div>
      </div>

      {/* 主内容区 - 左右布局 */}
      <div className="mx-auto max-w-[1600px] px-6 py-6">
        <div className="grid grid-cols-[1fr_400px] gap-6 h-[calc(100vh-140px)]">
          {/* 左侧：导学案内容预览 */}
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 bg-slate-50 px-6 py-4">
              <h2 className="font-semibold text-slate-900">📄 导学案内容</h2>
            </div>
            <div className="h-[calc(100%-60px)] overflow-y-auto p-6">
              {isLoading ? (
                <div className="flex h-full items-center justify-center">
                  <div className="text-center">
                    <div className="text-6xl mb-4">📝</div>
                    <p className="text-slate-500">加载中...</p>
                  </div>
                </div>
              ) : contentHtml ? (
                <div
                  className="prose prose-slate max-w-none"
                  dangerouslySetInnerHTML={{ __html: contentHtml }}
                />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <div className="text-center">
                    <div className="text-6xl mb-4">📭</div>
                    <p className="text-slate-500">暂无内容</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 右侧：备课思路 */}
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 bg-gradient-to-r from-amber-50 to-orange-50 px-6 py-4">
              <h2 className="font-semibold text-slate-900">💡 备课思路</h2>
              <p className="mt-1 text-xs text-slate-600">5 个核心维度</p>
            </div>
            <div className="h-[calc(100%-76px)] overflow-y-auto p-4">
              {isLoading ? (
                <div className="flex h-full items-center justify-center">
                  <div className="text-center">
                    <div className="text-6xl mb-4">🤔</div>
                    <p className="text-slate-500">整理中...</p>
                  </div>
                </div>
              ) : thoughts.length > 0 ? (
                <div className="space-y-3">
                  {thoughts.map((thought, index) => (
                    <div
                      key={index}
                      className="rounded-xl border border-slate-200 bg-gradient-to-br from-white to-slate-50 p-4 transition-all hover:shadow-md"
                    >
                      <div className="mb-2 flex items-center gap-2">
                        <span className="text-2xl">{thought.icon}</span>
                        <h3 className="font-semibold text-slate-900 text-sm">
                          {thought.title}
                        </h3>
                      </div>
                      
                      <p className="mb-3 text-xs leading-relaxed text-slate-700">
                        {thought.content}
                      </p>

                      {thought.suggestions.length > 0 && (
                        <div className="space-y-1.5">
                          {thought.suggestions.map((suggestion, idx) => (
                            <div
                              key={idx}
                              className="flex items-start gap-2 text-xs text-slate-600"
                            >
                              <span className="mt-0.5 text-blue-500">•</span>
                              <span>{suggestion}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center">
                  <div className="text-center">
                    <div className="text-6xl mb-4">💭</div>
                    <p className="text-slate-500 text-sm">暂无备课思路</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {isLoading && (
        <Loading message="正在加载..." />
      )}
    </div>
  )
}
