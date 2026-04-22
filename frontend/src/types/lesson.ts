export interface LessonPlanRequest {
  topic: string
  notes?: string
  collection?: string
  template_category?: 'comprehensive' | 'teaching_design' | 'guide' | 'ppt'
  model?: string
  ai_visual_enabled?: boolean
  ai_visual_prompt?: string
  ai_visual_style?: string
}

export interface LessonImageResource {
  image_id: string
  url: string
  source: string
  page?: number
  caption?: string
  source_type?: string
  role?: string
  model?: string
}

export interface LessonPlanResponse {
  topic: string
  lesson_content: string
  additional_resources: Array<{
    source: string
    score: number
    text: string
  }>
  image_resources: LessonImageResource[]
  template_category?: string
}

export interface LessonHistoryItem {
  id: string
  topic: string
  template_category: string
  created_at: string
  summary?: string
}

// SSE event: "progress" with stage field
export type LessonStreamStage =
  | 'queued'
  | 'internal_start'
  | 'started'
  | 'planner_done'
  | 'tools_done'
  | 'retriever_done'
  | 'writer_done'
  | 'completed'

export interface LessonProgressEvent {
  stage: LessonStreamStage
  [key: string]: any
}

export interface LessonErrorEvent {
  code: string
  message: string
  stage?: string
}
