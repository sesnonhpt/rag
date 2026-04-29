export interface TemplateFileInfo {
  filename: string
  size_bytes: number
  size_display: string
  modified_at: string
  file_type: string
}

export interface TemplateListResponse {
  templates: TemplateFileInfo[]
  total: number
  directory: string
}

export interface TemplateContent {
  template_id: string
  filename: string
  content_html: string
  version_id: string | null
  metadata: Record<string, any>
}

export interface VersionInfo {
  version_id: string
  created_at: string
  change_summary: string | null
}

export interface VersionListResponse {
  versions: VersionInfo[]
  total: number
}

export interface AIModifyResponse {
  modified_text: string
  processing_time_ms: number
}

export interface CourseDraftStreamRequest {
  platform?: string
  teacher_name?: string
  subject?: string
  grade?: string
  topic?: string
  duration_minutes?: number
  notes?: string
  source_url?: string
  source_text?: string
  prefer_physics_pilot?: boolean
  source_file?: File | null
}

export type CourseDraftStreamStage =
  | 'queued'
  | 'parsing_source'
  | 'understanding_course'
  | 'extracting_thoughts'
  | 'building_design'
  | 'teacher_rewrite'

export interface TeachingThought {
  dimension: string
  icon: string
  title: string
  content: string
  key_points?: string[]
}

export interface CourseDraftProgressEvent {
  stage: CourseDraftStreamStage
  title?: string
  detail?: string
  source_kind?: string
}

export interface CourseDraftResult {
  source_kind: string
  source_label: string
  subject: string
  grade?: string | null
  topic: string
  platform?: string | null
  teacher_name?: string | null
  duration_minutes: number
  source_summary: string
  transcript_text: string
  draft_markdown: string
}

// 导学案分析结果
export interface LessonAnalysis {
  topic: string
  subject: string
  grade: string
  difficulty: string
  question_types: {
    choice: number
    fill: number
    application: number
    other: number
  }
  teaching_sections: string[]
  key_points: string[]
  lesson_mainline: string
  design_logic: string[]
  teacher_moves: string[]
  student_difficulties: string[]
  activity_flow: string[]
  edit_priorities: string[]
  reusable_sections: string[]
  skeleton_markdown: string
}

export interface LegacyGuidePilotPackage {
  success: boolean
  analysis: LessonAnalysis
  thoughts: TeachingThought[]
  source_text_preview: string
}

// 思考过程步骤
export interface ThinkingStep {
  id: number
  title: string
  icon: string
  status: 'pending' | 'thinking' | 'completed'
  thoughts: string[]
  summary?: string
}


// ============================================================================
// 导学案深度分析 - 流式输出
// ============================================================================

export interface BasicInfo {
  topic: string
  subject: string
  grade: string
  difficulty: string
  duration_estimate: number
}

export interface StructureAnalysis {
  teaching_sections: string[]
  key_points: string[]
  question_types: {
    choice: number
    fill: number
    application: number
    other: number
  }
  key_questions: string[]
  blackboard_design?: string
  time_allocation: Record<string, number>
}

export interface DesignIntent {
  lesson_mainline: string
  design_logic: string[]
  why_this_intro: string
  why_this_order: string
  why_this_ending: string
}

export interface StudentPerspective {
  difficulties: string[]
  misconceptions: string[]
  engagement_points: string[]
}

export interface ImprovementSuggestions {
  priorities: string[]
  reusable_parts: string[]
  missing_elements: string[]
  subject_specific_tips: string[]
}

export interface LessonDeepAnalysis {
  basic: BasicInfo
  structure: StructureAnalysis
  design_intent: DesignIntent
  student_perspective: StudentPerspective
  improvement: ImprovementSuggestions
  skeleton_markdown: string
}

export type LessonAnalysisStage = 
  | 'basic'
  | 'structure'
  | 'design_intent'
  | 'student_perspective'
  | 'improvement'
  | 'skeleton'

export type LessonAnalysisMode = 'quick' | 'deep' | 'improve'

export interface LessonAnalysisEvent {
  event: 'stage_start' | 'stage_complete' | 'complete' | 'error'
  stage?: LessonAnalysisStage
  data: any
}
