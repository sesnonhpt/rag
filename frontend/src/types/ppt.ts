export interface PPTElement {
  id: string
  type: 'text' | 'shape' | 'image'
  x: number
  y: number
  w: number
  h: number
  text?: string
  font_size?: number
  bold?: boolean
  fill_color?: string
  text_color?: string
  src?: string
}

export interface PPTSlide {
  id: string
  layout: 'cover' | 'standard' | 'two_column' | 'practice' | 'summary'
  title: string
  accent_text?: string
  bullets?: string[]
  paragraphs?: string[]
  notes?: string[]
  image_sources?: string[]
  elements: PPTElement[]
}

export interface PPTDeck {
  id: string
  title: string
  slides: PPTSlide[]
  created_at: string
  updated_at: string
}
