export type MediaType = 'image' | 'video'

export interface SearchResult {
  media_id: string
  media_type: MediaType
  score: number
  path: string
  width?: number
  height?: number
  duration?: number
  start_sec?: number
  end_sec?: number
  thumbnail_sec?: number
  coarse_score?: number
  rerank_score?: number
  thumbnail_url: string
  media_url: string
  clip_url?: string
}

export interface SearchResponse {
  query_id: string
  query: string
  total: number
  results: SearchResult[]
}

export interface IndexStatus {
  state: 'idle' | 'running' | 'complete' | 'failed'
  operation?: 'scan' | 'build'
  current: number
  total: number
  message: string
  error?: string
  library: {
    images: number
    videos: number
    clips: number
    indexed_media: number
  }
  index: {
    size: number
    dimension: number
    backend: string
    model_version: string
  }
  encoder: {
    name: string
    version: string
  }
}

