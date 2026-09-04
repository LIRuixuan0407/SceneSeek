import type { IndexStatus, SearchResponse } from './types'

const apiBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, init)
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail || `请求失败 (${response.status})`)
  }
  return response.json() as Promise<T>
}

function absolutize(response: SearchResponse): SearchResponse {
  response.results = response.results.map((result) => ({
    ...result,
    thumbnail_url: `${apiBase}${result.thumbnail_url}`,
    media_url: `${apiBase}${result.media_url}`,
    clip_url: result.clip_url ? `${apiBase}${result.clip_url}` : undefined,
  }))
  return response
}

export const api = {
  status: () => request<IndexStatus>('/api/index/status'),
  scan: (path: string) =>
    request('/api/index/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }),
  build: (rebuild = false) =>
    request('/api/index/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rebuild }),
    }),
  searchText: async (query: string, mediaType?: string) =>
    absolutize(
      await request<SearchResponse>('/api/search/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit: 30, media_type: mediaType || null }),
      }),
    ),
  searchImage: async (file: File, mediaType?: string) => {
    const body = new FormData()
    body.append('file', file)
    body.append('limit', '30')
    if (mediaType) body.append('media_type', mediaType)
    return absolutize(
      await request<SearchResponse>('/api/search/image', { method: 'POST', body }),
    )
  },
  feedback: (queryId: string, mediaId: string, relevance: number) =>
    request('/api/feedback/relevance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id: queryId, media_id: mediaId, relevance }),
    }),
}

