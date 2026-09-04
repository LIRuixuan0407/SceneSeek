import {
  Aperture,
  ArrowRight,
  Check,
  ChevronRight,
  CircleAlert,
  Clock3,
  Database,
  FolderSearch,
  HardDrive,
  Image as ImageIcon,
  Images,
  LoaderCircle,
  Menu,
  Play,
  Search,
  SlidersHorizontal,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Upload,
  Video,
  X,
} from 'lucide-react'
import { type ChangeEvent, type FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import type { IndexStatus, MediaType, SearchResponse, SearchResult } from './types'

type Filter = 'all' | MediaType
type QueryMode = 'text' | 'image'

const examples = [
  '雨夜里一个人撑黑伞过马路',
  '观众开始鼓掌的镜头',
  '海边落日骑自行车',
]

const emptyStatus: IndexStatus = {
  state: 'idle',
  current: 0,
  total: 0,
  message: '正在连接',
  library: { images: 0, videos: 0, clips: 0, indexed_media: 0 },
  index: { size: 0, dimension: 0, backend: '—', model_version: '—' },
  encoder: { name: '—', version: '—' },
}

function App() {
  const [status, setStatus] = useState<IndexStatus>(emptyStatus)
  const [query, setQuery] = useState('')
  const [queryMode, setQueryMode] = useState<QueryMode>('text')
  const [queryImage, setQueryImage] = useState<File | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [selected, setSelected] = useState<SearchResult | null>(null)
  const [searching, setSearching] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [setupOpen, setSetupOpen] = useState(false)
  const [libraryPath, setLibraryPath] = useState('')
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const refreshStatus = async () => {
    try {
      setStatus(await api.status())
    } catch {
      setError('无法连接 SceneSeek API，请确认后端已启动。')
    }
  }

  useEffect(() => {
    void refreshStatus()
  }, [])

  useEffect(() => {
    if (status.state !== 'running') return
    const timer = window.setInterval(() => void refreshStatus(), 1200)
    return () => window.clearInterval(timer)
  }, [status.state])

  const indexedPercent = status.total ? Math.round((status.current / status.total) * 100) : 0
  const hasLibrary = status.library.images + status.library.videos > 0

  const filteredResults = useMemo(
    () =>
      response?.results.filter((result) => filter === 'all' || result.media_type === filter) || [],
    [filter, response],
  )

  const search = async (event?: FormEvent) => {
    event?.preventDefault()
    if (queryMode === 'text' && !query.trim()) {
      inputRef.current?.focus()
      return
    }
    if (queryMode === 'image' && !queryImage) return
    setSearching(true)
    setError('')
    setSelected(null)
    try {
      const result =
        queryMode === 'text'
          ? await api.searchText(query, filter === 'all' ? undefined : filter)
          : await api.searchImage(queryImage!, filter === 'all' ? undefined : filter)
      setResponse(result)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '搜索失败')
    } finally {
      setSearching(false)
    }
  }

  const startScan = async () => {
    if (!libraryPath.trim()) return
    setError('')
    try {
      await api.scan(libraryPath.trim())
      setSetupOpen(false)
      await refreshStatus()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '扫描任务启动失败')
    }
  }

  const startBuild = async () => {
    setError('')
    try {
      await api.build(false)
      await refreshStatus()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '索引任务启动失败')
    }
  }

  const onImage = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null
    setQueryImage(file)
    if (file) setQueryMode('image')
  }

  const chooseExample = (value: string) => {
    setQueryMode('text')
    setQuery(value)
    window.setTimeout(() => inputRef.current?.focus(), 0)
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'is-open' : ''}`}>
        <div className="brand-row">
          <div className="brand-mark"><Aperture size={20} strokeWidth={1.8} /></div>
          <span>SceneSeek</span>
          <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)}><X size={18} /></button>
        </div>

        <nav className="nav-stack">
          <button className="nav-item active"><Search size={18} /><span>语义搜索</span></button>
          <button className="nav-item" onClick={() => setSetupOpen(true)}><Images size={18} /><span>媒体库</span><span className="nav-count">{status.library.images + status.library.videos}</span></button>
          <button className="nav-item" disabled><Sparkles size={18} /><span>发现</span><span className="soon">即将推出</span></button>
        </nav>

        <div className="sidebar-section">
          <p className="eyebrow">本地资料库</p>
          <div className="library-card">
            <div className="library-title"><HardDrive size={16} /><span>SceneSeek Library</span></div>
            <div className="library-stats">
              <span><b>{status.library.images}</b> 图片</span>
              <span><b>{status.library.videos}</b> 视频</span>
            </div>
            <div className="library-foot">
              <span className={`status-dot ${status.state}`} />
              {status.state === 'running' ? `${status.message} · ${indexedPercent}%` : `${status.index.size} 个向量`}
            </div>
            {status.state === 'running' && <div className="mini-progress"><span style={{ width: `${indexedPercent}%` }} /></div>}
          </div>
        </div>

        <div className="privacy-note">
          <Database size={15} />
          <span>媒体与特征默认只保存在本机</span>
        </div>
        <button className="sidebar-action" onClick={() => setSetupOpen(true)}><FolderSearch size={17} />管理索引<ChevronRight size={16} /></button>
      </aside>

      {sidebarOpen && <button className="sidebar-scrim" aria-label="关闭侧栏" onClick={() => setSidebarOpen(false)} />}

      <main className="main-area">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)}><Menu size={20} /></button>
          <div className="topbar-title">语义搜索</div>
          <div className="topbar-meta">
            <span className="model-pill"><span className="status-dot complete" />{status.encoder.name === 'lite' ? 'Lite 本地模型' : status.encoder.name}</span>
            <button className="icon-button" aria-label="筛选设置"><SlidersHorizontal size={18} /></button>
          </div>
        </header>

        <section className="workspace">
          <div className={`search-hero ${response ? 'compact' : ''}`}>
            <p className="eyebrow accent">MULTIMODAL RETRIEVAL</p>
            <h1>{response ? '继续寻找下一个画面' : '找到你记得的画面'}</h1>
            {!response && <p className="hero-copy">不用记住文件名。描述场景、动作或氛围，直接定位图片与视频中的相关时刻。</p>}

            <form className="search-panel" onSubmit={search}>
              <div className="mode-tabs">
                <button type="button" className={queryMode === 'text' ? 'active' : ''} onClick={() => setQueryMode('text')}><Search size={15} />文字描述</button>
                <button type="button" className={queryMode === 'image' ? 'active' : ''} onClick={() => setQueryMode('image')}><ImageIcon size={15} />以图搜图</button>
              </div>
              {queryMode === 'text' ? (
                <div className="query-row">
                  <Search size={22} className="query-icon" />
                  <input ref={inputRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：雨夜里一个人撑黑伞过马路" autoFocus />
                  <label className="upload-query" title="切换到以图搜图"><Upload size={18} /><input type="file" accept="image/*" onChange={onImage} /></label>
                  <button className="search-button" aria-label="搜索" disabled={searching}>{searching ? <LoaderCircle className="spin" size={20} /> : <ArrowRight size={20} />}</button>
                </div>
              ) : (
                <label className={`image-drop ${queryImage ? 'has-file' : ''}`}>
                  <input type="file" accept="image/*" onChange={onImage} />
                  <span className="drop-icon">{queryImage ? <Check size={22} /> : <Upload size={22} />}</span>
                  <span><b>{queryImage?.name || '选择一张参考图片'}</b><small>{queryImage ? '点击可更换图片' : '支持 JPG、PNG、WebP，最大 20 MB'}</small></span>
                  <button type="button" className="search-button image-submit" aria-label="搜索" disabled={!queryImage || searching} onClick={() => void search()}>{searching ? <LoaderCircle className="spin" size={20} /> : <ArrowRight size={20} />}</button>
                </label>
              )}
              <div className="search-options">
                <span>搜索范围</span>
                {(['all', 'image', 'video'] as Filter[]).map((value) => (
                  <button type="button" key={value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>
                    {value === 'all' ? '全部' : value === 'image' ? '图片' : '视频'}
                  </button>
                ))}
              </div>
            </form>

            {!response && (
              <div className="example-row"><span>试着搜索</span>{examples.map((example) => <button key={example} onClick={() => chooseExample(example)}>{example}</button>)}</div>
            )}
          </div>

          {error && <div className="error-banner"><CircleAlert size={18} /><span>{error}</span><button onClick={() => setError('')}><X size={16} /></button></div>}

          {!response && !hasLibrary && (
            <section className="onboarding">
              <div className="onboarding-visual"><div className="frame frame-a" /><div className="frame frame-b" /><div className="scan-line" /><Aperture size={42} /></div>
              <div><span className="step-label">开始使用</span><h2>把本地媒体变成可搜索的记忆</h2><p>添加一个图片或视频目录。SceneSeek 会在后台扫描、抽取特征并建立本地索引。</p><button className="primary-action" onClick={() => setSetupOpen(true)}><FolderSearch size={18} />添加媒体目录</button></div>
            </section>
          )}

          {response && (
            <section className="results-section">
              <div className="results-heading">
                <div><p className="eyebrow">SEARCH RESULTS</p><h2>找到 {filteredResults.length} 个相关画面</h2></div>
                <div className="result-query">“{response.query}”</div>
              </div>
              {filteredResults.length ? (
                <div className="result-grid">
                  {filteredResults.map((result, index) => <ResultCard key={`${result.media_id}-${result.start_sec ?? 0}`} result={result} index={index} onSelect={setSelected} />)}
                </div>
              ) : (
                <div className="empty-results"><Search size={28} /><h3>没有匹配结果</h3><p>换一种说法，或把搜索范围切换为“全部”。</p></div>
              )}
            </section>
          )}
        </section>
      </main>

      {setupOpen && (
        <div className="modal-layer" role="dialog" aria-modal="true" aria-labelledby="setup-title">
          <button className="modal-scrim" aria-label="关闭" onClick={() => setSetupOpen(false)} />
          <div className="setup-modal">
            <button className="modal-close" onClick={() => setSetupOpen(false)}><X size={19} /></button>
            <div className="modal-icon"><FolderSearch size={23} /></div>
            <p className="eyebrow accent">LIBRARY SETUP</p>
            <h2 id="setup-title">连接本地媒体目录</h2>
            <p>输入运行 API 的机器上的绝对路径。媒体不会上传；扫描仅保存元数据、缩略图与向量特征。</p>
            <label className="field-label">媒体目录路径<input value={libraryPath} onChange={(event) => setLibraryPath(event.target.value)} placeholder="/home/me/Pictures 或 D:\Media" /></label>
            <div className="modal-stats"><span><Images size={17} />{status.library.images} 张图片</span><span><Video size={17} />{status.library.videos} 个视频</span><span><Database size={17} />{status.index.size} 个向量</span></div>
            <div className="modal-actions"><button className="secondary-action" disabled={!hasLibrary || status.state === 'running'} onClick={() => void startBuild()}>构建待更新索引</button><button className="primary-action" disabled={!libraryPath.trim() || status.state === 'running'} onClick={() => void startScan()}>{status.state === 'running' ? <LoaderCircle className="spin" size={18} /> : <FolderSearch size={18} />}扫描目录</button></div>
          </div>
        </div>
      )}

      {selected && <DetailPanel result={selected} queryId={response?.query_id || ''} onClose={() => setSelected(null)} />}
    </div>
  )
}

function ResultCard({ result, index, onSelect }: { result: SearchResult; index: number; onSelect: (value: SearchResult) => void }) {
  const name = result.path.split(/[\\/]/).pop() || result.path
  const scoreValue = result.score.toFixed(3)
  return (
    <button className="result-card" style={{ animationDelay: `${Math.min(index, 10) * 45}ms` }} onClick={() => onSelect(result)}>
      <div className="card-image">
        <img src={result.thumbnail_url} alt={name} loading="lazy" />
        <span className="type-badge">{result.media_type === 'video' ? <><Play size={12} fill="currentColor" /> 视频</> : <><ImageIcon size={12} /> 图片</>}</span>
        {result.media_type === 'video' && <span className="time-badge">{formatTime(result.start_sec)} — {formatTime(result.end_sec)}</span>}
        <span className="score-badge">{scoreValue}</span>
      </div>
      <div className="card-copy"><strong title={name}>{name}</strong><span>{result.media_type === 'video' ? `相关片段 · ${formatDuration((result.end_sec || 0) - (result.start_sec || 0))}` : `${result.width || '—'} × ${result.height || '—'}`}</span></div>
    </button>
  )
}

function DetailPanel({ result, queryId, onClose }: { result: SearchResult; queryId: string; onClose: () => void }) {
  const [rated, setRated] = useState<number | null>(null)
  const sendFeedback = async (value: number) => {
    setRated(value)
    try { await api.feedback(queryId, result.media_id, value) } catch { setRated(null) }
  }
  return (
    <div className="detail-layer">
      <button className="detail-scrim" aria-label="关闭详情" onClick={onClose} />
      <aside className="detail-panel">
        <div className="detail-head"><div><p className="eyebrow">RESULT DETAIL</p><h2>{result.path.split(/[\\/]/).pop()}</h2></div><button className="icon-button" aria-label="关闭详情" onClick={onClose}><X size={20} /></button></div>
        <div className="detail-preview">
          {result.media_type === 'video' ? <video src={result.clip_url} controls autoPlay poster={result.thumbnail_url} /> : <img src={result.media_url} alt="检索结果" />}
        </div>
        {result.media_type === 'video' && <div className="moment-row"><div className="moment-icon"><Play size={16} fill="currentColor" /></div><div><span>相关时刻</span><strong>{formatTime(result.start_sec)} — {formatTime(result.end_sec)}</strong></div><div className="moment-duration">{formatDuration((result.end_sec || 0) - (result.start_sec || 0))}</div></div>}
        <dl className="metadata-list"><div><dt>匹配分数</dt><dd>{result.score.toFixed(3)}</dd></div><div><dt>媒体类型</dt><dd>{result.media_type === 'video' ? '视频片段' : '图片'}</dd></div><div><dt>分辨率</dt><dd>{result.width || '—'} × {result.height || '—'}</dd></div>{result.duration && <div><dt>完整时长</dt><dd>{formatTime(result.duration)}</dd></div>}</dl>
        <div className="path-block"><span>来源路径</span><code>{result.path}</code></div>
        <div className="feedback-row"><span>这个结果相关吗？</span><div><button className={rated === 3 ? 'rated' : ''} onClick={() => void sendFeedback(3)}><ThumbsUp size={17} /></button><button className={rated === 0 ? 'rated' : ''} onClick={() => void sendFeedback(0)}><ThumbsDown size={17} /></button></div></div>
      </aside>
    </div>
  )
}

function formatTime(value?: number) {
  const seconds = Math.max(0, value || 0)
  const minutes = Math.floor(seconds / 60)
  return `${String(minutes).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

function formatDuration(value: number) {
  return `${Math.max(0, value).toFixed(value < 10 ? 1 : 0)} 秒`
}

export default App
