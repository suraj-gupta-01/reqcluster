

import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, FileText, CheckCircle, AlertCircle, Loader, ChevronRight, Settings, X, BookOpen, Sparkles, Wrench, Workflow, Brain, Network, Activity } from 'lucide-react'
import { uploadRequirements, runClustering, getProgress, getSession } from '../utils/api.js'

const STEP_ORDER = ['embedding', 'umap', 'clustering', 'labeling', 'graph', 'done']
const STEP_LABELS = {
  embedding: 'Generating Embeddings',
  umap: 'UMAP Reduction',
  clustering: 'HDBSCAN Clustering',
  labeling: 'c-TF-IDF Labeling',
  graph: 'Building Similarity Graph',
  done: 'Complete',
}

function ProgressStep({ step, currentStep, message }) {
  const steps = STEP_ORDER.slice(0, -1)
  const currentIdx = steps.indexOf(currentStep)
  const thisIdx = steps.indexOf(step)

  let state = 'pending'
  if (currentStep === 'done') state = 'done'
  else if (thisIdx < currentIdx) state = 'done'
  else if (thisIdx === currentIdx) state = 'active'

  return (
    <div className={`flex items-center gap-3 py-2 transition-all ${state === 'pending' ? 'opacity-30' : ''}`}>
      <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold
        ${state === 'done' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
        : state === 'active' ? 'bg-brand-600/30 text-brand-400 border border-brand-500/50'
        : 'bg-gray-800 text-gray-600 border border-gray-700'}`}>
        {state === 'done' ? '✓' : thisIdx + 1}
      </div>
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium ${state === 'active' ? 'text-white' : state === 'done' ? 'text-emerald-400' : 'text-gray-500'}`}>
          {STEP_LABELS[step]}
        </div>
        {state === 'active' && message && (
          <div className="text-xs text-gray-500 mt-0.5 truncate">{message}</div>
        )}
      </div>
      {state === 'active' && <Loader size={14} className="animate-spin text-brand-400 flex-shrink-0" />}
    </div>
  )
}

export default function UploadPage({ onSessionCreated }) {
  const navigate = useNavigate()
  const fileInputRef = useRef(null)
  const pollRef = useRef(null)

  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState(null)
  const [uploadResult, setUploadResult] = useState(null)
  const [phase, setPhase] = useState('idle') // idle | uploading | uploaded | clustering | done | error
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState({ step: 'idle', progress: 0, message: '' })
  const [showParams, setShowParams] = useState(false)
  const [params, setParams] = useState({ min_cluster_size: '', min_samples: 3, similarity_threshold: 0.65 })

  const validateAndSetFile = useCallback((f) => {
    if (!f.name.match(/\.(csv|xlsx|xls)$/i)) {
      setError('Only CSV and XLSX files are supported.')
      return
    }
    setFile(f)
    setError(null)
    setUploadResult(null)
    setPhase('idle')
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) validateAndSetFile(dropped)
  }, [validateAndSetFile])

  const handleUpload = async () => {
    if (!file) return
    setPhase('uploading')
    setError(null)
    try {
      const result = await uploadRequirements(file)
      setUploadResult(result)
      setPhase('uploaded')
      onSessionCreated?.(result.session_id, 'uploaded')
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed.')
      setPhase('error')
    }
  }

  const startPolling = (sessionId) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const prog = await getProgress(sessionId)
        setProgress(prog)
        if (prog.step === 'done' || prog.step === 'error') {
          clearInterval(pollRef.current)
          if (prog.step === 'done') {
            const session = await getSession(sessionId)
            onSessionCreated?.(sessionId, session.status)
            setPhase('done')
          } else {
            setError(prog.message)
            setPhase('error')
          }
        }
      } catch {
        clearInterval(pollRef.current)
      }
    }, 800)
  }

  const handleCluster = async () => {
    if (!uploadResult) return
    setPhase('clustering')
    setProgress({ step: 'embedding', progress: 0, message: 'Starting pipeline...' })
    try {
      const payload = {
        session_id: uploadResult.session_id,
        min_samples: params.min_samples,
        similarity_threshold: params.similarity_threshold,
      }
      if (params.min_cluster_size) payload.min_cluster_size = parseInt(params.min_cluster_size)
      startPolling(uploadResult.session_id)
      await runClustering(payload)
    } catch (err) {
      clearInterval(pollRef.current)
      setError(err.response?.data?.detail || 'Clustering failed.')
      setPhase('error')
    }
  }

  const goToOverview = () => navigate(`/overview/${uploadResult.session_id}`)

  const reset = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    setFile(null); setUploadResult(null); setPhase('idle')
    setError(null); setProgress({ step: 'idle', progress: 0, message: '' })
  }

  const overallPct = phase === 'done' ? 100 : phase === 'clustering' ? progress.progress : phase === 'uploaded' ? 0 : 0

  return (
    <div className="p-8 max-w-6xl animate-fade-up">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column - Forms / Upload Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Header */}
          <div className="mb-6">
            <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-brand-400/80 mb-2">
              Stage 01 — Ingest
            </div>
            <h1 className="text-3xl font-bold text-white tracking-tight">Upload requirements</h1>
            <div className="mt-2 h-px w-16 bg-gradient-to-r from-brand-500 to-transparent" />
            <p className="text-gray-400 mt-3 max-w-lg">
              Import a CSV or XLSX of engineering requirements. The pipeline embeds, reduces,
              clusters, labels, and graphs them automatically.
            </p>
          </div>

          {/* Drop zone */}
          <div
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => !file && fileInputRef.current?.click()}
            className={`relative border border-dashed rounded-2xl p-10 text-center transition-all duration-200 cursor-pointer
              ${dragOver ? 'border-brand-500 bg-brand-500/[0.07]' : file ? 'border-brand-500/20 bg-white/[0.02] cursor-default' : 'border-brand-500/20 hover:border-brand-500/40 hover:bg-white/[0.02]'}`}
          >
            <input ref={fileInputRef} type="file" accept=".csv,.xlsx,.xls" className="hidden"
              onChange={e => e.target.files[0] && validateAndSetFile(e.target.files[0])} />

            {!file ? (
              <div>
                <div className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4 border border-brand-500/20 bg-brand-500/[0.06]">
                  <Upload size={22} className={`${dragOver ? 'text-brand-300' : 'text-brand-400/70'} transition-colors`} />
                </div>
                <p className="text-sm font-medium text-gray-300 mb-1">
                  {dragOver ? 'Drop to upload' : 'Drag & drop or click to browse'}
                </p>
                <p className="text-xs text-gray-600">CSV or XLSX · columns: id, text, module, section</p>
              </div>
            ) : (
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-brand-600/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <FileText size={18} className="text-brand-400" />
                </div>
                <div className="flex-1 text-left min-w-0">
                  <p className="text-sm font-medium text-white truncate">{file.name}</p>
                  <p className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
                {phase === 'idle' && (
                  <button onClick={e => { e.stopPropagation(); reset() }}
                    className="text-gray-600 hover:text-gray-400 transition-colors p-1">
                    <X size={16} />
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-3 p-4 rounded-lg bg-red-950/30 border border-red-900/50 text-red-400 text-sm">
              <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Upload result summary */}
          {uploadResult && phase !== 'idle' && (
            <div className="card p-4 flex gap-6">
              <div>
                <div className="text-2xl font-bold text-white">{uploadResult.total_requirements}</div>
                <div className="text-xs text-gray-500 mt-0.5">Requirements</div>
              </div>
              {uploadResult.duplicates_removed > 0 && (
                <div>
                  <div className="text-2xl font-bold text-amber-400">{uploadResult.duplicates_removed}</div>
                  <div className="text-xs text-gray-500 mt-0.5">Duplicates removed</div>
                </div>
              )}
              {uploadResult.empty_removed > 0 && (
                <div>
                  <div className="text-2xl font-bold text-red-400">{uploadResult.empty_removed}</div>
                  <div className="text-xs text-gray-500 mt-0.5">Empty removed</div>
                </div>
              )}
            </div>
          )}

          {/* Clustering params (collapsible) */}
          {phase === 'uploaded' && (
            <div className="card overflow-hidden">
              <button onClick={() => setShowParams(p => !p)}
                className="w-full flex items-center justify-between px-4 py-3 text-sm text-gray-400 hover:text-gray-200 transition-colors">
                <div className="flex items-center gap-2">
                  <Settings size={14} />
                  <span className="font-medium">Clustering Parameters</span>
                  <span className="text-xs text-gray-600">(optional)</span>
                </div>
                <ChevronRight size={14} className={`transition-transform ${showParams ? 'rotate-90' : ''}`} />
              </button>
              {showParams && (
                <div className="px-4 pb-4 border-t border-gray-800 pt-4 grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5">Min Cluster Size</label>
                    <input type="number" min="2" placeholder="auto"
                      value={params.min_cluster_size}
                      onChange={e => setParams(p => ({ ...p, min_cluster_size: e.target.value }))}
                      className="input text-sm w-full" />
                    <p className="text-xs text-gray-600 mt-1">Default: max(5, N/50)</p>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5">Min Samples</label>
                    <input type="number" min="1"
                      value={params.min_samples}
                      onChange={e => setParams(p => ({ ...p, min_samples: parseInt(e.target.value) || 3 }))}
                      className="input text-sm w-full" />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5">Similarity Threshold</label>
                    <input type="number" min="0.5" max="0.99" step="0.05"
                      value={params.similarity_threshold}
                      onChange={e => setParams(p => ({ ...p, similarity_threshold: parseFloat(e.target.value) || 0.65 }))}
                      className="input text-sm w-full" />
                    <p className="text-xs text-gray-600 mt-1">For edge filtering in graph</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Pipeline progress */}
          {phase === 'clustering' && (
            <div className="card p-5">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-300">Pipeline Running</span>
                <span className="text-sm font-mono text-brand-400">{overallPct}%</span>
              </div>
              <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden mb-5">
                <div className="h-full bg-brand-500 rounded-full transition-all duration-500"
                  style={{ width: `${overallPct}%` }} />
              </div>
              <div className="divide-y divide-gray-800/50">
                {STEP_ORDER.slice(0, -1).map(step => (
                  <ProgressStep key={step} step={step}
                    currentStep={progress.step}
                    message={progress.message} />
                ))}
              </div>
            </div>
          )}

          {/* Done state */}
          {phase === 'done' && (
            <div className="card p-5 border-emerald-900/40 bg-emerald-950/10">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle size={20} className="text-emerald-400" />
                <span className="font-semibold text-emerald-300">Clustering Complete!</span>
              </div>
              <p className="text-sm text-gray-400 mb-4">Your requirements have been clustered. Click below to explore the results.</p>
              <button onClick={goToOverview} className="btn-primary flex items-center gap-2">
                <span>View Results</span>
                <ChevronRight size={15} />
              </button>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-3">
            {phase === 'idle' && file && (
              <button onClick={handleUpload} className="btn-primary flex items-center gap-2">
                <Upload size={15} />
                <span>Upload File</span>
              </button>
            )}
            {phase === 'uploading' && (
              <button disabled className="btn-primary flex items-center gap-2">
                <Loader size={15} className="animate-spin" />
                <span>Uploading...</span>
              </button>
            )}
            {phase === 'uploaded' && (
              <div className="flex gap-3">
                <button onClick={handleCluster} className="btn-primary flex items-center gap-2">
                  <span>Run Clustering Pipeline</span>
                  <ChevronRight size={15} />
                </button>
                <button onClick={reset} className="btn-secondary text-sm">Reset</button>
              </div>
            )}
            {phase === 'error' && (
              <button onClick={reset} className="btn-secondary">Start Over</button>
            )}
          </div>

          {/* Format guide */}
          {phase === 'idle' && !file && (
            <div className="mt-8">
              <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-gray-500 mb-3">Expected CSV / XLSX format</p>
              <div className="panel p-5 font-mono text-xs text-gray-400 overflow-x-auto">
                <div className="text-gray-600 mb-1"># columns: id, text, module, section</div>
                <div><span className="text-brand-400">REQ-001</span>,<span className="text-signal-400">"Cooling fan shall activate above 70°C"</span>,<span className="text-emerald-400">Thermal</span>,<span className="text-brand-300">Temperature</span></div>
                <div><span className="text-brand-400">REQ-002</span>,<span className="text-signal-400">"System shall survive 15g shock"</span>,<span className="text-emerald-400">Mechanical</span>,<span className="text-brand-300">Reliability</span></div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column - How to Use Guide */}
        <div className="lg:col-span-1 space-y-4 animate-fade-in">
          <div className="card card-interactive card-accent-top p-5 bg-white/[0.02] border border-white/[0.05]">
            <div className="flex items-center gap-2 mb-4 pb-3 border-b border-white/[0.05]">
              <BookOpen size={18} className="text-brand-400" />
              <h2 className="text-sm font-semibold text-white uppercase tracking-wider">How to Use Guide</h2>
            </div>

            <div className="space-y-4">
              <div>
                <span className="text-xs font-semibold text-brand-400 uppercase tracking-wider block mb-1">1. Import & Cluster</span>
                <p className="text-xs text-gray-400 leading-relaxed">
                  Upload your raw requirement sheets. The pipeline generates SBERT semantic embeddings, projects them into 2D space with UMAP, clusters related requirements via HDBSCAN, and extracts dominant keywords with c-TF-IDF.
                </p>
              </div>

              <div>
                <span className="text-xs font-semibold text-brand-400 uppercase tracking-wider block mb-1">2. Explore the Workspace</span>
                <ul className="text-xs text-gray-400 space-y-2 mt-1">
                  <li className="flex items-start gap-2">
                    <Activity size={12} className="text-brand-400 mt-0.5 flex-shrink-0" />
                    <span><strong>Scatter Plot</strong>: Interactively inspect your requirement embeddings layout and color-coded cluster density.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Network size={12} className="text-brand-400 mt-0.5 flex-shrink-0" />
                    <span><strong>Similarity Graph</strong>: Visualize similarity connections between requirements with tunable edge filters.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Workflow size={12} className="text-brand-400 mt-0.5 flex-shrink-0" />
                    <span><strong>Dependency Tree</strong>: Explore hierarchical dependency layers in dynamic 3D representations.</span>
                  </li>
                </ul>
              </div>

              <div>
                <span className="text-xs font-semibold text-brand-400 uppercase tracking-wider block mb-1">3. Apply Intelligence</span>
                <ul className="text-xs text-gray-400 space-y-2 mt-1">
                  <li className="flex items-start gap-2">
                    <Sparkles size={12} className="text-brand-400 mt-0.5 flex-shrink-0" />
                    <span><strong>Enrichment</strong>: Use LLMs to expand requirements, extract intent, assumptions, and flags.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Wrench size={12} className="text-brand-400 mt-0.5 flex-shrink-0" />
                    <span><strong>Refinement</strong>: Automatically clean vague items, solve conflicts, and review changes.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Brain size={12} className="text-brand-400 mt-0.5 flex-shrink-0" />
                    <span><strong>Active Learning</strong>: Train classification models with iterative human-in-the-loop validation.</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
