import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity,
  AlertCircle,
  CheckCircle,
  ChevronRight,
  Database,
  GitCompare,
  Loader,
  Network,
  Play,
  RefreshCw,
  Sparkles,
} from 'lucide-react'

import {
  clusterSession,
  enrichSession,
  getEnrichmentResults,
  getEnrichmentStatus,
  getErrorMessage,
  getProgress,
  getRequirements,
  getSessions,
} from '../utils/api.js'
import AblationReportPanel from '../components/AblationReportPanel.jsx'
import DomainVocabularyTags from '../components/DomainVocabularyTags.jsx'
import EmbeddingComparisonPanel from '../components/EmbeddingComparisonPanel.jsx'
import EnrichmentResultsTable from '../components/EnrichmentResultsTable.jsx'
import EnrichmentStatusCard from '../components/EnrichmentStatusCard.jsx'

const MISSING_ENRICHMENT_MESSAGE = 'Run enrichment for this session before clustering with enriched or hybrid embeddings.'

const initialEnrichmentOptions = {
  provider_name: 'mock',
  embedding_mode: 'hybrid',
  batch_size: 8,
  max_concurrency: 4,
  timeout_seconds: 30,
  force_refresh: false,
  fail_fast: false,
  use_cache: true,
}

const initialClusterOptions = {
  similarity_threshold: 0.65,
  min_cluster_size: '',
  min_samples: '',
  enable_embedding_comparison: true,
  run_ablation: false,
}

function clampNumber(value, min, max) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return min
  return Math.min(max, Math.max(min, parsed))
}

function isPositiveIntegerOrEmpty(value, min = 1) {
  if (value === '' || value === null || value === undefined) return true
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed >= min
}

function statusReady(status) {
  return !!status && status.status === 'complete' && status.total > 0 && status.succeeded === status.total
}

function formErrors(enrichmentOptions, clusterOptions) {
  const errors = {}
  if (enrichmentOptions.batch_size < 1 || enrichmentOptions.batch_size > 64) errors.batch_size = 'Batch size must be between 1 and 64.'
  if (enrichmentOptions.max_concurrency < 1 || enrichmentOptions.max_concurrency > 16) errors.max_concurrency = 'Max concurrency must be between 1 and 16.'
  if (enrichmentOptions.timeout_seconds < 1 || enrichmentOptions.timeout_seconds > 120) errors.timeout_seconds = 'Timeout must be between 1 and 120 seconds.'
  if (clusterOptions.similarity_threshold < 0 || clusterOptions.similarity_threshold > 1) errors.similarity_threshold = 'Similarity threshold must be between 0 and 1.'
  if (!isPositiveIntegerOrEmpty(clusterOptions.min_cluster_size, 2)) errors.min_cluster_size = 'Min cluster size must be 2 or greater, or empty.'
  if (!isPositiveIntegerOrEmpty(clusterOptions.min_samples, 1)) errors.min_samples = 'Min samples must be a positive integer, or empty.'
  return errors
}

export default function EnrichmentPage() {
  const statusPollRef = useRef(null)
  const clusterPollRef = useRef(null)
  const mountedRef = useRef(true)

  const [sessions, setSessions] = useState([])
  const [selectedSessionId, setSelectedSessionId] = useState('')
  const [requirements, setRequirements] = useState([])
  const [enrichmentOptions, setEnrichmentOptions] = useState(initialEnrichmentOptions)
  const [clusterOptions, setClusterOptions] = useState(initialClusterOptions)

  const [loadingSessions, setLoadingSessions] = useState(true)
  const [loadingStatus, setLoadingStatus] = useState(false)
  const [loadingResults, setLoadingResults] = useState(false)
  const [enriching, setEnriching] = useState(false)
  const [clustering, setClustering] = useState(false)

  const [status, setStatus] = useState(null)
  const [results, setResults] = useState([])
  const [domainVocabulary, setDomainVocabulary] = useState([])
  const [qualityReport, setQualityReport] = useState(null)
  const [clusterResult, setClusterResult] = useState(null)
  const [clusterProgress, setClusterProgress] = useState(null)

  const [error, setError] = useState(null)
  const [clusterError, setClusterError] = useState(null)
  const [notice, setNotice] = useState(null)

  const validation = useMemo(
    () => formErrors(enrichmentOptions, clusterOptions),
    [enrichmentOptions, clusterOptions],
  )
  const hasValidationErrors = Object.keys(validation).length > 0
  const enrichmentReady = statusReady(status)

  const vocabularyForDisplay = useMemo(() => {
    if (domainVocabulary?.length) return domainVocabulary
    const terms = []
    const seen = new Set()
    results.forEach(row => {
      row.domain_terms?.forEach(term => {
        const key = String(term).toLowerCase()
        if (!seen.has(key)) {
          seen.add(key)
          terms.push(term)
        }
      })
    })
    return terms
  }, [domainVocabulary, results])

  const selectedSession = useMemo(
    () => sessions.find(session => String(session.id) === String(selectedSessionId)),
    [sessions, selectedSessionId],
  )

  const stopStatusPolling = useCallback(() => {
    if (statusPollRef.current) clearInterval(statusPollRef.current)
    statusPollRef.current = null
  }, [])

  const stopClusterPolling = useCallback(() => {
    if (clusterPollRef.current) clearInterval(clusterPollRef.current)
    clusterPollRef.current = null
  }, [])

  const loadStatus = useCallback(async (sessionId = selectedSessionId, quiet = false) => {
    if (!sessionId) return null
    if (!quiet) setLoadingStatus(true)
    try {
      const next = await getEnrichmentStatus(sessionId)
      if (mountedRef.current) setStatus(next)
      return next
    } catch (err) {
      if (mountedRef.current && !quiet) setError(getErrorMessage(err, 'Failed to load enrichment status.'))
      return null
    } finally {
      if (mountedRef.current && !quiet) setLoadingStatus(false)
    }
  }, [selectedSessionId])

  const loadResults = useCallback(async (sessionId = selectedSessionId) => {
    if (!sessionId) return []
    setLoadingResults(true)
    try {
      const [rows, reqs] = await Promise.all([
        getEnrichmentResults(sessionId),
        getRequirements(sessionId),
      ])
      if (mountedRef.current) {
        setResults(rows)
        setRequirements(reqs)
      }
      return rows
    } catch (err) {
      if (mountedRef.current) setError(getErrorMessage(err, 'Failed to load enrichment results.'))
      return []
    } finally {
      if (mountedRef.current) setLoadingResults(false)
    }
  }, [selectedSessionId])

  const startStatusPolling = useCallback((sessionId) => {
    stopStatusPolling()
    let polls = 0
    statusPollRef.current = setInterval(async () => {
      polls += 1
      const next = await loadStatus(sessionId, true)
      if (!mountedRef.current) return
      if (next?.status === 'complete' || next?.status === 'failed' || polls >= 90) {
        stopStatusPolling()
      }
    }, 1000)
  }, [loadStatus, stopStatusPolling])

  useEffect(() => {
    mountedRef.current = true
    const load = async () => {
      try {
        const list = await getSessions()
        if (!mountedRef.current) return
        setSessions(list)
        const latest = list?.[0]?.id
        if (latest) setSelectedSessionId(String(latest))
      } catch (err) {
        if (mountedRef.current) setError(getErrorMessage(err, 'Failed to load sessions.'))
      } finally {
        if (mountedRef.current) setLoadingSessions(false)
      }
    }
    load()
    return () => {
      mountedRef.current = false
      stopStatusPolling()
      stopClusterPolling()
    }
  }, [stopClusterPolling, stopStatusPolling])

  useEffect(() => {
    if (!selectedSessionId) return
    const timer = window.setTimeout(() => {
      if (!mountedRef.current) return
      setError(null)
      setClusterError(null)
      setNotice(null)
      setStatus(null)
      setResults([])
      setRequirements([])
      setDomainVocabulary([])
      setQualityReport(null)
      setClusterResult(null)
      void loadStatus(selectedSessionId, true)
      void loadResults(selectedSessionId)
    }, 0)
    return () => window.clearTimeout(timer)
  }, [selectedSessionId, loadStatus, loadResults])

  const updateEnrichmentOption = (key, value) => {
    setEnrichmentOptions(prev => ({ ...prev, [key]: value }))
  }

  const updateClusterOption = (key, value) => {
    setClusterOptions(prev => ({ ...prev, [key]: value }))
  }

  const handleStartEnrichment = async () => {
    if (!selectedSessionId || hasValidationErrors) return
    setError(null)
    setNotice(null)
    setClusterError(null)
    setEnriching(true)
    startStatusPolling(selectedSessionId)
    try {
      const response = await enrichSession({
        session_id: Number(selectedSessionId),
        ...enrichmentOptions,
      })
      if (!mountedRef.current) return
      setStatus({
        session_id: response.session_id,
        status: response.status,
        total: response.total,
        succeeded: response.succeeded,
        failed: response.failed,
        pending: Math.max((response.total || 0) - (response.succeeded || 0) - (response.failed || 0), 0),
        provider: response.provider,
        model: response.model,
        latest_run_created_at: new Date().toISOString(),
        warnings: response.warnings || [],
      })
      setDomainVocabulary(response.domain_vocabulary || [])
      setQualityReport(response.quality_report || null)
      setNotice(response.status === 'complete' ? 'Enrichment completed.' : 'Enrichment finished with warnings.')
      await loadResults(selectedSessionId)
      await loadStatus(selectedSessionId, true)
    } catch (err) {
      if (mountedRef.current) setError(getErrorMessage(err, 'Enrichment failed.'))
    } finally {
      stopStatusPolling()
      if (mountedRef.current) setEnriching(false)
    }
  }

  const startClusterPolling = (sessionId, onDone) => {
    stopClusterPolling()
    clusterPollRef.current = setInterval(async () => {
      try {
        const progress = await getProgress(sessionId)
        if (!mountedRef.current) return
        setClusterProgress(progress)
        if (progress.step === 'done') {
          stopClusterPolling()
          if (onDone) onDone(progress)
        } else if (progress.step === 'error') {
          stopClusterPolling()
          if (mountedRef.current) setClusterError(progress.message || 'Clustering failed.')
        }
      } catch {
        stopClusterPolling()
      }
    }, 1000)
  }

  const handleCluster = async (mode) => {
    if (!selectedSessionId) return
    if (mode !== 'base' && !enrichmentReady) {
      setClusterError(MISSING_ENRICHMENT_MESSAGE)
      return
    }
    if (hasValidationErrors) return

    setClusterError(null)
    setClusterResult(null)
    setClustering(true)
    setClusterProgress({ step: 'starting', progress: 0, message: 'Starting clustering...' })

    const payload = {
      session_id: Number(selectedSessionId),
      embedding_mode: mode,
      similarity_threshold: Number(clusterOptions.similarity_threshold),
      enable_embedding_comparison: mode === 'base' ? false : Boolean(clusterOptions.enable_embedding_comparison),
      run_ablation: Boolean(clusterOptions.run_ablation),
    }
    if (clusterOptions.min_cluster_size !== '') payload.min_cluster_size = Number(clusterOptions.min_cluster_size)
    if (clusterOptions.min_samples !== '') payload.min_samples = Number(clusterOptions.min_samples)

    try {
      // POST /cluster returns 202 immediately — pipeline runs in background.
      // We start polling for progress, then fetch the real session result on completion.
      await clusterSession(payload)
      if (!mountedRef.current) return

      startClusterPolling(selectedSessionId, async (doneProgress) => {
        if (!mountedRef.current) return
        setClusterResult({
          total_clusters: doneProgress.total_clusters,
          noise_count: doneProgress.noise_count,
          embedding_mode: doneProgress.embedding_mode || mode,
          embedding_comparison: doneProgress.embedding_comparison || null,
          ablation_report: doneProgress.ablation_report || null,
        })
        localStorage.setItem(`reqcluster:lastEmbeddingMode:${selectedSessionId}`, doneProgress.embedding_mode || mode)
        setNotice(`${mode} clustering completed.`)
        setClustering(false)
      })
    } catch (err) {
      stopClusterPolling()
      if (mountedRef.current) {
        setClusterError(getErrorMessage(err, 'Clustering failed.'))
        setClustering(false)
      }
    }
  }

  if (loadingSessions) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader size={24} className="animate-spin text-brand-400" />
      </div>
    )
  }

  if (!sessions.length) {
    return (
      <div className="p-8 max-w-3xl">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white">Enrichment</h1>
          <p className="text-gray-400 mt-1">Upload a requirements file first.</p>
        </div>
        <Link to="/" className="btn-primary inline-flex items-center gap-2">
          <Database size={15} />
          Upload Requirements
        </Link>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-7xl space-y-6">
      <div className="flex items-start justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Sparkles size={20} className="text-brand-400" />
            <h1 className="text-2xl font-bold text-white">Phase 2 Enrichment</h1>
          </div>
          <p className="text-gray-400 max-w-3xl">
            Run LLM enrichment, inspect domain context, then cluster with base, enriched, or hybrid embeddings.
            Base clustering is the Phase 1 behavior. Hybrid uses the original requirement plus LLM-enriched context.
            The Mock provider is offline and safe for testing.
          </p>
        </div>
        <div className="flex gap-2">
          <Link to={`/scatter/${selectedSessionId}`} className="btn-secondary text-sm flex items-center gap-2">
            <Activity size={14} />
            Scatter
          </Link>
          <Link to={`/graph/${selectedSessionId}`} className="btn-secondary text-sm flex items-center gap-2">
            <Network size={14} />
            Graph
          </Link>
        </div>
      </div>

      {error && <Alert text={error} tone="red" />}
      {notice && <Alert text={notice} tone="emerald" icon={CheckCircle} />}

      <div className="grid lg:grid-cols-[380px_minmax(0,1fr)] gap-6">
        <div className="space-y-6">
          <section className="card p-4 space-y-4">
            <h2 className="text-sm font-semibold text-gray-200">Session And Provider</h2>
            <div>
              <label htmlFor="session-select" className="block text-xs font-medium text-gray-400 mb-1.5">Session</label>
              <select
                id="session-select"
                value={selectedSessionId}
                onChange={e => setSelectedSessionId(e.target.value)}
                className="input text-sm w-full"
              >
                {sessions.map(session => (
                  <option key={session.id} value={session.id}>
                    {session.name || session.filename} - {session.total_requirements} requirements
                  </option>
                ))}
              </select>
              {selectedSession && <p className="text-xs text-gray-500 mt-1">{selectedSession.filename}</p>}
            </div>

            <SelectControl
              id="provider-name"
              label="Provider"
              value={enrichmentOptions.provider_name}
              onChange={value => updateEnrichmentOption('provider_name', value)}
              options={[
                ['mock', 'Mock'],
                ['openai_compatible', 'Cloud LLM'],
                ['local', 'Local LLM'],
              ]}
            />
            <SelectControl
              id="enrichment-mode"
              label="Recommended embedding mode"
              value={enrichmentOptions.embedding_mode}
              onChange={value => updateEnrichmentOption('embedding_mode', value)}
              options={[
                ['hybrid', 'hybrid'],
                ['enriched', 'enriched'],
              ]}
            />

            <div className="grid grid-cols-3 gap-3">
              <NumberControl
                id="batch-size"
                label="Batch"
                min={1}
                max={64}
                value={enrichmentOptions.batch_size}
                error={validation.batch_size}
                onChange={value => updateEnrichmentOption('batch_size', clampNumber(value, 1, 64))}
              />
              <NumberControl
                id="max-concurrency"
                label="Concurrency"
                min={1}
                max={16}
                value={enrichmentOptions.max_concurrency}
                error={validation.max_concurrency}
                onChange={value => updateEnrichmentOption('max_concurrency', clampNumber(value, 1, 16))}
              />
              <NumberControl
                id="timeout-seconds"
                label="Timeout"
                min={1}
                max={120}
                value={enrichmentOptions.timeout_seconds}
                error={validation.timeout_seconds}
                onChange={value => updateEnrichmentOption('timeout_seconds', clampNumber(value, 1, 120))}
              />
            </div>

            <Toggle label="Force refresh" checked={enrichmentOptions.force_refresh} onChange={value => updateEnrichmentOption('force_refresh', value)} />
            <Toggle label="Use file cache" checked={enrichmentOptions.use_cache} onChange={value => updateEnrichmentOption('use_cache', value)} />
            <Toggle label="Fail fast" checked={enrichmentOptions.fail_fast} onChange={value => updateEnrichmentOption('fail_fast', value)} />

            <div className="flex gap-2 pt-2">
              <button
                type="button"
                onClick={handleStartEnrichment}
                disabled={enriching || hasValidationErrors}
                className="btn-primary flex items-center gap-2"
              >
                {enriching ? <Loader size={15} className="animate-spin" /> : <Sparkles size={15} />}
                Start Enrichment
              </button>
              <button type="button" onClick={() => loadStatus()} className="btn-secondary text-sm flex items-center gap-2">
                <RefreshCw size={14} />
                Refresh Status
              </button>
            </div>
          </section>

          <EnrichmentStatusCard status={status} loading={loadingStatus || enriching} />

          <section className="card p-4 space-y-4">
            <h2 className="text-sm font-semibold text-gray-200">Clustering Controls</h2>
            <div className="grid grid-cols-3 gap-3">
              <NumberControl
                id="similarity-threshold"
                label="Similarity"
                min={0}
                max={1}
                step={0.05}
                value={clusterOptions.similarity_threshold}
                error={validation.similarity_threshold}
                onChange={value => updateClusterOption('similarity_threshold', Number(value))}
              />
              <OptionalNumberControl
                id="min-cluster-size"
                label="Min cluster"
                min={2}
                value={clusterOptions.min_cluster_size}
                error={validation.min_cluster_size}
                onChange={value => updateClusterOption('min_cluster_size', value)}
              />
              <OptionalNumberControl
                id="min-samples"
                label="Min samples"
                min={1}
                value={clusterOptions.min_samples}
                error={validation.min_samples}
                onChange={value => updateClusterOption('min_samples', value)}
              />
            </div>

            <Toggle label="Embedding comparison" checked={clusterOptions.enable_embedding_comparison} onChange={value => updateClusterOption('enable_embedding_comparison', value)} />
            <Toggle label="Run ablation" checked={clusterOptions.run_ablation} onChange={value => updateClusterOption('run_ablation', value)} />

            {!enrichmentReady && (
              <p className="text-xs text-amber-300">{MISSING_ENRICHMENT_MESSAGE}</p>
            )}
            {clusterError && <Alert text={clusterError} tone="red" />}

            <div className="grid gap-2">
              <ClusterButton mode="base" loading={clustering} onClick={() => handleCluster('base')} />
              <ClusterButton mode="hybrid" loading={clustering} disabled={!enrichmentReady} onClick={() => handleCluster('hybrid')} />
              <ClusterButton mode="enriched" loading={clustering} disabled={!enrichmentReady} onClick={() => handleCluster('enriched')} />
            </div>

            {clusterProgress && (
              <div className="bg-gray-950/50 border border-gray-800 rounded-lg p-3">
                <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                  <span>{clusterProgress.step}</span>
                  <span>{clusterProgress.progress ?? 0}%</span>
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-brand-500" style={{ width: `${clusterProgress.progress ?? 0}%` }} />
                </div>
                {clusterProgress.message && <p className="text-xs text-gray-500 mt-2">{clusterProgress.message}</p>}
              </div>
            )}
          </section>
        </div>

        <div className="space-y-6 min-w-0">
          <DomainVocabularyTags terms={vocabularyForDisplay} />

          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-200">Enrichment Results</h2>
            <button type="button" onClick={() => loadResults()} className="btn-secondary text-sm flex items-center gap-2">
              {loadingResults ? <Loader size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              Load Results
            </button>
          </div>
          <EnrichmentResultsTable results={results} requirements={requirements} />

          {qualityReport?.aggregate && (
            <div className="card p-4">
              <h2 className="text-sm font-semibold text-gray-200 mb-3">Quality Summary</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Metric label="Succeeded" value={qualityReport.succeeded ?? 0} />
                <Metric label="Failed" value={qualityReport.failed ?? 0} />
                <Metric label="Mean risk" value={qualityReport.aggregate.mean_hallucination_risk_score ?? '-'} />
                <Metric label="Mean confidence" value={qualityReport.aggregate.mean_adjusted_confidence_score ?? '-'} />
              </div>
            </div>
          )}

          {clusterResult && (
            <section className="space-y-4">
              <div className="card p-4 border-emerald-900/40 bg-emerald-950/10">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 text-emerald-300 font-semibold">
                      <CheckCircle size={16} />
                      Clustering Complete
                    </div>
                    <p className="text-sm text-gray-400 mt-1">
                      {clusterResult.total_clusters} clusters, {clusterResult.noise_count} noise points using {clusterResult.embedding_mode || 'selected'} mode.
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Link to={`/scatter/${selectedSessionId}`} className="btn-primary text-sm flex items-center gap-2">
                      Scatter
                      <ChevronRight size={14} />
                    </Link>
                    <Link to={`/graph/${selectedSessionId}`} className="btn-secondary text-sm">Graph</Link>
                  </div>
                </div>
              </div>
              <EmbeddingComparisonPanel report={clusterResult.embedding_comparison} />
              <AblationReportPanel report={clusterResult.ablation_report} />
            </section>
          )}


        </div>
      </div>
    </div>
  )
}

function Alert({ text, tone = 'amber', icon: Icon = AlertCircle }) {
  const style = tone === 'red'
    ? 'bg-red-950/30 border-red-900/50 text-red-300'
    : tone === 'emerald'
      ? 'bg-emerald-950/20 border-emerald-900/40 text-emerald-300'
      : 'bg-amber-950/20 border-amber-900/40 text-amber-300'
  return (
    <div className={`flex items-start gap-3 p-3 rounded-lg border text-sm ${style}`}>
      <Icon size={15} className="mt-0.5 flex-shrink-0" />
      <span>{text}</span>
    </div>
  )
}

function SelectControl({ id, label, value, onChange, options }) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-gray-400 mb-1.5">{label}</label>
      <select id={id} value={value} onChange={e => onChange(e.target.value)} className="input text-sm w-full">
        {options.map(([optionValue, optionLabel]) => <option key={optionValue} value={optionValue}>{optionLabel}</option>)}
      </select>
    </div>
  )
}

function NumberControl({ id, label, value, onChange, error, min, max, step = 1 }) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-gray-400 mb-1.5">{label}</label>
      <input id={id} type="number" min={min} max={max} step={step} value={value} onChange={e => onChange(e.target.value)} className="input text-sm w-full" />
      {error && <p className="text-xs text-red-300 mt-1">{error}</p>}
    </div>
  )
}

function OptionalNumberControl({ id, label, value, onChange, error, min }) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-gray-400 mb-1.5">{label}</label>
      <input id={id} type="number" min={min} placeholder="auto" value={value} onChange={e => onChange(e.target.value)} className="input text-sm w-full" />
      {error && <p className="text-xs text-red-300 mt-1">{error}</p>}
    </div>
  )
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="flex items-center justify-between gap-3 text-sm text-gray-400">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} className="rounded accent-brand-500" />
    </label>
  )
}

function ClusterButton({ mode, loading, disabled, onClick }) {
  const title = disabled ? MISSING_ENRICHMENT_MESSAGE : `Run ${mode} clustering`
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={loading || disabled}
      className={mode === 'hybrid' ? 'btn-primary flex items-center justify-center gap-2' : 'btn-secondary flex items-center justify-center gap-2'}
    >
      {loading ? <Loader size={15} className="animate-spin" /> : mode === 'base' ? <Play size={15} /> : <GitCompare size={15} />}
      Run {mode.charAt(0).toUpperCase() + mode.slice(1)} Clustering
    </button>
  )
}

function Metric({ label, value }) {
  return (
    <div className="bg-gray-950/50 border border-gray-800 rounded-lg p-3">
      <div className="text-lg font-semibold text-white">{typeof value === 'number' ? value.toFixed(3) : value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  )
}
