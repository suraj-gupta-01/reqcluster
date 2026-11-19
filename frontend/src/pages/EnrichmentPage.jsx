import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity,
  AlertCircle,
  CheckCircle,
  ChevronRight,
  GitCompare,
  Loader,
  Network,
  Play,
  RefreshCw,
  Sparkles,
} from 'lucide-react'

import { startEnrichment, getEnrichment, getEnrichmentSummary, getMetrics, startClustering } from '../api/endpoints.js'
import { useApi } from '../hooks/useApi.js'
import { useJob } from '../hooks/useJob.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage, conflictAs } from '../utils/format.js'
import AblationReportPanel from '../components/AblationReportPanel.jsx'
import DomainVocabularyTags from '../components/DomainVocabularyTags.jsx'
import EmbeddingComparisonPanel from '../components/EmbeddingComparisonPanel.jsx'
import EnrichmentResultsTable from '../components/EnrichmentResultsTable.jsx'
import EnrichmentStatusCard from '../components/EnrichmentStatusCard.jsx'
import JobProgress from '../components/JobProgress.jsx'

const MISSING_ENRICHMENT_MESSAGE = 'Run enrichment for this session before clustering with enriched or hybrid embeddings.'

function isPositiveIntegerOrEmpty(value, min = 1) {
  if (value === '' || value === null || value === undefined) return true
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed >= min
}

export default function EnrichmentPage() {
  const { sessionId, reloadSession } = useSession()
  if (!sessionId) {
    return (
      <div className="p-8 text-sm text-gray-400">
        <Link to="/" className="text-brand-400 hover:underline">Select or upload a session first</Link>
      </div>
    )
  }
  return <Enrichment sessionId={sessionId} reloadSession={reloadSession} />
}

function Enrichment({ sessionId, reloadSession }) {
  const [provider, setProvider] = useState('')
  const [clusterOptions, setClusterOptions] = useState({ clusterer: 'hdbscan', sim_threshold: 0.5, min_cluster_size: '', min_samples: '' })
  const [enrichJobId, setEnrichJobId] = useState(null)
  const [clusterJobId, setClusterJobId] = useState(null)
  const [clusterMode, setClusterMode] = useState('base')
  const [clusterResult, setClusterResult] = useState(null)
  const [error, setError] = useState(null)
  const [clusterError, setClusterError] = useState(null)
  const [notice, setNotice] = useState(null)

  const resultsQ = useApi((o) => getEnrichment(sessionId, o), [sessionId])
  // 409 until enrichment exists: show nothing
  const summaryQ = useApi((o) => getEnrichmentSummary(sessionId, o).catch(conflictAs(null)), [sessionId, resultsQ.data])
  const qualityReport = summaryQ.data?.quality_report
  const metricsQ = useApi((o) => getMetrics(sessionId, o).catch(conflictAs(null)), [sessionId, clusterResult])
  const results = useMemo(() => resultsQ.data || [], [resultsQ.data])

  const enrichJob = useJob(enrichJobId, (j) => {
    setEnrichJobId(null)
    if (j.status === 'succeeded') setNotice('Enrichment completed.')
    else setError(j.status === 'cancelled' ? 'Enrichment cancelled.' : j.error || 'Enrichment failed.')
    resultsQ.reload()
  })
  const clusterJob = useJob(clusterJobId, (j) => {
    setClusterJobId(null)
    if (j.status === 'succeeded') {
      setClusterResult({ ...(j.result || {}), embedding_mode: clusterMode })
      setNotice('Clustering completed.')
    } else {
      setClusterError(j.status === 'cancelled' ? 'Clustering cancelled.' : j.error || 'Clustering failed.')
    }
    reloadSession()
  })
  const enriching = enrichJobId != null
  const clustering = clusterJobId != null

  const status = useMemo(() => {
    if (!results.length) return null
    const succeeded = results.filter(r => r.status === 'succeeded').length
    return {
      session_id: sessionId,
      status: succeeded === results.length ? 'complete' : 'partial',
      total: results.length,
      succeeded,
      failed: results.length - succeeded,
      pending: 0,
      provider: results[0].provider,
      warnings: [...new Set(results.flatMap(r => r.warnings || []))],
    }
  }, [results, sessionId])
  const enrichmentReady = status?.status === 'complete'

  const vocabulary = useMemo(() => {
    if (summaryQ.data?.domain_vocabulary?.length) return summaryQ.data.domain_vocabulary
    const seen = new Set()
    const terms = []
    results.forEach(row => (row.domain_terms || []).forEach(term => {
      const key = String(term).toLowerCase()
      if (!seen.has(key)) { seen.add(key); terms.push(term) }
    }))
    return terms
  }, [results, summaryQ.data])

  const validation = {}
  if (clusterOptions.sim_threshold < 0 || clusterOptions.sim_threshold > 1) validation.sim_threshold = 'Similarity threshold must be between 0 and 1.'
  if (!isPositiveIntegerOrEmpty(clusterOptions.min_cluster_size, 2)) validation.min_cluster_size = 'Min cluster size must be 2 or greater, or empty.'
  if (!isPositiveIntegerOrEmpty(clusterOptions.min_samples, 1)) validation.min_samples = 'Min samples must be a positive integer, or empty.'
  const hasValidationErrors = Object.keys(validation).length > 0
  const updateClusterOption = (key, value) => setClusterOptions(prev => ({ ...prev, [key]: value }))

  const handleStartEnrichment = async () => {
    setError(null)
    setNotice(null)
    try {
      setEnrichJobId((await startEnrichment(sessionId, provider ? { provider } : {})).id)
    } catch (err) {
      setError(getErrorMessage(err, 'Enrichment failed.'))
    }
  }

  const handleCluster = async (mode) => {
    if (mode !== 'base' && !enrichmentReady) {
      setClusterError(MISSING_ENRICHMENT_MESSAGE)
      return
    }
    if (hasValidationErrors) return
    setClusterError(null)
    setClusterResult(null)
    setNotice(null)
    setClusterMode(mode)
    const payload = {
      embedding_mode: mode,
      clusterer: clusterOptions.clusterer,
      sim_threshold: Number(clusterOptions.sim_threshold),
      enable_embedding_comparison: !!clusterOptions.enable_embedding_comparison,
      run_ablation: !!clusterOptions.run_ablation,
    }
    if (clusterOptions.min_cluster_size !== '') payload.min_cluster_size = Number(clusterOptions.min_cluster_size)
    if (clusterOptions.min_samples !== '') payload.min_samples = Number(clusterOptions.min_samples)
    try {
      setClusterJobId((await startClustering(sessionId, payload)).id)
    } catch (err) {
      setClusterError(getErrorMessage(err, 'Clustering failed.'))
    }
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
            Local runs the real on-device model; Mock is an offline stand-in for testing only.
          </p>
        </div>
        <div className="flex gap-2">
          <Link to={`/s/${sessionId}/scatter`} className="btn-secondary text-sm flex items-center gap-2">
            <Activity size={14} />
            Scatter
          </Link>
          <Link to={`/s/${sessionId}/graph`} className="btn-secondary text-sm flex items-center gap-2">
            <Network size={14} />
            Graph
          </Link>
        </div>
      </div>

      {error && <Alert text={error} tone="red" />}
      {resultsQ.error && <Alert text={getErrorMessage(resultsQ.error, 'Failed to load enrichment results.')} tone="red" />}
      {summaryQ.error && <Alert text={getErrorMessage(summaryQ.error, 'Failed to load the enrichment summary.')} tone="red" />}
      {metricsQ.error && <Alert text={getErrorMessage(metricsQ.error, 'Failed to load clustering metrics.')} tone="red" />}
      {notice && <Alert text={notice} tone="emerald" icon={CheckCircle} />}

      <div className="grid lg:grid-cols-[380px_minmax(0,1fr)] gap-6">
        <div className="space-y-6">
          <section className="card p-4 space-y-4">
            <h2 className="text-sm font-semibold text-gray-200">Provider</h2>
            <SelectControl
              id="provider-name"
              label="Provider"
              value={provider}
              onChange={setProvider}
              options={[
                ['', 'Server default'],
                ['local', 'Local LLM (on-device)'],
                ['openai', 'Cloud LLM'],
                ['mock', 'Mock (testing only)'],
              ]}
            />
            <div className="flex gap-2 pt-2">
              <button type="button" onClick={handleStartEnrichment} disabled={enriching} className="btn-primary flex items-center gap-2">
                {enriching ? <Loader size={15} className="animate-spin" /> : <Sparkles size={15} />}
                Start Enrichment
              </button>
            </div>
            {enriching && <JobProgress job={enrichJob.job} onCancel={enrichJob.cancel} title="Enriching" />}
          </section>

          <EnrichmentStatusCard status={status} loading={resultsQ.loading || enriching} />

          <section className="card p-4 space-y-4">
            <h2 className="text-sm font-semibold text-gray-200">Clustering Controls</h2>
            <label className="block text-xs text-gray-400">
              Clustering method
              <select value={clusterOptions.clusterer} onChange={e => updateClusterOption('clusterer', e.target.value)} className="input text-sm w-full mt-1">
                <option value="hdbscan">HDBSCAN (finds outliers)</option>
                <option value="ward">Ward (every requirement assigned)</option>
              </select>
            </label>
            <div className="grid grid-cols-3 gap-3">
              <NumberControl
                id="similarity-threshold"
                label="Similarity"
                min={0}
                max={1}
                step={0.05}
                value={clusterOptions.sim_threshold}
                error={validation.sim_threshold}
                onChange={value => updateClusterOption('sim_threshold', Number(value))}
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

            {[['enable_embedding_comparison', 'Embedding comparison'], ['run_ablation', 'Ablation report']].map(([key, label]) => (
              <label key={key} className="flex items-center gap-2 text-xs text-gray-400">
                <input type="checkbox" checked={!!clusterOptions[key]} onChange={e => updateClusterOption(key, e.target.checked)} />
                {label}
              </label>
            ))}

            {!enrichmentReady && (
              <p className="text-xs text-amber-300">{MISSING_ENRICHMENT_MESSAGE}</p>
            )}
            {clusterError && <Alert text={clusterError} tone="red" />}

            <div className="grid gap-2">
              <ClusterButton mode="base" loading={clustering} onClick={() => handleCluster('base')} />
              <ClusterButton mode="hybrid" loading={clustering} disabled={!enrichmentReady} onClick={() => handleCluster('hybrid')} />
              <ClusterButton mode="enriched" loading={clustering} disabled={!enrichmentReady} onClick={() => handleCluster('enriched')} />
            </div>

            {clustering && <JobProgress job={clusterJob.job} onCancel={clusterJob.cancel} title="Clustering" />}
          </section>
        </div>

        <div className="space-y-6 min-w-0">
          <DomainVocabularyTags terms={vocabulary} />

          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-200">Enrichment Results</h2>
            <button type="button" onClick={resultsQ.reload} className="btn-secondary text-sm flex items-center gap-2">
              {resultsQ.loading ? <Loader size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              Load Results
            </button>
          </div>
          <EnrichmentResultsTable results={results} />

          {qualityReport?.aggregate && (
            <div className="card p-4">
              <h2 className="text-sm font-semibold text-gray-200 mb-3">Quality Summary</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  ['Succeeded', qualityReport.succeeded ?? 0],
                  ['Failed', qualityReport.failed ?? 0],
                  ['Mean risk', qualityReport.aggregate.mean_hallucination_risk_score ?? '-'],
                  ['Mean confidence', qualityReport.aggregate.mean_adjusted_confidence_score ?? '-'],
                ].map(([label, value]) => (
                  <div key={label} className="bg-gray-950/50 border border-gray-800 rounded-lg p-3">
                    <div className="text-lg font-semibold text-white">{value}</div>
                    <div className="text-xs text-gray-500">{label}</div>
                  </div>
                ))}
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
                    <p className="text-sm text-gray-400 mt-1">Using {clusterResult.embedding_mode} embeddings.</p>
                  </div>
                  <div className="flex gap-2">
                    <Link to={`/s/${sessionId}/scatter`} className="btn-primary text-sm flex items-center gap-2">
                      Scatter
                      <ChevronRight size={14} />
                    </Link>
                    <Link to={`/s/${sessionId}/graph`} className="btn-secondary text-sm">Graph</Link>
                  </div>
                </div>
              </div>
              <EmbeddingComparisonPanel report={metricsQ.data?.embedding_comparison} />
              <AblationReportPanel report={metricsQ.data?.ablation_report} />
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
