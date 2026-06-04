import { AlertTriangle, CheckCircle, Clock, Loader, XCircle } from 'lucide-react'

function statusStyle(status) {
  if (status === 'complete') return { icon: CheckCircle, label: 'Complete', color: 'text-emerald-400', border: 'border-emerald-900/40 bg-emerald-950/10' }
  if (status === 'partial') return { icon: AlertTriangle, label: 'Partial', color: 'text-amber-400', border: 'border-amber-900/40 bg-amber-950/10' }
  if (status === 'failed') return { icon: XCircle, label: 'Failed', color: 'text-red-400', border: 'border-red-900/40 bg-red-950/10' }
  return { icon: Clock, label: status || 'Not started', color: 'text-gray-400', border: 'border-gray-800' }
}

export default function EnrichmentStatusCard({ status, loading = false, error = null }) {
  if (error) {
    return (
      <div className="card p-4 border-red-900/50 bg-red-950/20">
        <div className="flex items-start gap-3 text-sm text-red-300">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      </div>
    )
  }

  if (!status && !loading) {
    return (
      <div className="card p-4">
        <div className="text-sm font-medium text-gray-300">Enrichment Status</div>
        <p className="text-sm text-gray-500 mt-1">No enrichment results yet. Start enrichment for this session.</p>
      </div>
    )
  }

  const style = statusStyle(status?.status)
  const Icon = loading ? Loader : style.icon
  const total = status?.total ?? 0
  const succeeded = status?.succeeded ?? 0
  const failed = status?.failed ?? 0
  const pending = status?.pending ?? 0
  const pct = total > 0 ? Math.round((succeeded / total) * 100) : 0

  return (
    <div className={`card p-4 ${style.border}`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Icon size={16} className={`${loading ? 'animate-spin text-brand-400' : style.color}`} />
            <span className={`text-sm font-semibold ${style.color}`}>
              {loading ? 'Working' : style.label}
            </span>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Session {status?.session_id ?? '-'} {status?.provider ? `- ${status.provider}` : ''}
            {status?.model ? ` / ${status.model}` : ''}
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-white">{pct}%</div>
          <div className="text-xs text-gray-500">complete</div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mt-4">
        {[
          ['Total', total, 'text-gray-200'],
          ['Succeeded', succeeded, 'text-emerald-400'],
          ['Failed', failed, 'text-red-400'],
          ['Pending', pending, 'text-amber-400'],
        ].map(([label, value, color]) => (
          <div key={label} className="bg-gray-950/50 border border-gray-800 rounded-lg p-3">
            <div className={`text-lg font-semibold ${color}`}>{value}</div>
            <div className="text-xs text-gray-500">{label}</div>
          </div>
        ))}
      </div>

      {status?.latest_run_created_at && (
        <div className="text-xs text-gray-500 mt-3">
          Latest run: {new Date(status.latest_run_created_at).toLocaleString()}
        </div>
      )}

      {status?.warnings?.length > 0 && (
        <div className="mt-3 space-y-1">
          {status.warnings.slice(0, 5).map((warning) => (
            <div key={warning} className="text-xs text-amber-300 flex items-start gap-1.5">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              <span>{warning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
