import { Loader } from 'lucide-react'

// Progress bar + step + Cancel for a running job (see hooks/useJob.js).
export default function JobProgress({ job, onCancel, title = 'Running' }) {
  const pct = Math.round((job?.progress ?? 0) * 100)
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Loader size={14} className="animate-spin text-brand-400" />
          {title}{job?.step ? ` - ${job.step}` : ''}
        </span>
        <span className="text-sm font-mono text-brand-400">{pct}%</span>
      </div>
      <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden mb-3">
        <div className="h-full bg-brand-500 rounded-full transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      {job?.message && <div className="text-xs text-gray-500 mb-3 truncate">{job.message}</div>}
      <button onClick={onCancel} className="btn-secondary text-sm">Cancel</button>
    </div>
  )
}
