import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Loader } from 'lucide-react'
import { useSession } from '../state/SessionContext.jsx'
import { useJob } from '../hooks/useJob.js'
import { startClustering } from '../api/endpoints.js'
import { getErrorMessage } from '../utils/format.js'
import JobProgress from './JobProgress.jsx'

// Renders children only once the session is clustered; otherwise prompts, shows progress, or the job error.
export default function SessionGate({ children }) {
  const { sessionId, session, reloadSession } = useSession()
  const [startedId, setStartedId] = useState(null)
  const [err, setErr] = useState(null)
  const active = session && ['queued', 'running'].includes(session.status)
  const jobId = startedId ?? (active ? session.last_job?.id : null)
  const { job, cancel } = useJob(jobId, () => { setStartedId(null); reloadSession() })

  if (!sessionId) return (
    <div className="p-8 text-gray-400 text-sm">
      <Link to="/" className="text-brand-400 hover:underline">Select or upload a session first</Link>
    </div>
  )
  if (!session) return (
    <div className="flex items-center justify-center h-64"><Loader size={24} className="animate-spin text-brand-400" /></div>
  )
  if (jobId && !(session.revision > 0)) return <div className="p-8 max-w-xl"><JobProgress job={job} onCancel={cancel} title="Clustering" /></div>
  const shown = session.status === 'done' || session.revision > 0
  if (shown) {
    const lj = session.last_job
    const banner = session.status === 'error'
      ? `Last run ${lj?.status ?? 'failed'}${lj?.error ? `: ${lj.error}` : ''}; showing revision ${session.revision}`
      : active ? `Re-clustering… showing revision ${session.revision}${job ? ` (${Math.round((job.progress ?? 0) * 100)}%)` : ''}` : null
    return <>{banner && <div className="px-4 py-2 text-xs text-amber-300/90 bg-amber-900/20 border-b border-amber-900/30">{banner}</div>}{children}</>
  }

  const run = async () => {
    setErr(null)
    try { setStartedId((await startClustering(sessionId, {})).id); reloadSession() } catch (e) { setErr(getErrorMessage(e)) }
  }
  return (
    <div className="p-8 max-w-xl space-y-3">
      {session.status === 'error' && (
        <div className="text-red-400 text-sm">Clustering failed: {session.last_job?.error || 'unknown error'}</div>
      )}
      <p className="text-gray-400 text-sm">This session has not been clustered yet.</p>
      <button onClick={run} className="btn-primary">Run clustering</button>
      {err && <div className="text-red-400 text-sm">{err}</div>}
    </div>
  )
}
