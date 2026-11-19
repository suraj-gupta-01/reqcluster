import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Download, Loader, FileCode, FileSpreadsheet, Boxes, Network, FileText } from 'lucide-react'
import { download } from '../api/client.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage } from '../utils/format.js'

const FORMATS = [
  {
    key: 'pdf', icon: FileText, name: 'PDF report',
    desc: 'A print-ready clustering report: headline metrics, the cluster summary table, and per-cluster requirements. The shareable final output.',
    accent: '#2fbcaa',
  },
  {
    key: 'reqif', icon: FileCode, name: 'ReqIF 1.2',
    desc: 'OMG ReqIF XML with cluster-grouped spec hierarchy. Imports into IBM DOORS Next, ReqView, and Polarion.',
    accent: '#2fbcaa',
  },
  {
    key: 'sysml', icon: Boxes, name: 'SysML / UML XMI',
    desc: 'XMI 2.5.1 model: clusters as packages, requirements as «requirement» blocks, dependencies as «deriveReqt». Imports into Papyrus and MagicDraw.',
    accent: '#0ea5e9',
  },
  {
    key: 'jama', icon: Network, name: 'Jama Connect',
    desc: 'Jama REST item + relationship bundle ready for upload, or offline JSON import.',
    accent: '#f59e0b',
  },
  {
    key: 'csv', icon: FileSpreadsheet, name: 'CSV',
    desc: 'Flat requirements table with cluster assignments for spreadsheets and quick review.',
    accent: '#84cc16',
  },
  {
    key: 'feedback', icon: FileSpreadsheet, name: 'Feedback CSV',
    desc: 'Reviewer feedback and constraints recorded for this session.',
    accent: '#a78bfa',
  },
]

const EXT = { sysml: 'xmi', jama: 'json', reqif: 'reqif' }

export default function ExportPage() {
  const { sessionId } = useSession()
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)

  const handleDownload = async (key) => {
    setBusy(key)
    setError(null)
    try {
      if (key === 'feedback') {
        await download(`/sessions/${sessionId}/feedback/export`, `session-${sessionId}-feedback.csv`, { format: 'csv' })
      } else {
        await download(`/sessions/${sessionId}/export/${key}`, `session-${sessionId}.${EXT[key] ?? key}`)
      }
    } catch (err) {
      setError(getErrorMessage(err, 'Download failed.'))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="p-8 max-w-6xl animate-fade-up">
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Download size={20} className="text-brand-400" />
          <h1 className="text-2xl font-bold text-white tracking-tight">MBSE Export</h1>
        </div>
        <p className="text-gray-400 text-sm max-w-xl">
          Export the clustered requirements, their groupings, and dependency links to standard
          requirements-engineering and model-based formats.
        </p>
      </div>

      {!sessionId && (
        <div className="card p-4 mb-6 text-sm text-gray-500">
          <Link to="/" className="text-brand-400 hover:underline">Select or upload a session first</Link>
        </div>
      )}
      {error && <div className="card p-3 mb-4 text-sm text-amber-300/80">{error}</div>}

      <div className="grid sm:grid-cols-2 gap-4">
        {FORMATS.map(({ key, icon: Icon, name, desc, accent }) => (
          <div key={key} className="card p-5 flex flex-col gap-3 transition-transform duration-200 hover:-translate-y-0.5">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                   style={{ background: accent + '1f', border: `1px solid ${accent}40` }}>
                <Icon size={18} style={{ color: accent }} />
              </div>
              <h2 className="text-base font-semibold text-white">{name}</h2>
            </div>
            <p className="text-sm text-gray-400 leading-relaxed flex-1">{desc}</p>
            <button
              onClick={() => handleDownload(key)}
              disabled={!sessionId || busy === key}
              className="btn-secondary text-sm justify-center disabled:opacity-40"
            >
              {busy === key ? <Loader size={14} className="animate-spin" /> : <Download size={14} />} Download {name}
            </button>
          </div>
        ))}
      </div>

      <p className="text-xs text-gray-600 mt-6 leading-relaxed">
        Dependency links are included when a dependency tree has been generated for the session.
        Configure <span className="font-mono text-gray-500">JAMA_PROJECT_ID</span> and item-type ids to match a live Jama project.
      </p>
    </div>
  )
}
