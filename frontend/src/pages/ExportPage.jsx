import { useState, useEffect } from 'react'
import { Download, Loader, FileCode, FileSpreadsheet, Boxes, Network } from 'lucide-react'
import { getSessions, getExportUrl } from '../utils/api.js'

const FORMATS = [
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
]

export default function ExportPage() {
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getSessions()
      .then(list => {
        const done = list.filter(s => s.status === 'done')
        setSessions(done)
        if (done.length && !selected) setSelected(done[0].id)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-8 max-w-5xl animate-fade-up">
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

      <div className="card p-4 mb-6">
        <label className="text-xs text-gray-500 block mb-1.5">Session</label>
        {loading ? (
          <div className="flex items-center gap-2 text-gray-400 text-sm"><Loader size={14} className="animate-spin" /> Loading…</div>
        ) : sessions.length === 0 ? (
          <div className="text-sm text-gray-500">No completed sessions. Run clustering first.</div>
        ) : (
          <select value={selected || ''} onChange={e => setSelected(parseInt(e.target.value))} className="input text-sm w-full md:w-96">
            {sessions.map(s => <option key={s.id} value={s.id}>#{s.id} - {s.filename} ({s.total_requirements} reqs)</option>)}
          </select>
        )}
      </div>

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
            <a
              href={selected ? getExportUrl(selected, key) : undefined}
              onClick={e => { if (!selected) e.preventDefault() }}
              className={`btn-secondary text-sm justify-center ${!selected ? 'opacity-40 pointer-events-none' : ''}`}
              download
            >
              <Download size={14} /> Download {name}
            </a>
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
