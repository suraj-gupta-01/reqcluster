import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  Upload, LayoutDashboard, Network, Database, Activity, Sparkles,
  Wrench, Inbox, Workflow, Brain, Download,
} from 'lucide-react'

import { getSession, getFeedbackQueue } from './utils/api.js'
import UploadPage from './pages/UploadPage.jsx'
import OverviewPage from './pages/OverviewPage.jsx'
import ScatterPage from './pages/ScatterPage.jsx'
import ClusterDetailPage from './pages/ClusterDetailPage.jsx'
import GraphPage from './pages/GraphPage.jsx'
import RequirementsPage from './pages/RequirementsPage.jsx'
import EnrichmentPage from './pages/EnrichmentPage.jsx'
import RefinementPage from './pages/RefinementPage.jsx'
import ReviewQueuePage from './pages/ReviewQueuePage.jsx'
import DependencyTreePage from './pages/DependencyTreePage.jsx'
import ActiveLearningPage from './pages/ActiveLearningPage.jsx'
import ExportPage from './pages/ExportPage.jsx'

/* Custom node-graph mark - represents clustering, not a generic boxed logo. */
function GlyphMark() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden="true">
      <line x1="6.5" y1="7" x2="13" y2="13" stroke="#2fbcaa" strokeWidth="1.3" strokeOpacity="0.55" />
      <line x1="13" y1="13" x2="20" y2="6.5" stroke="#2fbcaa" strokeWidth="1.3" strokeOpacity="0.55" />
      <line x1="13" y1="13" x2="8" y2="20" stroke="#2fbcaa" strokeWidth="1.3" strokeOpacity="0.55" />
      <line x1="13" y1="13" x2="20" y2="19" stroke="#2fbcaa" strokeWidth="1.3" strokeOpacity="0.55" />
      <circle cx="6.5" cy="7" r="2.4" fill="#5fd6c3" />
      <circle cx="20" cy="6.5" r="2.1" fill="#0d8175" />
      <circle cx="8" cy="20" r="2.1" fill="#0d8175" />
      <circle cx="20" cy="19" r="2.4" fill="#5fd6c3" />
      <circle cx="13" cy="13" r="3.2" fill="#14a08f" stroke="#0a0e0d" strokeWidth="1.2" />
    </svg>
  )
}

function StatusChip({ status }) {
  const map = {
    done: { label: 'Ready', dot: '#2fbcaa', ring: 'rgba(47,188,170,0.4)', pulse: false },
    processing: { label: 'Processing', dot: '#f4b860', ring: 'rgba(244,184,96,0.4)', pulse: true },
    uploaded: { label: 'Uploaded', dot: '#f4b860', ring: 'rgba(244,184,96,0.4)', pulse: false },
  }
  const s = map[status] || { label: 'No session', dot: '#5b6b68', ring: 'rgba(91,107,104,0.3)', pulse: false }
  return (
    <div className="flex items-center gap-2 px-2.5 py-1 rounded-full text-[11px] font-medium text-gray-300"
         style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${s.ring}` }}>
      <span className="relative flex w-2 h-2">
        {s.pulse && (
          <span className="absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping"
                style={{ background: s.dot }} />
        )}
        <span className="relative inline-flex rounded-full w-2 h-2" style={{ background: s.dot }} />
      </span>
      {s.label}
    </div>
  )
}

function CommandBar({ status }) {
  return (
    <header className="h-14 flex-shrink-0 flex items-center justify-between pl-4 pr-5
                       border-b border-brand-500/10 bg-[#0a0e0d]/70 backdrop-blur-xl">
      <div className="flex items-center gap-3">
        <GlyphMark />
        <div className="leading-none">
          <div className="text-[15px] font-bold text-white tracking-tight">ReqCluster</div>
          <div className="text-[9px] font-mono uppercase tracking-[0.26em] text-brand-400/70 mt-1">
            Requirements Intelligence
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <span className="hidden md:block font-mono text-[10px] tracking-wider text-gray-600">
          SBERT · UMAP · HDBSCAN
        </span>
        <StatusChip status={status} />
      </div>
    </header>
  )
}

function SectionLabel({ index, children }) {
  return (
    <div className="px-5 pt-5 pb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-500">
      <span className="font-mono text-brand-500">{index}</span>
      {children}
    </div>
  )
}

/* A stage on the pipeline timeline (node sits on the connecting line). */
function StageNode({ item }) {
  if (item.disabled) {
    return (
      <div className="relative flex items-center gap-3 pl-[34px] pr-3 py-2 text-gray-700 cursor-not-allowed select-none">
        <span className="absolute left-[21px] top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-[#0a0e0d] ring-2 ring-gray-700/50" />
        <item.icon size={15} />
        <span className="text-[13px]">{item.label}</span>
      </div>
    )
  }
  return (
    <NavLink
      to={item.to}
      end={item.exact}
      className={({ isActive }) =>
        `relative flex items-center gap-3 pl-[34px] pr-3 py-2 rounded-lg text-[13px] transition-all duration-200 ${
          isActive
            ? 'text-brand-200 font-semibold bg-brand-500/[0.08]'
            : 'text-gray-400 hover:text-gray-100 hover:bg-white/[0.03]'
        }`
      }
    >
      {({ isActive }) => (
        <>
          <span
            className="absolute left-[21px] top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full transition-all duration-200"
            style={
              isActive
                ? { background: '#2fbcaa', boxShadow: '0 0 0 3px rgba(47,188,170,0.18), 0 0 12px 1px rgba(47,188,170,0.55)' }
                : { background: '#0a0e0d', boxShadow: 'inset 0 0 0 2px rgba(47,188,170,0.45)' }
            }
          />
          <item.icon size={15} />
          <span>{item.label}</span>
          {item.badge > 0 && (
            <span className="ml-auto text-[10px] font-semibold py-0.5 px-1.5 rounded-full text-[#0a0e0d] bg-signal-400">
              {item.badge}
            </span>
          )}
        </>
      )}
    </NavLink>
  )
}

/* An Intelligence tool (icon tile, not a timeline node). */
function ToolLink({ item }) {
  if (item.disabled) {
    return (
      <div className="flex items-center gap-3 px-2.5 py-1.5 text-gray-700 cursor-not-allowed select-none">
        <span className="w-7 h-7 rounded-lg flex items-center justify-center border border-white/[0.05]">
          <item.icon size={14} />
        </span>
        <span className="text-[13px]">{item.label}</span>
      </div>
    )
  }
  return (
    <NavLink
      to={item.to}
      className={({ isActive }) =>
        `flex items-center gap-3 px-2.5 py-1.5 rounded-lg text-[13px] transition-all duration-200 ${
          isActive ? 'text-brand-200 font-semibold' : 'text-gray-400 hover:text-gray-100'
        }`
      }
    >
      {({ isActive }) => (
        <>
          <span
            className="w-7 h-7 rounded-lg flex items-center justify-center border transition-all duration-200"
            style={
              isActive
                ? { background: 'rgba(47,188,170,0.14)', borderColor: 'rgba(47,188,170,0.4)', color: '#5fd6c3' }
                : { background: 'rgba(255,255,255,0.02)', borderColor: 'rgba(255,255,255,0.06)' }
            }
          >
            <item.icon size={14} />
          </span>
          <span>{item.label}</span>
          {item.badge > 0 && (
            <span className="ml-auto text-[10px] font-semibold py-0.5 px-1.5 rounded-full text-[#0a0e0d] bg-signal-400">
              {item.badge}
            </span>
          )}
        </>
      )}
    </NavLink>
  )
}

function PipelineRail({ sessionId, sessionStatus, pendingCount }) {
  const done = sessionStatus === 'done'
  const pipeline = [
    { to: '/', icon: Upload, label: 'Upload', exact: true, disabled: false },
    { to: `/overview/${sessionId}`, icon: LayoutDashboard, label: 'Overview', disabled: !sessionId },
    { to: `/scatter/${sessionId}`, icon: Activity, label: 'Scatter Plot', disabled: !sessionId || !done },
    { to: `/graph/${sessionId}`, icon: Network, label: 'Similarity Graph', disabled: !sessionId || !done },
    { to: '/dependencies', icon: Workflow, label: 'Dependency Tree', disabled: false },
    { to: `/requirements/${sessionId}`, icon: Database, label: 'Requirements', disabled: !sessionId || !done },
  ]
  const intelligence = [
    { to: '/enrichment', icon: Sparkles, label: 'Enrichment', disabled: false },
    { to: '/refinement', icon: Wrench, label: 'Refinement', disabled: false },
    { to: `/review-queue/${sessionId}`, icon: Inbox, label: 'Review Queue', disabled: !sessionId || !done, badge: pendingCount },
    { to: '/active-learning', icon: Brain, label: 'Active Learning', disabled: false },
    { to: '/export', icon: Download, label: 'Export', disabled: false },
  ]

  return (
    <aside className="w-[244px] flex-shrink-0 flex flex-col overflow-y-auto
                     border-r border-brand-500/10 bg-[#0b100f]/60 backdrop-blur-xl">
      <SectionLabel index="01">Pipeline</SectionLabel>
      <div className="relative px-3">
        {/* connecting line behind the stage nodes */}
        <span className="absolute left-[26px] top-3 bottom-3 w-px bg-gradient-to-b from-brand-500/50 via-brand-500/15 to-transparent" />
        {pipeline.map((item) => <StageNode key={item.label} item={item} />)}
      </div>

      <SectionLabel index="02">Intelligence</SectionLabel>
      <div className="px-3 space-y-0.5">
        {intelligence.map((item) => <ToolLink key={item.label} item={item} />)}
      </div>

      <div className="mt-auto px-5 py-4 border-t border-brand-500/10">
        <div className="font-mono text-[10px] leading-relaxed text-gray-600">
          <span className="text-gray-500">c-TF-IDF</span> · ANN graph<br />
          MBSE · PDF export
        </div>
      </div>
    </aside>
  )
}

function AppContent() {
  const location = useLocation()
  const [uploadSessionId, setUploadSessionId] = useState(null)
  const [sessionStatus, setSessionStatus] = useState(null)
  const [pendingCount, setPendingCount] = useState(0)

  // Derive the active session from the URL so a page refresh keeps the rail
  // working; fall back to the id captured during the upload flow.
  const match = location.pathname.match(/\/(?:overview|scatter|graph|requirements|cluster|review-queue)\/(\d+)/)
  const sessionId = match ? parseInt(match[1]) : uploadSessionId
  const done = sessionStatus === 'done'

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    getSession(sessionId)
      .then(s => { if (!cancelled) setSessionStatus(s.status) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [sessionId])

  useEffect(() => {
    if (!sessionId || !done) {
      setPendingCount(0)
      return
    }
    let cancelled = false
    getFeedbackQueue(sessionId, 'pending')
      .then(items => { if (!cancelled) setPendingCount(items.length) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [sessionId, location.pathname, done])

  return (
    <div className="flex flex-col h-screen">
      <CommandBar status={sessionStatus} />
      <div className="flex flex-1 min-h-0">
        <PipelineRail sessionId={sessionId} sessionStatus={sessionStatus} pendingCount={pendingCount} />
        <main className="flex-1 overflow-auto min-w-0 app-main">
          <Routes>
            <Route path="/" element={<UploadPage onSessionCreated={(id, status) => { setUploadSessionId(id); setSessionStatus(status) }} />} />
            <Route path="/enrichment" element={<EnrichmentPage />} />
            <Route path="/refinement" element={<RefinementPage />} />
            <Route path="/dependencies" element={<DependencyTreePage />} />
            <Route path="/active-learning" element={<ActiveLearningPage />} />
            <Route path="/export" element={<ExportPage />} />
            <Route path="/overview/:sessionId" element={<OverviewPage onStatusChange={setSessionStatus} />} />
            <Route path="/scatter/:sessionId" element={<ScatterPage />} />
            <Route path="/cluster/:sessionId/:clusterId" element={<ClusterDetailPage />} />
            <Route path="/graph/:sessionId" element={<GraphPage />} />
            <Route path="/requirements/:sessionId" element={<RequirementsPage />} />
            <Route path="/review-queue/:sessionId" element={<ReviewQueuePage />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return <BrowserRouter><AppContent /></BrowserRouter>
}
