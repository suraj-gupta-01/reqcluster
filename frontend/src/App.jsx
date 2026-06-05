import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Upload, LayoutDashboard, Network, Database, Activity, Layers, Sparkles, Wrench, Inbox, Workflow, Brain, Download } from 'lucide-react'

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

function Sidebar({ sessionId, sessionStatus, pendingCount }) {
  const done = sessionStatus === 'done'
  const navItems = [
    { section: 'Workspace' },
    { to: '/', icon: Upload, label: 'Upload', exact: true, disabled: false },
    { to: `/overview/${sessionId}`, icon: LayoutDashboard, label: 'Overview', disabled: !sessionId },
    { to: `/scatter/${sessionId}`, icon: Activity, label: 'Scatter Plot', disabled: !sessionId || !done },
    { to: `/graph/${sessionId}`, icon: Network, label: 'Similarity Graph', disabled: !sessionId || !done },
    { to: `/dependencies`, icon: Workflow, label: 'Dependency Tree', disabled: false },
    { to: `/requirements/${sessionId}`, icon: Database, label: 'Requirements', disabled: !sessionId || !done },
    { section: 'Intelligence' },
    { to: '/enrichment', icon: Sparkles, label: 'Enrichment', disabled: false },
    { to: '/refinement', icon: Wrench, label: 'Refinement', disabled: false },
    { to: `/review-queue/${sessionId}`, icon: Inbox, label: 'Review Queue', disabled: !sessionId || !done, badge: pendingCount },
    { to: '/active-learning', icon: Brain, label: 'Active Learning', disabled: false },
    { to: '/export', icon: Download, label: 'Export', disabled: false },
  ]
  return (
    <aside className="w-60 min-h-screen flex flex-col flex-shrink-0 border-r border-white/[0.05] bg-black/20 backdrop-blur-xl">
      <div className="px-5 py-5 border-b border-white/[0.05]">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
               style={{ background: 'linear-gradient(180deg,#16b0a0,#0d8175)', boxShadow: '0 6px 18px -6px rgba(13,129,117,0.8)' }}>
            <Layers size={17} className="text-white" />
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-tight">ReqCluster</div>
            <div className="text-[11px] text-gray-500">Requirements Intelligence</div>
          </div>
        </div>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map((item, i) =>
          item.section ? (
            <div key={`s-${i}`} className="px-3 pt-4 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-gray-600">
              {item.section}
            </div>
          ) : item.disabled ? (
            <div key={item.label} className="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-700 cursor-not-allowed text-sm select-none">
              <item.icon size={16} /><span>{item.label}</span>
            </div>
          ) : (
            <NavLink key={item.label} to={item.to} end={item.exact}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-200 ${isActive ? 'bg-brand-500/15 text-brand-300 font-semibold shadow-[inset_0_1px_0_0_rgba(255,255,255,0.05)]' : 'text-gray-400 hover:text-gray-100 hover:bg-white/[0.04]'}`
              }>
              <item.icon size={16} /><span>{item.label}</span>
              {item.badge > 0 && (
                <span className="ml-auto text-[10px] font-semibold py-0.5 px-1.5 rounded-full text-white"
                      style={{ background: '#0d8175' }}>
                  {item.badge}
                </span>
              )}
            </NavLink>
          )
        )}
      </nav>
      <div className="px-5 py-4 border-t border-white/[0.05]">
        <div className="text-[11px] text-gray-600 leading-relaxed">
          <div className="font-medium text-gray-500 mb-0.5">SBERT · UMAP · HDBSCAN</div>
          <div>c-TF-IDF · Dependency Graph</div>
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

  // Derive the active session from the URL so a page refresh keeps the sidebar
  // working; fall back to the id captured during the upload flow.
  const match = location.pathname.match(/\/(?:overview|scatter|graph|requirements|cluster|review-queue)\/(\d+)/)
  const sessionId = match ? parseInt(match[1]) : uploadSessionId
  const done = sessionStatus === 'done'

  // Keep the sidebar's status in sync when navigating directly to a session URL.
  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    getSession(sessionId)
      .then(s => { if (!cancelled) setSessionStatus(s.status) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [sessionId])

  // Fetch pending feedback corrections count
  useEffect(() => {
    if (!sessionId || !done) {
      setPendingCount(0)
      return
    }
    let cancelled = false
    getFeedbackQueue(sessionId, 'pending')
      .then(items => {
        if (!cancelled) setPendingCount(items.length)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [sessionId, location.pathname, done])

  return (
    <div className="flex min-h-screen">
      <Sidebar sessionId={sessionId} sessionStatus={sessionStatus} pendingCount={pendingCount} />
      <main className="flex-1 overflow-auto min-w-0">
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
  )
}

export default function App() {
  return <BrowserRouter><AppContent /></BrowserRouter>
}
