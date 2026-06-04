import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Upload, LayoutDashboard, Network, Database, Activity, Layers, Sparkles, Wrench, Inbox } from 'lucide-react'

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

function Sidebar({ sessionId, sessionStatus, pendingCount }) {
  const done = sessionStatus === 'done'
  const navItems = [
    { to: '/', icon: Upload, label: 'Upload', exact: true, disabled: false },
    { to: '/enrichment', icon: Sparkles, label: 'Enrichment', disabled: false },
    { to: '/refinement', icon: Wrench, label: 'Refinement', disabled: false },
    { to: `/overview/${sessionId}`, icon: LayoutDashboard, label: 'Overview', disabled: !sessionId },
    { to: `/scatter/${sessionId}`, icon: Activity, label: 'Scatter Plot', disabled: !sessionId || !done },
    { to: `/graph/${sessionId}`, icon: Network, label: 'Similarity Graph', disabled: !sessionId || !done },
    { to: `/requirements/${sessionId}`, icon: Database, label: 'Requirements', disabled: !sessionId || !done },
    { to: `/review-queue/${sessionId}`, icon: Inbox, label: 'Review Queue', disabled: !sessionId || !done, badge: pendingCount },
  ]
  return (
    <aside className="w-56 min-h-screen bg-gray-950 border-r border-gray-800 flex flex-col flex-shrink-0">
      <div className="px-5 py-5 border-b border-gray-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center">
            <Layers size={16} className="text-white" />
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-tight">ReqCluster</div>
            <div className="text-xs text-gray-500">Phase 4 Workflow</div>
          </div>
        </div>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map(({ to, icon: Icon, label, disabled, exact, badge }) =>
          disabled ? (
            <div key={label} className="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-600 cursor-not-allowed text-sm select-none">
              <Icon size={16} /><span>{label}</span>
            </div>
          ) : (
            <NavLink key={label} to={to} end={exact}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${isActive ? 'bg-brand-600/20 text-brand-400 font-medium' : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'}`
              }>
              <Icon size={16} /><span>{label}</span>
              {badge > 0 && (
                <span className="badge bg-brand-600 text-white text-[10px] py-0 px-1.5 ml-auto">
                  {badge}
                </span>
              )}
            </NavLink>
          )
        )}
      </nav>
      <div className="px-5 py-4 border-t border-gray-800">
        <div className="text-xs text-gray-600">
          <div className="font-medium text-gray-500 mb-0.5">SBERT · UMAP · HDBSCAN</div>
          <div>c-TF-IDF Labeling</div>
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
