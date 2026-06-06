import { useState, useEffect } from 'react'

export default function MoveToClusterModal({
  isOpen,
  onClose,
  requirement,
  clusters,
  onSubmit,
}) {
  const [targetClusterId, setTargetClusterId] = useState('')
  const [confidence, setConfidence] = useState(100)
  const [comments, setComments] = useState('')
  const [analystName, setAnalystName] = useState('')

  useEffect(() => {
    if (isOpen && requirement) {
      setTargetClusterId(requirement.cluster_id !== null ? String(requirement.cluster_id) : '-1')
      setConfidence(100)
      setComments('')
      
      const cachedName = localStorage.getItem('analystName') || ''
      setAnalystName(cachedName)
    }
  }, [isOpen, requirement])

  if (!isOpen || !requirement) return null

  const handleSubmit = (e) => {
    e.preventDefault()
    
    // Save analyst name
    localStorage.setItem('analystName', analystName)

    const newClusterVal = targetClusterId === '-1' ? -1 : parseInt(targetClusterId, 10)
    onSubmit({
      new_cluster_id: newClusterVal,
      confidence_score: confidence / 100,
      comments: comments.trim(),
      applied_by: analystName.trim() || 'Expert Analyst',
    })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg overflow-hidden border bg-slate-900 border-slate-700/80 rounded-xl shadow-2xl animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="px-6 py-4 border-b bg-slate-800/40 border-slate-700/80">
          <h3 className="text-lg font-semibold text-slate-100">
            Reassign Cluster - {requirement.req_id || `REQ-${requirement.id}`}
          </h3>
          <p className="mt-1 text-xs text-slate-400 truncate">
            Current: {requirement.cluster_id !== null ? `Cluster ${requirement.cluster_id}` : 'Noise / Unclustered'}
          </p>
        </div>

        {/* Content Form */}
        <form onSubmit={handleSubmit}>
          <div className="p-6 space-y-5">
            
            {/* Req Preview */}
            <div className="p-3 text-sm rounded bg-slate-950/60 text-slate-300 border border-slate-800 max-h-24 overflow-y-auto italic">
              "{requirement.text}"
            </div>

            {/* Target Cluster */}
            <div className="space-y-2">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Target Cluster
              </label>
              <select
                value={targetClusterId}
                onChange={(e) => setTargetClusterId(e.target.value)}
                required
                className="w-full px-3 py-2 text-sm rounded border bg-slate-950 border-slate-800 text-slate-200 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors"
              >
                <option value="-1">Noise / Unclustered (Remove from clusters)</option>
                {clusters
                  .filter((c) => c.cluster_id !== requirement.cluster_id)
                  .map((c) => (
                    <option key={c.cluster_id} value={String(c.cluster_id)}>
                      {c.label} (Size: {c.size})
                    </option>
                  ))}
              </select>
            </div>

            {/* Confidence Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Confidence Rating
                </label>
                <span className="text-sm font-medium text-brand-400">{confidence}%</span>
              </div>
              <input
                type="range"
                min="50"
                max="100"
                value={confidence}
                onChange={(e) => setConfidence(parseInt(e.target.value, 10))}
                className="w-full h-1.5 rounded-lg bg-slate-800 accent-brand-500 cursor-pointer focus:outline-none"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>50% (Uncertain)</span>
                <span>100% (Certain)</span>
              </div>
            </div>

            {/* Analyst Name */}
            <div className="space-y-2">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Analyst Name
              </label>
              <input
                type="text"
                placeholder="e.g. Jane Doe"
                value={analystName}
                onChange={(e) => setAnalystName(e.target.value)}
                required
                className="w-full px-3 py-2 text-sm rounded border bg-slate-950 border-slate-800 text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors"
              />
            </div>

            {/* Comment */}
            <div className="space-y-2">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Correction Comments
              </label>
              <textarea
                rows="3"
                placeholder="Explain the rationale for this change..."
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                className="w-full px-3 py-2 text-sm rounded border bg-slate-950 border-slate-800 text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors resize-none"
              />
            </div>

          </div>

          {/* Footer Buttons */}
          <div className="px-6 py-4 flex justify-end gap-3 border-t bg-slate-800/20 border-slate-700/80">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm rounded font-medium border border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-slate-100 transition-colors focus:outline-none"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm rounded font-medium bg-brand-600 hover:bg-brand-500 text-white transition-colors shadow-lg focus:outline-none"
            >
              Reassign Cluster
            </button>
          </div>
        </form>

      </div>
    </div>
  )
}
