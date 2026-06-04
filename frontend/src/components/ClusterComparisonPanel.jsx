import { getClusterColor } from '../utils/colors.js'

export default function ClusterComparisonPanel({ suggestion, clusterInfo = {} }) {
  if (!suggestion) return null

  const isMerge = suggestion.suggestion_type === 'merge'

  if (isMerge) {
    const labelA = suggestion.cluster_a_label || `Cluster ${suggestion.cluster_a_id}`
    const labelB = suggestion.cluster_b_label || `Cluster ${suggestion.cluster_b_id}`
    const infoA = clusterInfo[suggestion.cluster_a_id] || {}
    const infoB = clusterInfo[suggestion.cluster_b_id] || {}
    const colorA = getClusterColor(suggestion.cluster_a_id)
    const colorB = getClusterColor(suggestion.cluster_b_id)

    return (
      <div className="grid grid-cols-2 gap-3 mt-3">
        <ClusterColumn
          label={labelA}
          color={colorA}
          size={infoA.size}
          keywords={infoA.keywords || []}
          tag="A"
        />
        <ClusterColumn
          label={labelB}
          color={colorB}
          size={infoB.size}
          keywords={infoB.keywords || []}
          tag="B"
        />
      </div>
    )
  }

  // Split view
  const label = suggestion.cluster_label || `Cluster ${suggestion.cluster_id}`
  const sizes = suggestion.sub_cluster_sizes || []
  const color = getClusterColor(suggestion.cluster_id)

  return (
    <div className="mt-3">
      <div className="text-xs text-gray-500 mb-2">Proposed split of "{label}"</div>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-xs font-medium text-gray-300">Sub-group A</span>
          </div>
          <div className="text-lg font-bold text-white">{sizes[0] ?? '?'}</div>
          <div className="text-xs text-gray-500">requirements</div>
        </div>
        <div className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2.5 h-2.5 rounded-full bg-brand-500" />
            <span className="text-xs font-medium text-gray-300">Sub-group B</span>
          </div>
          <div className="text-lg font-bold text-white">{sizes[1] ?? '?'}</div>
          <div className="text-xs text-gray-500">requirements</div>
        </div>
      </div>
    </div>
  )
}

function ClusterColumn({ label, color, size, keywords, tag }) {
  return (
    <div className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/50">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-xs font-medium text-gray-400">Cluster {tag}</span>
      </div>
      <div className="text-sm font-medium text-gray-200 truncate mb-1">{label}</div>
      {size != null && (
        <div className="text-xs text-gray-500 mb-2">{size} requirements</div>
      )}
      {keywords.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {keywords.slice(0, 4).map(kw => (
            <span key={kw} className="text-xs bg-gray-700/60 text-gray-400 px-1.5 py-0.5 rounded">
              {kw}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
