// Deterministic color palette for up to 20 clusters
const CLUSTER_COLORS = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#ec4899', // pink
  '#14b8a6', // teal
  '#6366f1', // indigo
  '#d97706', // yellow-dark
  '#22c55e', // green
  '#e11d48', // rose
  '#7c3aed', // purple
  '#0ea5e9', // sky
  '#16a34a', // green-dark
  '#dc2626', // red-dark
  '#2563eb', // blue-dark
  '#9333ea', // purple-alt
]

export const NOISE_COLOR = '#6b7280' // gray for noise

export const getClusterColor = (clusterId) => {
  if (clusterId === -1 || clusterId === null || clusterId === undefined) return NOISE_COLOR
  return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length]
}