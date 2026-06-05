// Deterministic, purple-free palette for up to 20 clusters.
const CLUSTER_COLORS = [
  '#2fbcaa', // teal (brand)
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#0ea5e9', // sky
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#ec4899', // pink
  '#14b8a6', // deep teal
  '#3b82f6', // blue
  '#d97706', // ochre
  '#22c55e', // green
  '#e11d48', // rose
  '#eab308', // gold
  '#38bdf8', // light sky
  '#16a34a', // forest
  '#dc2626', // crimson
  '#0d9488', // pine
  '#f43f5e', // raspberry
]

export const NOISE_COLOR = '#6b7280' // gray for noise

export const getClusterColor = (clusterId) => {
  if (clusterId === -1 || clusterId === null || clusterId === undefined) return NOISE_COLOR
  return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length]
}