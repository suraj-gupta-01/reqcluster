import { Star } from 'lucide-react'

export default function RepresentativesList({ reqIds = [], className = '' }) {
  if (!reqIds || reqIds.length === 0) return null

  return (
    <div className={className}>
      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
        Representative Requirements
      </div>
      <div className="space-y-1.5">
        {reqIds.map((id, i) => (
          <div
            key={id}
            className="flex items-center gap-2 text-sm text-gray-300 bg-gray-800/50 px-3 py-1.5 rounded-md"
          >
            <Star size={12} className={i === 0 ? 'text-amber-400' : 'text-gray-600'} />
            <span className="font-mono text-xs text-gray-400">{id}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
