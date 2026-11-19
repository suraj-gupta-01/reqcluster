import { useCallback, useEffect, useRef, useState } from 'react'
import { getJob, cancelJob } from '../api/endpoints.js'

const TERMINAL = ['succeeded', 'failed', 'cancelled']

export function useJob(jobId, onDone) {
  const [job, setJob] = useState(null)
  const onDoneRef = useRef(onDone)
  useEffect(() => { onDoneRef.current = onDone })

  useEffect(() => {
    setJob(null)
    if (jobId == null) return
    const ctrl = new AbortController()
    const started = Date.now()
    let timer
    const tick = async () => {
      try {
        const j = await getJob(jobId, { signal: ctrl.signal })
        setJob(j)
        if (TERMINAL.includes(j.status)) {
          onDoneRef.current?.(j)
          return
        }
      } catch (e) {
        if (ctrl.signal.aborted || e?.name === 'AbortError') return
      }
      timer = setTimeout(tick, Date.now() - started > 30000 ? 3000 : 1000)
    }
    tick()
    return () => { ctrl.abort(); clearTimeout(timer) }
  }, [jobId])

  const cancel = useCallback(() => (jobId == null ? Promise.resolve() : cancelJob(jobId)), [jobId])
  return { job, cancel }
}
