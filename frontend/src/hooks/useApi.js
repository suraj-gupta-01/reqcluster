import { useCallback, useEffect, useState } from 'react'

export function useApi(fn, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const [tick, setTick] = useState(0)
  const reload = useCallback(() => setTick((t) => t + 1), [])

  useEffect(() => {
    const ctrl = new AbortController()
    setState((p) => ({ ...p, loading: true, error: null }))
    fn({ signal: ctrl.signal }).then(
      (data) => { if (!ctrl.signal.aborted) setState({ data, error: null, loading: false }) },
      (error) => {
        if (ctrl.signal.aborted || error?.name === 'AbortError') return
        setState((p) => ({ ...p, error, loading: false }))
      },
    )
    return () => ctrl.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])

  return { ...state, reload }
}
