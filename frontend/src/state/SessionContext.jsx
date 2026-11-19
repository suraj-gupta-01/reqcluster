import { createContext, useContext, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import { useApi } from '../hooks/useApi.js'
import { getSession } from '../api/endpoints.js'

const Ctx = createContext({ sessionId: null, session: null, reloadSession: () => {} })

export function SessionProvider({ children }) {
  const { sessionId } = useParams()
  const { data, reload } = useApi((o) => (sessionId ? getSession(sessionId, o) : Promise.resolve(null)), [sessionId])
  const value = useMemo(() => ({ sessionId, session: data, reloadSession: reload }), [sessionId, data, reload])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export const useSession = () => useContext(Ctx)
