import { useState, useCallback, useRef, useEffect } from 'react'

export interface SSEOptions {
  onMessage?: (event: MessageEvent) => void
  onError?: (error: Event) => void
  onOpen?: () => void
}

export function useSSE(url: string, options?: SSEOptions) {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'error' | 'closed'>('idle')
  const [events, setEvents] = useState<any[]>([])
  const eventSourceRef = useRef<EventSource | null>(null)

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    setStatus('connecting')
    setEvents([])

    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onopen = () => {
      setStatus('connected')
      options?.onOpen?.()
    }

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setEvents((prev) => [...prev, data])
        options?.onMessage?.(event)
      } catch (error) {
        console.error('Failed to parse SSE message:', error)
      }
    }

    eventSource.onerror = (error) => {
      setStatus('error')
      options?.onError?.(error)
      eventSource.close()
    }

    return eventSource
  }, [url, options])

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
      setStatus('closed')
    }
  }, [])

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    status,
    events,
    connect,
    disconnect,
  }
}
