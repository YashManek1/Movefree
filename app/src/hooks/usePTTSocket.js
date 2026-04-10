import { useRef, useEffect, useCallback, useState } from 'react';

const PTT_WS_URL = process.env.EXPO_PUBLIC_PTT_WS_URL || 'ws://192.168.1.100:8765';

export function usePTTSocket(role, callbacks = {}) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const reconnectTimer = useRef(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(PTT_WS_URL);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        ws.send(JSON.stringify({ type: 'register', role }));
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;

        if (event.data instanceof ArrayBuffer) {
          callbacks.onBinaryChunk?.(event.data);
          return;
        }

        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'registered') {
            setConnected(true);
            callbacks.onConnect?.();
          } else if (msg.type === 'ptt_start') {
            callbacks.onPTTStart?.(msg.caretaker || 'Caretaker');
          } else if (msg.type === 'ptt_stop') {
            callbacks.onPTTStop?.();
          }
        } catch {}
      };

      ws.onerror = () => {};

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnected(false);
        callbacks.onDisconnect?.();

        reconnectTimer.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, 3000);
      };
    } catch {}
  }, [role]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendBinary = useCallback((buffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(buffer);
    }
  }, []);

  const sendJSON = useCallback((obj) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  }, []);

  return { connected, sendBinary, sendJSON };
}
