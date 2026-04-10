const WebSocket = require('ws');
const os = require('os');

const PORT = process.env.PTT_PORT || 8765;

const wss = new WebSocket.Server({ port: PORT });

const clients = new Map(); 
const ROLES = { CARETAKER: 'caretaker', BLIND: 'blind' };

function getLocalIP() {
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) return net.address;
    }
  }
  return '127.0.0.1';
}

wss.on('listening', () => {
  console.log(`\n╔════════════════════════════════════╗`);
  console.log(`║   MooveFree PTT Relay Server       ║`);
  console.log(`╠════════════════════════════════════╣`);
  console.log(`║  LAN: ws://${getLocalIP()}:${PORT}`.padEnd(43) + `║`);
  console.log(`║  Local: ws://127.0.0.1:${PORT}`.padEnd(43) + `║`);
  console.log(`╚════════════════════════════════════╝\n`);
});

wss.on('connection', (ws, req) => {
  let role = null;
  let clientId = req.socket.remoteAddress + ':' + req.socket.remotePort;

  ws.on('message', (data) => {

    if (typeof data === 'string') {
      try {
        const msg = JSON.parse(data);

        if (msg.type === 'register') {
          role = msg.role;
          if (!clients.has(role)) clients.set(role, new Set());
          clients.get(role).add(ws);
          console.log(`[+] ${role.toUpperCase()} connected (${clientId}) — total: ${clientCount()}`);
          ws.send(JSON.stringify({ type: 'registered', role, timestamp: Date.now() }));
          return;
        }

        if (msg.type === 'ptt_start' && role === ROLES.CARETAKER) {
          broadcast(ROLES.BLIND, JSON.stringify({ type: 'ptt_start', caretaker: msg.name || 'Caretaker' }));
          return;
        }

        if (msg.type === 'ptt_stop' && role === ROLES.CARETAKER) {
          broadcast(ROLES.BLIND, JSON.stringify({ type: 'ptt_stop' }));
          return;
        }
      } catch {

      }
      return;
    }

    if (role === ROLES.CARETAKER) {
      broadcast(ROLES.BLIND, data);
    }
  });

  ws.on('close', () => {
    if (role && clients.has(role)) {
      clients.get(role).delete(ws);
      console.log(`[-] ${role?.toUpperCase() || 'UNKNOWN'} disconnected (${clientId}) — total: ${clientCount()}`);
    }
  });

  ws.on('error', (err) => {
    console.error(`[!] Socket error (${clientId}):`, err.message);
  });
});

function broadcast(targetRole, data) {
  const targets = clients.get(targetRole);
  if (!targets) return;
  for (const sock of targets) {
    if (sock.readyState === WebSocket.OPEN) {
      sock.send(data);
    }
  }
}

function clientCount() {
  let count = 0;
  for (const s of clients.values()) count += s.size;
  return count;
}
