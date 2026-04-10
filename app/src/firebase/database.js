import {
  ref,
  set,
  push,
  get,
  onValue,
  off,
  serverTimestamp,
} from 'firebase/database';
import {
  ref as storageRef,
  uploadBytes,
  getDownloadURL,
} from 'firebase/storage';
import { db, storage } from './config';

export const pushLocation = (blindUid, coords) =>
  set(ref(db, `sessions/${blindUid}/location`), {
    latitude: coords.latitude,
    longitude: coords.longitude,
    accuracy: coords.accuracy || 0,
    timestamp: Date.now(),
  });

export const pushTelemetry = (blindUid, data) =>
  set(ref(db, `sessions/${blindUid}/telemetry`), {
    ...data,
    timestamp: Date.now(),
  });

export const pushHazardLog = (blindUid, entry) =>
  push(ref(db, `sessions/${blindUid}/hazard_log`), {
    ...entry,
    timestamp: Date.now(),
  });

export const triggerSOS = (blindUid, location) =>
  set(ref(db, `sessions/${blindUid}/sos`), {
    active: true,
    timestamp: Date.now(),
    location: location || null,
  });

export const clearSOS = (blindUid) =>
  set(ref(db, `sessions/${blindUid}/sos`), { active: false });

export const setStreamUrl = (blindUid, url) =>
  set(ref(db, `sessions/${blindUid}/stream_url`), url);

export const getStreamUrl = async (blindUid) => {
  const snap = await get(ref(db, `sessions/${blindUid}/stream_url`));
  return snap.exists() ? snap.val() : null;
};

export const subscribeToLocation = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/location`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const subscribeToTelemetry = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/telemetry`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const subscribeToSOS = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/sos`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const subscribeToStreamUrl = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/stream_url`);
  onValue(r, (snap) => callback(snap.exists() ? snap.val() : null));
  return () => off(r);
};

export const subscribeToHazardLog = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/hazard_log`);
  onValue(r, (snap) => {
    const val = snap.val();
    if (val) {
      const entries = Object.entries(val)
        .map(([id, data]) => ({ id, ...data }))
        .sort((a, b) => b.timestamp - a.timestamp);
      callback(entries);
    } else {
      callback([]);
    }
  });
  return () => off(r);
};

export const pushGeofenceZones = (blindUid, zones) =>
  set(ref(db, `sessions/${blindUid}/geofence`), { zones, enabled: true });

export const subscribeToGeofence = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/geofence`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const pushRemoteConfig = (blindUid, config) =>
  set(ref(db, `sessions/${blindUid}/remote_config`), {
    ...config,
    updatedAt: Date.now(),
  });

export const subscribeToRemoteConfig = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/remote_config`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const pushNextTurn = (blindUid, step) =>
  set(ref(db, `sessions/${blindUid}/next_turn`), step);

export const subscribeToNextTurn = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/next_turn`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const pushNavInstruction = (blindUid, message, hapticCode = 0) =>
  set(ref(db, `sessions/${blindUid}/nav_instruction`), {
    message,
    haptic: hapticCode,
    ts: Date.now(),
  });

export const subscribeToNavInstruction = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/nav_instruction`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const subscribeToGeofenceAlert = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/geofence_alert`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const uploadPTTAudio = async (caretakerUid, audioUri) => {
  const response = await fetch(audioUri);
  const blob = await response.blob();
  const fileRef = storageRef(storage, `ptt/${caretakerUid}/${Date.now()}.m4a`);
  await uploadBytes(fileRef, blob);
  return getDownloadURL(fileRef);
};

export const setPTTMessage = (blindUid, url, caretakerName) =>
  set(ref(db, `sessions/${blindUid}/ptt`), {
    url,
    caretakerName,
    timestamp: Date.now(),
    played: false,
  });

export const subscribeToPTT = (blindUid, callback) => {
  const r = ref(db, `sessions/${blindUid}/ptt`);
  onValue(r, (snap) => callback(snap.val()));
  return () => off(r);
};

export const markPTTPlayed = (blindUid) =>
  set(ref(db, `sessions/${blindUid}/ptt/played`), true);

export const setGeofenceAlert = (blindUid, data) =>
  set(ref(db, `sessions/${blindUid}/geofence_alert`), {
    ...data,
    timestamp: Date.now(),
  });
