import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  StatusBar,
  Alert,
  Vibration,
  Platform,
} from 'react-native';
import * as Location from 'expo-location';
import * as FileSystem from 'expo-file-system';
import * as Speech from 'expo-speech';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '../src/context/AppContext';
import {
  pushLocation,
  pushTelemetry,
  triggerSOS,
  clearSOS,
  subscribeToRemoteConfig,
  subscribeToNavInstruction,
  subscribeToSOS,
} from '../src/firebase/database';
import { signOut } from '../src/firebase/auth';
import { usePTTSocket } from '../src/hooks/usePTTSocket';
import { COLORS } from '../src/colors';

const SOS_PATTERN    = [0, 500, 300, 500, 300, 500, 300, 500, 300, 500, 300, 500];
const HAPTIC_LEFT    = [0, 300];                      
const HAPTIC_RIGHT   = [0, 300];                      
const HAPTIC_STOP    = [0, 600, 200, 600, 200, 600];  

function triggerHaptic(code) {
  if (code === 1) Vibration.vibrate(HAPTIC_LEFT);
  else if (code === 2) Vibration.vibrate(HAPTIC_RIGHT);
  else if (code >= 3) Vibration.vibrate(HAPTIC_STOP);
}

export default function TransmittingScreen({ navigation }) {
  const { user } = useApp();

  const pulseAnim = useRef(new Animated.Value(1)).current;
  const dotAnim   = useRef(new Animated.Value(0)).current;

  const [sosActive,    setSosActive]    = useState(false);
  const [coords,       setCoords]       = useState(null);
  const [battery]                       = useState(88);
  const [signal]                        = useState('4G');
  const [pttActive,    setPttActive]    = useState(false);
  const [pttFrom,      setPttFrom]      = useState('');
  const [lastNav,      setLastNav]      = useState('');   

  const sosAudioRef    = useRef(null);
  const intervalRef    = useRef(null);
  const chunkBufferRef = useRef([]);
  const pttTimerRef    = useRef(null);
  const lastNavTs      = useRef(0);         

  const { connected: wsConnected } = usePTTSocket('blind', {
    onBinaryChunk: (buf) => { chunkBufferRef.current.push(buf); },
    onPTTStart: (caretaker) => {
      setPttActive(true);
      setPttFrom(caretaker);
      chunkBufferRef.current = [];
      clearTimeout(pttTimerRef.current);
    },
    onPTTStop: () => {
      pttTimerRef.current = setTimeout(() => {
        playBufferedAudio();
        setPttActive(false);
      }, 200);
    },
  });

  const playBufferedAudio = useCallback(async () => {
    const chunks = chunkBufferRef.current;
    chunkBufferRef.current = [];
    if (!chunks.length) return;
    try {
      const totalLen = chunks.reduce((s, b) => s + b.byteLength, 0);
      const combined = new Uint8Array(totalLen);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(new Uint8Array(chunk), offset);
        offset += chunk.byteLength;
      }
      const tempUri = FileSystem.cacheDirectory + `ptt_${Date.now()}.m4a`;
      let binary = '';
      for (let i = 0; i < combined.byteLength; i++) binary += String.fromCharCode(combined[i]);
      await FileSystem.writeAsStringAsync(tempUri, btoa(binary), {
        encoding: FileSystem.EncodingType.Base64,
      });
      await Audio.setAudioModeAsync({ playsInSilentModeIOS: true, shouldDuckAndroid: true });
      const { sound } = await Audio.Sound.createAsync({ uri: tempUri });
      await sound.playAsync();
      sound.setOnPlaybackStatusUpdate((s) => {
        if (s.didJustFinish) {
          sound.unloadAsync();
          FileSystem.deleteAsync(tempUri, { idempotent: true }).catch(() => {});
        }
      });
    } catch {}
  }, []);

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.3, duration: 1200, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1,   duration: 1200, useNativeDriver: true }),
      ])
    ).start();
    Animated.loop(
      Animated.sequence([
        Animated.timing(dotAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
        Animated.timing(dotAnim, { toValue: 0, duration: 800, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  useEffect(() => {
    let sub;
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') return;
      sub = await Location.watchPositionAsync(
        { accuracy: Location.Accuracy.High, timeInterval: 5000, distanceInterval: 5 },
        (loc) => {
          const c = { latitude: loc.coords.latitude, longitude: loc.coords.longitude, accuracy: loc.coords.accuracy };
          setCoords(c);
          if (user?.uid) pushLocation(user.uid, c);
        }
      );
    })();
    return () => sub?.remove();
  }, [user]);

  useEffect(() => {
    if (!user?.uid) return;
    intervalRef.current = setInterval(() => {
      pushTelemetry(user.uid, {
        battery, signal, mode: 'active', platform: Platform.OS,
        mic_active: true, gps_active: !!coords,
        stream_active: false, gemini_active: false, sonar_active: false,
        temperature: 38 + Math.random() * 5,
      });
    }, 8000);
    return () => clearInterval(intervalRef.current);
  }, [user, battery, signal, coords]);

  useEffect(() => {
    if (!user?.uid) return;
    const unsub = subscribeToNavInstruction(user.uid, (data) => {
      if (!data || !data.message) return;

      if (data.ts && data.ts === lastNavTs.current) return;
      lastNavTs.current = data.ts || 0;

      Speech.stop();
      Speech.speak(data.message, {
        language: 'en-US',
        pitch: 1.0,
        rate: 0.95,
      });

      setLastNav(data.message);

      if (data.haptic && data.haptic > 0) {
        triggerHaptic(data.haptic);
      }
    });
    return unsub;
  }, [user]);

  useEffect(() => {
    if (!user?.uid) return;
    const unsub = subscribeToSOS(user.uid, async (data) => {
      if (!data?.active) return;

      if (sosActive) return;

      setSosActive(true);
      Vibration.vibrate(SOS_PATTERN, true);
      try {
        await Audio.setAudioModeAsync({ playsInSilentModeIOS: true, shouldDuckAndroid: false });
        const { sound } = await Audio.Sound.createAsync(
          require('../assets/alert.mp3'),
          { isLooping: true, volume: 1.0 }
        );
        sosAudioRef.current = sound;
        await sound.playAsync();
      } catch {}
    });
    return unsub;
  }, [user, sosActive]);

  useEffect(() => {
    if (!user?.uid) return;
    const unsub = subscribeToRemoteConfig(user.uid, () => {});
    return unsub;
  }, [user]);

  const handleSOS = async () => {
    if (sosActive) {
      setSosActive(false);
      Vibration.cancel();
      Speech.stop();
      if (sosAudioRef.current) {
        await sosAudioRef.current.stopAsync().catch(() => {});
        await sosAudioRef.current.unloadAsync().catch(() => {});
        sosAudioRef.current = null;
      }
      if (user?.uid) clearSOS(user.uid);
      return;
    }

    setSosActive(true);
    Vibration.vibrate(SOS_PATTERN, true);
    Speech.speak('Emergency SOS activated. Alerting your caretaker.', { language: 'en-US' });

    try {
      await Audio.setAudioModeAsync({ playsInSilentModeIOS: true, shouldDuckAndroid: false });
      const { sound } = await Audio.Sound.createAsync(
        require('../assets/alert.mp3'),
        { isLooping: true, volume: 1.0 }
      );
      sosAudioRef.current = sound;
      await sound.playAsync();
    } catch {}

    if (user?.uid) triggerSOS(user.uid, coords);
  };

  const handleSignOut = () => {
    Alert.alert('Sign Out', 'Are you sure?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Sign Out', style: 'destructive',
        onPress: async () => {
          Vibration.cancel();
          Speech.stop();
          if (sosAudioRef.current) await sosAudioRef.current.unloadAsync().catch(() => {});
          await signOut();
          navigation.replace('Login');
        },
      },
    ]);
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      <View style={styles.header}>
        <View style={styles.logoRow}>
          <View style={styles.logoDot} />
          <Text style={styles.logoText}>MOOVEFREE</Text>
        </View>
        <TouchableOpacity onPress={handleSignOut} style={styles.signOutBtn}>
          <Ionicons name="log-out-outline" size={20} color={COLORS.textSecondary} />
        </TouchableOpacity>
      </View>

      <View style={styles.roleTag}>
        <Ionicons name="accessibility" size={13} color={COLORS.cyan} />
        <Text style={styles.roleText}>VISUALLY IMPAIRED</Text>
      </View>

      {}
      {pttActive && (
        <View style={styles.pttBanner}>
          <Ionicons name="volume-high" size={16} color="#000" />
          <Text style={styles.pttBannerText}>Caretaker speaking: {pttFrom}</Text>
        </View>
      )}

      {}
      {lastNav !== '' && (
        <View style={styles.navBanner}>
          <Ionicons name="navigate-outline" size={14} color={COLORS.cyan} />
          <Text style={styles.navBannerText} numberOfLines={2}>{lastNav}</Text>
        </View>
      )}

      <View style={styles.pulseContainer}>
        <Animated.View style={[styles.pulseOuter, { transform: [{ scale: pulseAnim }] }]} />
        <View style={styles.pulseInner}>
          <Ionicons name="radio-outline" size={36} color={COLORS.cyan} />
        </View>
      </View>

      <Text style={styles.transmittingLabel}>TRANSMITTING</Text>
      <View style={styles.statusRow}>
        <Animated.View style={[styles.liveDot, { opacity: dotAnim }]} />
        <Text style={styles.statusText}>Live data active</Text>
      </View>

      <View style={styles.dataGrid}>
        <View style={styles.dataCard}>
          <Ionicons name="location-outline" size={16} color={COLORS.cyan} />
          <Text style={styles.dataLabel}>GPS</Text>
          <Text style={styles.dataValue}>
            {coords ? `${coords.latitude.toFixed(4)}` : 'Acquiring...'}
          </Text>
        </View>
        <View style={styles.dataCard}>
          <Ionicons name="battery-charging-outline" size={16} color={COLORS.green} />
          <Text style={styles.dataLabel}>BATTERY</Text>
          <Text style={[styles.dataValue, { color: COLORS.green }]}>{Math.floor(battery)}%</Text>
        </View>
        <View style={styles.dataCard}>
          <Ionicons
            name={wsConnected ? 'wifi-outline' : 'wifi'}
            size={16}
            color={wsConnected ? COLORS.cyan : COLORS.textMuted}
          />
          <Text style={styles.dataLabel}>PTT</Text>
          <Text style={[styles.dataValue, { color: wsConnected ? COLORS.cyan : COLORS.textMuted, fontSize: 11 }]}>
            {wsConnected ? 'ONLINE' : 'OFFLINE'}
          </Text>
        </View>
      </View>

      <TouchableOpacity
        style={[styles.sosBtn, sosActive && styles.sosBtnActive]}
        onPress={handleSOS}
        activeOpacity={0.85}
      >
        <Ionicons
          name={sosActive ? 'close-circle-outline' : 'alert-circle-outline'}
          size={24}
          color={sosActive ? '#000' : COLORS.red}
        />
        <Text style={[styles.sosBtnText, sosActive && { color: '#000' }]}>
          {sosActive ? 'CANCEL SOS' : '✱  EMERGENCY SOS'}
        </Text>
      </TouchableOpacity>

      <Text style={styles.footerNote}>Keep this screen open to continue transmitting.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg, padding: 24, paddingTop: 60 },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 },
  logoRow: { flexDirection: 'row', alignItems: 'center' },
  logoDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.cyan, marginRight: 10 },
  logoText: { fontSize: 18, fontWeight: '900', color: COLORS.cyan, letterSpacing: 4 },
  signOutBtn: { padding: 8 },
  roleTag: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: COLORS.cyanBg, borderWidth: 1, borderColor: COLORS.border,
    borderRadius: 8, paddingVertical: 6, paddingHorizontal: 12,
    alignSelf: 'flex-start', marginBottom: 12,
  },
  roleText: { color: COLORS.cyan, fontSize: 11, fontWeight: '700', letterSpacing: 1 },
  pttBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: COLORS.cyan, borderRadius: 10,
    paddingVertical: 10, paddingHorizontal: 16, marginBottom: 8,
  },
  pttBannerText: { color: '#000', fontWeight: '700', fontSize: 13 },
  navBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: COLORS.cyanBg, borderWidth: 1, borderColor: COLORS.border,
    borderRadius: 10, paddingVertical: 8, paddingHorizontal: 14, marginBottom: 8,
  },
  navBannerText: { flex: 1, color: COLORS.cyan, fontSize: 12, fontWeight: '600' },
  pulseContainer: { alignItems: 'center', justifyContent: 'center', height: 120, marginBottom: 14 },
  pulseOuter: {
    position: 'absolute', width: 130, height: 130, borderRadius: 65,
    backgroundColor: 'rgba(0,229,255,0.08)', borderWidth: 1, borderColor: 'rgba(0,229,255,0.2)',
  },
  pulseInner: {
    width: 80, height: 80, borderRadius: 40,
    backgroundColor: COLORS.cyanBg, borderWidth: 1.5, borderColor: COLORS.borderStrong,
    alignItems: 'center', justifyContent: 'center',
  },
  transmittingLabel: { fontSize: 22, fontWeight: '900', color: COLORS.white, textAlign: 'center', letterSpacing: 4, marginBottom: 8 },
  statusRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 20 },
  liveDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: COLORS.green },
  statusText: { color: COLORS.textSecondary, fontSize: 13, letterSpacing: 1 },
  dataGrid: { flexDirection: 'row', gap: 10, marginBottom: 20 },
  dataCard: {
    flex: 1, backgroundColor: COLORS.bgCard, borderRadius: 12,
    borderWidth: 1, borderColor: COLORS.border, padding: 14, alignItems: 'center', gap: 6,
  },
  dataLabel: { color: COLORS.textSecondary, fontSize: 10, letterSpacing: 1, fontWeight: '600' },
  dataValue: { color: COLORS.white, fontSize: 14, fontWeight: '700' },
  sosBtn: {
    borderWidth: 2, borderColor: COLORS.red, borderRadius: 14,
    paddingVertical: 16, flexDirection: 'row', alignItems: 'center',
    justifyContent: 'center', gap: 10, marginBottom: 20, backgroundColor: COLORS.redDim,
  },
  sosBtnActive: { backgroundColor: COLORS.red, borderColor: COLORS.red },
  sosBtnText: { color: COLORS.red, fontWeight: '900', fontSize: 15, letterSpacing: 2 },
  footerNote: { color: COLORS.textMuted, fontSize: 12, textAlign: 'center', letterSpacing: 0.5 },
});
