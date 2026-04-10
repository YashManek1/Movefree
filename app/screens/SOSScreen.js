import React, { useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  Alert,
  ActivityIndicator,
  Animated,
} from 'react-native';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '../src/context/AppContext';
import { triggerSOS, clearSOS } from '../src/firebase/database';
import { usePTTSocket } from '../src/hooks/usePTTSocket';
import { COLORS } from '../src/colors';

export default function SOSScreen() {
  const { user, patientUid, patientName } = useApp();
  const [sosActive, setSosActive] = useState(false);
  const [pttActive, setPttActive] = useState(false);
  const [pttStatus, setPttStatus] = useState('HOLD TO TALK');
  const pttAnim = useRef(new Animated.Value(1)).current;
  const recordingRef = useRef(null);
  const animLoopRef = useRef(null);

  const { connected, sendBinary, sendJSON } = usePTTSocket('caretaker', {
    onConnect: () => setPttStatus('HOLD TO TALK'),
    onDisconnect: () => setPttStatus('RELAY OFFLINE'),
  });

  const startPulse = useCallback(() => {
    animLoopRef.current = Animated.loop(
      Animated.sequence([
        Animated.timing(pttAnim, { toValue: 1.15, duration: 600, useNativeDriver: true }),
        Animated.timing(pttAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
      ])
    );
    animLoopRef.current.start();
  }, [pttAnim]);

  const stopPulse = useCallback(() => {
    animLoopRef.current?.stop();
    pttAnim.setValue(1);
  }, [pttAnim]);

  const startPTT = async () => {
    if (!connected) {
      Alert.alert('Not Connected', 'PTT relay server is not reachable. Make sure the laptop is running the relay server.');
      return;
    }
    try {
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) {
        Alert.alert('Permission Required', 'Microphone access needed.');
        return;
      }
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setPttActive(true);
      setPttStatus('RELEASE TO SEND');
      startPulse();

      sendJSON({ type: 'ptt_start', name: user?.displayName || 'Caretaker' });
    } catch (err) {
      Alert.alert('Error', 'Could not start microphone.');
    }
  };

  const stopPTT = async () => {
    if (!recordingRef.current) return;
    stopPulse();
    setPttActive(false);
    setPttStatus('SENDING...');

    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;

      const response = await fetch(uri);
      const buffer = await response.arrayBuffer();

      sendBinary(buffer);
      sendJSON({ type: 'ptt_stop' });

      await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
      setPttStatus('SENT ✓');
      setTimeout(() => setPttStatus('HOLD TO TALK'), 2000);
    } catch {
      setPttStatus('FAILED — Retry');
      setTimeout(() => setPttStatus('HOLD TO TALK'), 3000);
    }
  };

  const toggleSOS = async () => {
    if (!patientUid) {
      Alert.alert('No Patient', 'Connect a patient first.');
      return;
    }
    if (sosActive) {
      setSosActive(false);
      await clearSOS(patientUid);
      return;
    }
    Alert.alert(
      'Trigger Emergency SOS',
      `Send an SOS alert to ${patientName || 'patient'}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'TRIGGER',
          style: 'destructive',
          onPress: async () => {
            setSosActive(true);
            await triggerSOS(patientUid, null);
          },
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name="alert-circle-outline" size={18} color={COLORS.red} />
          <Text style={styles.title}>SOS & COMMUNICATION</Text>
        </View>
        <Text style={styles.subtitle}>
          {patientName ? `Monitoring: ${patientName}` : 'No patient connected'}
        </Text>
      </View>

      {}
      <View style={styles.section}>
        <View style={styles.sectionHeaderRow}>
          <Text style={styles.sectionLabel}>PUSH-TO-TALK</Text>
          <View style={[styles.wsIndicator, { borderColor: connected ? COLORS.green : COLORS.red }]}>
            <View style={[styles.wsDot, { backgroundColor: connected ? COLORS.green : COLORS.red }]} />
            <Text style={[styles.wsText, { color: connected ? COLORS.green : COLORS.red }]}>
              {connected ? 'RELAY ONLINE' : 'RELAY OFFLINE'}
            </Text>
          </View>
        </View>

        <Text style={styles.sectionDesc}>
          Hold button to record. Release to send audio to the glasses in real‑time via WebSocket relay.
        </Text>

        <View style={styles.pttContainer}>
          <Animated.View style={{ transform: [{ scale: pttAnim }] }}>
            <TouchableOpacity
              style={[
                styles.pttBtn,
                pttActive && styles.pttBtnActive,
                !connected && styles.pttBtnDisabled,
              ]}
              onPressIn={startPTT}
              onPressOut={stopPTT}
              disabled={!connected}
              activeOpacity={0.85}
            >
              <Ionicons
                name={pttActive ? 'mic' : 'mic-outline'}
                size={40}
                color={!connected ? COLORS.textMuted : pttActive ? '#000' : COLORS.cyan}
              />
            </TouchableOpacity>
          </Animated.View>
          <Text style={[styles.pttLabel, pttActive && { color: COLORS.cyan }]}>{pttStatus}</Text>
        </View>

        {!connected && (
          <View style={styles.warningCard}>
            <Ionicons name="warning-outline" size={16} color={COLORS.orange} />
            <Text style={styles.warningText}>
              Start the PTT relay server on the laptop:{'\n'}
              <Text style={styles.warningCode}>cd ptt-relay && npm start</Text>
            </Text>
          </View>
        )}
      </View>

      <View style={styles.divider} />

      {}
      <View style={styles.section}>
        <Text style={styles.sectionLabel}>EMERGENCY SOS</Text>
        <Text style={styles.sectionDesc}>
          Trigger an emergency alert on the patient's device with vibration and alarm sound.
        </Text>

        <TouchableOpacity
          style={[styles.sosBtn, sosActive && styles.sosBtnActive]}
          onPress={toggleSOS}
          activeOpacity={0.85}
        >
          <Ionicons
            name={sosActive ? 'close-circle' : 'alert-circle'}
            size={28}
            color={sosActive ? '#000' : COLORS.red}
          />
          <Text style={[styles.sosBtnText, sosActive && { color: '#000' }]}>
            {sosActive ? 'CANCEL SOS ALERT' : '✱  TRIGGER EMERGENCY SOS'}
          </Text>
        </TouchableOpacity>

        {sosActive && (
          <View style={styles.sosActiveCard}>
            <View style={styles.sosActiveDot} />
            <Text style={styles.sosActiveText}>SOS signal active on patient's device</Text>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    paddingHorizontal: 16, paddingTop: 56, paddingBottom: 14,
    borderBottomWidth: 1, borderBottomColor: COLORS.border,
  },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4 },
  title: { fontSize: 13, fontWeight: '900', color: COLORS.white, letterSpacing: 2 },
  subtitle: { color: COLORS.textSecondary, fontSize: 12 },
  section: { padding: 20 },
  sectionHeaderRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 },
  sectionLabel: { color: COLORS.textSecondary, fontSize: 10, fontWeight: '700', letterSpacing: 2 },
  wsIndicator: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    borderWidth: 1, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3,
  },
  wsDot: { width: 6, height: 6, borderRadius: 3 },
  wsText: { fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  sectionDesc: { color: COLORS.textMuted, fontSize: 13, lineHeight: 20, marginBottom: 24 },
  pttContainer: { alignItems: 'center', gap: 16 },
  pttBtn: {
    width: 110, height: 110, borderRadius: 55,
    backgroundColor: COLORS.cyanBg, borderWidth: 2, borderColor: COLORS.cyan,
    alignItems: 'center', justifyContent: 'center',
  },
  pttBtnActive: { backgroundColor: COLORS.cyan },
  pttBtnDisabled: { borderColor: COLORS.textMuted, backgroundColor: 'rgba(255,255,255,0.03)' },
  pttLabel: { color: COLORS.textSecondary, fontSize: 12, fontWeight: '700', letterSpacing: 2 },
  warningCard: {
    flexDirection: 'row', gap: 10, alignItems: 'flex-start',
    marginTop: 16, backgroundColor: COLORS.orangeDim,
    borderRadius: 10, borderWidth: 1, borderColor: COLORS.orange, padding: 12,
  },
  warningText: { flex: 1, color: COLORS.orange, fontSize: 12, lineHeight: 18 },
  warningCode: { fontFamily: 'monospace', color: COLORS.white, fontSize: 11 },
  divider: { height: 1, backgroundColor: COLORS.border, marginHorizontal: 20 },
  sosBtn: {
    borderWidth: 2, borderColor: COLORS.red, borderRadius: 14,
    paddingVertical: 18, flexDirection: 'row', alignItems: 'center',
    justifyContent: 'center', gap: 10, backgroundColor: COLORS.redDim,
  },
  sosBtnActive: { backgroundColor: COLORS.red },
  sosBtnText: { color: COLORS.red, fontWeight: '900', fontSize: 15, letterSpacing: 1 },
  sosActiveCard: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    marginTop: 14, backgroundColor: COLORS.redDim,
    borderRadius: 10, borderWidth: 1, borderColor: COLORS.redBorder, padding: 12,
  },
  sosActiveDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.red },
  sosActiveText: { color: COLORS.red, fontSize: 13, fontWeight: '600' },
});
