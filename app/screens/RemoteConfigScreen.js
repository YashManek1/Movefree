import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Switch,
  Alert,
  ActivityIndicator,
  ScrollView,
} from 'react-native';
import Slider from '@react-native-community/slider';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '../src/context/AppContext';
import { pushRemoteConfig, subscribeToRemoteConfig } from '../src/firebase/database';
import { COLORS } from '../src/colors';

export default function RemoteConfigScreen({ onClose, patientUid: propPatientUid }) {
  const { patientUid: ctxPatientUid } = useApp();
  const pid = propPatientUid || ctxPatientUid;
  const [mode, setMode] = useState('indoor');
  const [volume, setVolume] = useState(80);
  const [sensitivity, setSensitivity] = useState(45);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!pid) return;
    const u = subscribeToRemoteConfig(pid, (config) => {
      if (!config) return;
      if (config.mode) setMode(config.mode);
      if (config.volume !== undefined) setVolume(config.volume);
      if (config.sensitivity !== undefined) setSensitivity(config.sensitivity);
    });
    return u;
  }, [pid]);

  const save = async () => {
    if (!pid) {
      Alert.alert('No Patient', 'Connect a patient first.');
      return;
    }
    setSaving(true);
    try {
      await pushRemoteConfig(pid, { mode, volume, sensitivity });
      Alert.alert('Saved', 'Configuration sent to the glasses.');
    } catch {
      Alert.alert('Error', 'Failed to save configuration.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name="settings-outline" size={18} color={COLORS.cyan} />
          <Text style={styles.title}>REMOTE CONFIG</Text>
        </View>
        {onClose && (
          <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
            <Ionicons name="close" size={22} color={COLORS.textSecondary} />
          </TouchableOpacity>
        )}
      </View>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>AI MODE</Text>
          <Text style={styles.cardDesc}>
            Toggle between indoor object detection and outdoor GPS navigation.
          </Text>
          <View style={styles.modeRow}>
            {['indoor', 'outdoor'].map((m) => (
              <TouchableOpacity
                key={m}
                style={[styles.modeBtn, mode === m && styles.modeBtnActive]}
                onPress={() => setMode(m)}
                activeOpacity={0.8}
              >
                <Ionicons
                  name={m === 'indoor' ? 'home-outline' : 'navigate-outline'}
                  size={18}
                  color={mode === m ? '#000' : COLORS.textSecondary}
                />
                <Text style={[styles.modeBtnText, mode === m && styles.modeBtnTextActive]}>
                  {m.toUpperCase()}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.card}>
          <View style={styles.sliderHeader}>
            <Text style={styles.cardTitle}>AUDIO VOLUME</Text>
            <Text style={styles.sliderValue}>{Math.round(volume)}%</Text>
          </View>
          <Text style={styles.cardDesc}>Controls the volume of spoken navigation instructions.</Text>
          <Slider
            style={styles.slider}
            minimumValue={0}
            maximumValue={100}
            step={5}
            value={volume}
            onValueChange={setVolume}
            minimumTrackTintColor={COLORS.cyan}
            maximumTrackTintColor="rgba(255,255,255,0.1)"
            thumbTintColor={COLORS.cyan}
          />
        </View>

        <View style={styles.card}>
          <View style={styles.sliderHeader}>
            <Text style={styles.cardTitle}>DETECTION SENSITIVITY</Text>
            <Text style={styles.sliderValue}>{Math.round(sensitivity)}%</Text>
          </View>
          <Text style={styles.cardDesc}>Confidence threshold for AI object detection. Lower = more detections.</Text>
          <Slider
            style={styles.slider}
            minimumValue={20}
            maximumValue={80}
            step={5}
            value={sensitivity}
            onValueChange={setSensitivity}
            minimumTrackTintColor={COLORS.orange}
            maximumTrackTintColor="rgba(255,255,255,0.1)"
            thumbTintColor={COLORS.orange}
          />
        </View>

        <View style={styles.infoCard}>
          <Ionicons name="information-circle-outline" size={16} color={COLORS.cyan} />
          <Text style={styles.infoText}>
            Changes are applied to the glasses in real-time once saved.
          </Text>
        </View>

        <TouchableOpacity
          style={[styles.saveBtn, saving && { opacity: 0.6 }]}
          onPress={save}
          disabled={saving}
          activeOpacity={0.85}
        >
          {saving ? (
            <ActivityIndicator color="#000" />
          ) : (
            <>
              <Ionicons name="cloud-upload-outline" size={18} color="#000" />
              <Text style={styles.saveBtnText}>SAVE & APPLY</Text>
            </>
          )}
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingTop: 56, paddingBottom: 14,
    borderBottomWidth: 1, borderBottomColor: COLORS.border,
  },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  title: { fontSize: 14, fontWeight: '900', color: COLORS.white, letterSpacing: 3 },
  closeBtn: { padding: 4 },
  scroll: { padding: 16, gap: 12 },
  card: { backgroundColor: COLORS.bgCard, borderRadius: 14, borderWidth: 1, borderColor: COLORS.border, padding: 18, gap: 10 },
  cardTitle: { color: COLORS.textSecondary, fontSize: 10, fontWeight: '700', letterSpacing: 2 },
  cardDesc: { color: COLORS.textMuted, fontSize: 12, lineHeight: 18 },
  modeRow: { flexDirection: 'row', gap: 10 },
  modeBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    borderWidth: 1, borderColor: COLORS.border, borderRadius: 10, paddingVertical: 12,
    backgroundColor: COLORS.bg,
  },
  modeBtnActive: { backgroundColor: COLORS.cyan, borderColor: COLORS.cyan },
  modeBtnText: { color: COLORS.textSecondary, fontWeight: '700', fontSize: 12, letterSpacing: 1 },
  modeBtnTextActive: { color: '#000' },
  sliderHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  sliderValue: { color: COLORS.white, fontSize: 14, fontWeight: '700' },
  slider: { width: '100%', height: 32 },
  infoCard: {
    flexDirection: 'row', gap: 8, alignItems: 'flex-start',
    backgroundColor: COLORS.cyanBg, borderRadius: 10,
    borderWidth: 1, borderColor: COLORS.border, padding: 12,
  },
  infoText: { flex: 1, color: COLORS.cyan, fontSize: 12, lineHeight: 18 },
  saveBtn: {
    backgroundColor: COLORS.cyan, borderRadius: 12, paddingVertical: 16,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, marginTop: 4,
  },
  saveBtnText: { color: '#000', fontWeight: '900', fontSize: 14, letterSpacing: 2 },
});
