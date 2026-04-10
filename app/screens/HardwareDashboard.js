import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  StatusBar,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '../src/context/AppContext';
import { subscribeToTelemetry } from '../src/firebase/database';
import { COLORS } from '../src/colors';

function BatteryMeter({ level }) {
  const pct = Math.max(0, Math.min(100, level || 0));
  const color = pct > 50 ? COLORS.green : pct > 20 ? COLORS.orange : COLORS.red;
  return (
    <View style={bStyles.container}>
      <View style={bStyles.track}>
        <View style={[bStyles.fill, { width: `${pct}%`, backgroundColor: color }]} />
      </View>
      <View style={bStyles.cap} />
      <Text style={[bStyles.label, { color }]}>{pct.toFixed(0)}%</Text>
    </View>
  );
}

const bStyles = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  track: { flex: 1, height: 14, backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: 7, overflow: 'hidden', borderWidth: 1, borderColor: COLORS.border },
  fill: { height: '100%', borderRadius: 7 },
  cap: { width: 4, height: 8, backgroundColor: COLORS.textMuted, borderRadius: 1 },
  label: { width: 36, fontSize: 13, fontWeight: '700', textAlign: 'right' },
});

function SensorRow({ label, active }) {
  return (
    <View style={sStyles.row}>
      <Ionicons
        name={active ? 'checkmark-circle' : 'close-circle'}
        size={16}
        color={active ? COLORS.green : COLORS.red}
      />
      <Text style={sStyles.label}>{label}</Text>
      <View style={[sStyles.pill, { backgroundColor: active ? COLORS.greenDim : COLORS.redDim }]}>
        <Text style={[sStyles.pillText, { color: active ? COLORS.green : COLORS.red }]}>
          {active ? 'ONLINE' : 'OFFLINE'}
        </Text>
      </View>
    </View>
  );
}

const sStyles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  label: { flex: 1, color: COLORS.white, fontSize: 14 },
  pill: { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 3 },
  pillText: { fontSize: 10, fontWeight: '700', letterSpacing: 1 },
});

export default function HardwareDashboard() {
  const { patientUid } = useApp();
  const [telemetry, setTelemetry] = useState(null);

  useEffect(() => {
    if (!patientUid) return;
    const u = subscribeToTelemetry(patientUid, setTelemetry);
    return u;
  }, [patientUid]);

  const battery = telemetry?.battery || 0;
  const temperature = telemetry?.temperature || 0;
  const signal = telemetry?.signal || 'N/A';
  const mode = telemetry?.mode || 'unknown';
  const tempColor = temperature > 60 ? COLORS.red : temperature > 45 ? COLORS.orange : COLORS.green;

  const sensors = [
    { label: 'Camera Stream', active: !!telemetry?.stream_active },
    { label: 'GPS Location', active: !!telemetry?.gps_active },
    { label: 'Ultrasonic Sensor', active: !!telemetry?.sonar_active },
    { label: 'Microphone', active: !!telemetry?.mic_active },
    { label: 'Gemini AI', active: !!telemetry?.gemini_active },
  ];

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name="heart-outline" size={18} color={COLORS.cyan} />
          <Text style={styles.title}>HARDWARE VITALS</Text>
        </View>
        <Text style={styles.timestamp}>
          {telemetry?.timestamp ? new Date(telemetry.timestamp).toLocaleTimeString() : 'No data'}
        </Text>
      </View>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>BATTERY LEVEL</Text>
          <BatteryMeter level={battery} />
        </View>

        <View style={styles.row}>
          <View style={[styles.card, { flex: 1 }]}>
            <Ionicons name="thermometer-outline" size={20} color={tempColor} />
            <Text style={[styles.bigValue, { color: tempColor }]}>{temperature.toFixed(1)}°C</Text>
            <Text style={styles.cardTitle}>EDGE TEMP</Text>
          </View>
          <View style={[styles.card, { flex: 1 }]}>
            <Ionicons name="cellular-outline" size={20} color={COLORS.cyan} />
            <Text style={[styles.bigValue, { color: COLORS.cyan }]}>{signal}</Text>
            <Text style={styles.cardTitle}>SIGNAL</Text>
          </View>
          <View style={[styles.card, { flex: 1 }]}>
            <Ionicons name="navigate-outline" size={20} color={COLORS.orange} />
            <Text style={[styles.bigValue, { color: COLORS.orange, fontSize: 14 }]}>{mode.toUpperCase()}</Text>
            <Text style={styles.cardTitle}>MODE</Text>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>SENSOR CONNECTIVITY</Text>
          {sensors.map((s) => (
            <SensorRow key={s.label} label={s.label} active={s.active} />
          ))}
        </View>

        {!patientUid && (
          <View style={styles.noDataCard}>
            <Ionicons name="warning-outline" size={24} color={COLORS.orange} />
            <Text style={styles.noDataText}>No patient connected. Connect a patient from the app to see live hardware data.</Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    paddingHorizontal: 16, paddingTop: 56, paddingBottom: 14,
    borderBottomWidth: 1, borderBottomColor: COLORS.border,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
  },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  title: { fontSize: 14, fontWeight: '900', color: COLORS.white, letterSpacing: 3 },
  timestamp: { color: COLORS.textMuted, fontSize: 11 },
  scroll: { padding: 12, gap: 10 },
  card: {
    backgroundColor: COLORS.bgCard, borderRadius: 14,
    borderWidth: 1, borderColor: COLORS.border, padding: 16, gap: 10,
  },
  row: { flexDirection: 'row', gap: 10 },
  cardTitle: { color: COLORS.textSecondary, fontSize: 10, fontWeight: '700', letterSpacing: 2 },
  bigValue: { fontSize: 26, fontWeight: '900', color: COLORS.white },
  noDataCard: {
    backgroundColor: COLORS.orangeDim, borderRadius: 14,
    borderWidth: 1, borderColor: COLORS.orange, padding: 20,
    alignItems: 'center', gap: 10,
  },
  noDataText: { color: COLORS.orange, fontSize: 13, textAlign: 'center', lineHeight: 20 },
});
