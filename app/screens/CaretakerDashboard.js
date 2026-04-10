import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Switch,
  ScrollView,
  Dimensions,
  StatusBar,
  Alert,
  Modal,
  Animated,
} from 'react-native';
import { WebView } from 'react-native-webview';
import MapView, { Marker, Circle, UrlTile } from 'react-native-maps';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '../src/context/AppContext';
import {
  subscribeToLocation,
  subscribeToTelemetry,
  subscribeToSOS,
  subscribeToStreamUrl,
  subscribeToNextTurn,
  subscribeToHazardLog,
  subscribeToGeofence,
  subscribeToGeofenceAlert,
  clearSOS,
} from '../src/firebase/database';
import { signOut } from '../src/firebase/auth';
import { COLORS } from '../src/colors';
import IncidentLogScreen from './IncidentLogScreen';
import RemoteConfigScreen from './RemoteConfigScreen';

const { width } = Dimensions.get('window');

function VitalCard({ icon, value, unit, label, color = COLORS.cyan }) {
  return (
    <View style={vStyles.card}>
      <Ionicons name={icon} size={18} color={color} />
      <Text style={vStyles.value}>{value}</Text>
      <Text style={vStyles.unit}>{unit}</Text>
      <View style={vStyles.bar}>
        <View style={[vStyles.barFill, { backgroundColor: color, width: '60%' }]} />
      </View>
    </View>
  );
}

const vStyles = StyleSheet.create({
  card: {
    flex: 1,
    backgroundColor: COLORS.bgCard,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 12,
    alignItems: 'center',
    gap: 3,
  },
  value: { fontSize: 22, fontWeight: '900', color: COLORS.white, lineHeight: 26 },
  unit: { fontSize: 10, color: COLORS.textSecondary, letterSpacing: 1, fontWeight: '600' },
  bar: { width: '100%', height: 2, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 2, marginTop: 4 },
  barFill: { height: 2, borderRadius: 2 },
});

export default function CaretakerDashboard({ navigation }) {
  const { user, patientUid, patientName } = useApp();
  const [location, setLocation] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [sos, setSos] = useState(null);
  const [streamUrl, setStreamUrl] = useState(null);
  const [nextTurn, setNextTurn] = useState(null);
  const [hazardLog, setHazardLog] = useState([]);
  const [geofence, setGeofence] = useState(null);
  const [aiOverlay, setAiOverlay] = useState(true);
  const [showIncidents, setShowIncidents] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [mode, setMode] = useState('OUTDOOR');
  const sosAnim = useRef(new Animated.Value(1)).current;
  const mapRef = useRef(null);

  useEffect(() => {
    if (!patientUid) return;
    const unsubs = [
      subscribeToLocation(patientUid, setLocation),
      subscribeToTelemetry(patientUid, setTelemetry),
      subscribeToSOS(patientUid, setSos),
      subscribeToStreamUrl(patientUid, setStreamUrl),
      subscribeToNextTurn(patientUid, setNextTurn),
      subscribeToHazardLog(patientUid, setHazardLog),
      subscribeToGeofence(patientUid, setGeofence),
    ];
    return () => unsubs.forEach((u) => u && u());
  }, [patientUid]);

  useEffect(() => {
    if (sos?.active) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(sosAnim, { toValue: 0.3, duration: 300, useNativeDriver: true }),
          Animated.timing(sosAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
        ])
      ).start();
    } else {
      sosAnim.setValue(1);
    }
  }, [sos]);

  useEffect(() => {
    if (sos?.active) {
      Alert.alert(
        '🚨 EMERGENCY SOS',
        `${patientName || 'Patient'} has triggered an SOS alert!\n\nLocation: ${
          sos.location
            ? `${sos.location.latitude?.toFixed(5)}, ${sos.location.longitude?.toFixed(5)}`
            : 'Unknown'
        }`,
        [
          { text: 'Dismiss', style: 'cancel' },
          {
            text: 'Clear SOS',
            style: 'destructive',
            onPress: () => clearSOS(patientUid),
          },
        ]
      );
    }
  }, [sos?.active]);

  const region = location
    ? { latitude: location.latitude, longitude: location.longitude, latitudeDelta: 0.005, longitudeDelta: 0.005 }
    : { latitude: 19.076, longitude: 72.8777, latitudeDelta: 0.05, longitudeDelta: 0.05 };

  const streamHtml = streamUrl
    ? `<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1"><style>html,body{margin:0;padding:0;background:#000;overflow:hidden;width:100%;height:100%}img{width:100%;height:100%;object-fit:cover;background:#000}</style></head><body><img src="${streamUrl}" onerror="this.style.display='none';document.body.innerHTML='<div style=\\"color:#00E5FF;font-family:monospace;text-align:center;padding:40px;font-size:14px;\\">● CONNECTING TO STREAM...</div>'"/></body></html>`
    : `<!DOCTYPE html><html><head><style>html,body{margin:0;padding:0;background:#0A0E1A;display:flex;align-items:center;justify-content:center;height:100%}p{color:#00E5FF;font-family:monospace;font-size:13px;text-align:center;padding:20px}</style></head><body><p>● AWAITING STREAM URL<br/>Start indoor or outdoor AI module on the laptop.</p></body></html>`;

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      <View style={styles.topBar}>
        <View>
          <View style={styles.modeRow}>
            <Ionicons name="navigate-circle-outline" size={13} color={COLORS.cyan} />
            <Text style={styles.modeText}>WALKING · {mode} MODE</Text>
          </View>
          <View style={styles.liveRow}>
            <View style={[styles.liveDot, { backgroundColor: patientUid ? COLORS.green : COLORS.orange }]} />
            <Text style={styles.liveText}>{patientUid ? 'LIVE' : 'NO PATIENT'}</Text>
          </View>
        </View>
        <Text style={styles.brandText}>MOOVEFREE ✦</Text>
        <View style={styles.topRight}>
          <TouchableOpacity onPress={() => setShowConfig(true)} style={styles.iconBtn}>
            <Ionicons name="settings-outline" size={18} color={COLORS.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity onPress={() => { signOut(); navigation.replace('Login'); }} style={styles.iconBtn}>
            <Ionicons name="log-out-outline" size={18} color={COLORS.textSecondary} />
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>
        <View style={styles.videoSection}>
          <View style={styles.videoOverlayTop}>
            {aiOverlay && (
              <>
                {hazardLog.length > 0 && (
                  <View style={styles.hazardLabel}>
                    <Text style={styles.hazardLabelText}>HAZARD · {hazardLog[0]?.label?.toUpperCase() || 'DETECTED'}</Text>
                  </View>
                )}
                <View style={styles.personLabel}>
                  <Text style={styles.personLabelText}>PERSON</Text>
                </View>
              </>
            )}
            <View style={styles.aiToggleRow}>
              <Text style={styles.aiToggleText}>AI OVERLAY</Text>
              <Switch
                value={aiOverlay}
                onValueChange={setAiOverlay}
                trackColor={{ false: 'rgba(255,255,255,0.1)', true: 'rgba(0,229,255,0.4)' }}
                thumbColor={aiOverlay ? COLORS.cyan : '#555'}
                style={{ transform: [{ scaleX: 0.8 }, { scaleY: 0.8 }] }}
              />
            </View>
          </View>

          <WebView
            originWhitelist={['*']}
            source={{ html: streamHtml }}
            javaScriptEnabled
            scrollEnabled={false}
            style={styles.webview}
            cacheEnabled={false}
          />
        </View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>VITAL TELEMETRY</Text>
        </View>

        <View style={styles.vitalsRow}>
          <VitalCard
            icon="heart-outline"
            value={telemetry?.bpm || '--'}
            unit="BPM"
            color={COLORS.red}
          />
          <VitalCard
            icon="battery-charging-outline"
            value={telemetry?.battery ? `${Math.floor(telemetry.battery)}` : '--'}
            unit="%"
            color={COLORS.green}
          />
          <VitalCard
            icon="cellular-outline"
            value={telemetry?.signal || '4G'}
            unit="LT"
            color={COLORS.cyan}
          />
        </View>

        <View style={styles.progressBarContainer}>
          <View style={[styles.progressBarFill, { width: `${telemetry?.battery || 68}%` }]} />
        </View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>TACTICAL MAP</Text>
          <TouchableOpacity style={styles.expandBtn}>
            <Ionicons name="expand-outline" size={16} color={COLORS.textSecondary} />
          </TouchableOpacity>
        </View>

        <View style={styles.mapContainer}>
          <MapView
            ref={mapRef}
            style={styles.map}
            region={region}
            mapType="none"
            showsUserLocation={false}
            rotateEnabled={false}
          >
            <UrlTile
              urlTemplate="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
              subdomains="abcd"
              maximumZ={19}
              flipY={false}
            />
            {location && (
              <Marker coordinate={{ latitude: location.latitude, longitude: location.longitude }}>
                <View style={styles.markerOuter}>
                  <View style={styles.markerInner} />
                </View>
              </Marker>
            )}
            {geofence?.zones?.map((zone, i) => (
              <Circle
                key={i}
                center={{ latitude: zone.lat, longitude: zone.lng }}
                radius={zone.radius}
                strokeColor="rgba(0,229,255,0.5)"
                fillColor="rgba(0,229,255,0.06)"
                strokeWidth={1.5}
              />
            ))}
          </MapView>
        </View>

        <View style={styles.navBar}>
          <View style={styles.navLeft}>
            <Ionicons name="call-outline" size={18} color={COLORS.textSecondary} />
            <View>
              <Text style={styles.navLabel}>NEXT TURN</Text>
              <Text style={styles.navInstruction} numberOfLines={1}>
                {nextTurn?.instruction || 'No active navigation'}
              </Text>
            </View>
          </View>

          <TouchableOpacity
            style={[styles.sosButton, sos?.active && styles.sosButtonActive]}
            onPress={() => {
              if (sos?.active && patientUid) clearSOS(patientUid);
            }}
            activeOpacity={0.85}
          >
            <Animated.View style={{ opacity: sos?.active ? sosAnim : 1, flexDirection: 'row', alignItems: 'center', gap: 6 }}>
              <Text style={styles.sosButtonText}>✱  EMERGENCY SOS</Text>
            </Animated.View>
          </TouchableOpacity>

          <TouchableOpacity style={styles.expandBtn} onPress={() => setShowIncidents(true)}>
            <Ionicons name="list-outline" size={18} color={COLORS.textSecondary} />
          </TouchableOpacity>
        </View>
      </ScrollView>

      <Modal visible={showIncidents} animationType="slide" presentationStyle="pageSheet">
        <IncidentLogScreen onClose={() => setShowIncidents(false)} patientUid={patientUid} />
      </Modal>
      <Modal visible={showConfig} animationType="slide" presentationStyle="pageSheet">
        <RemoteConfigScreen onClose={() => setShowConfig(false)} patientUid={patientUid} />
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg },
  scroll: { paddingBottom: 16 },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 52,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  modeRow: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  modeText: { color: COLORS.cyan, fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  liveRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2 },
  liveDot: { width: 6, height: 6, borderRadius: 3 },
  liveText: { color: COLORS.textSecondary, fontSize: 10, letterSpacing: 1 },
  brandText: { fontSize: 15, fontWeight: '900', color: COLORS.white, letterSpacing: 2 },
  topRight: { flexDirection: 'row', gap: 4 },
  iconBtn: { padding: 6 },
  videoSection: {
    height: 200,
    marginHorizontal: 12,
    marginTop: 12,
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: COLORS.border,
    backgroundColor: '#000',
    position: 'relative',
  },
  webview: { flex: 1 },
  videoOverlayTop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 10,
    zIndex: 10,
  },
  personLabel: {
    backgroundColor: 'rgba(0,229,255,0.2)',
    borderWidth: 1,
    borderColor: COLORS.cyan,
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  personLabelText: { color: COLORS.cyan, fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  hazardLabel: {
    backgroundColor: 'rgba(255,59,48,0.25)',
    borderWidth: 1,
    borderColor: COLORS.red,
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
    position: 'absolute',
    bottom: 10,
    left: 10,
    zIndex: 11,
  },
  hazardLabelText: { color: COLORS.red, fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  aiToggleRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginLeft: 'auto' },
  aiToggleText: { color: COLORS.textSecondary, fontSize: 10, letterSpacing: 1 },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    marginTop: 16,
    marginBottom: 10,
  },
  sectionTitle: { color: COLORS.textSecondary, fontSize: 11, fontWeight: '700', letterSpacing: 2 },
  expandBtn: { padding: 4 },
  vitalsRow: { flexDirection: 'row', gap: 8, paddingHorizontal: 12 },
  progressBarContainer: {
    marginHorizontal: 12,
    marginTop: 10,
    height: 3,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBarFill: { height: 3, backgroundColor: COLORS.cyan, borderRadius: 2 },
  mapContainer: {
    height: 180,
    marginHorizontal: 12,
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  map: { flex: 1 },
  markerOuter: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: 'rgba(0,229,255,0.2)',
    borderWidth: 1.5,
    borderColor: COLORS.cyan,
    alignItems: 'center',
    justifyContent: 'center',
  },
  markerInner: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.cyan,
  },
  navBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 12,
    marginTop: 10,
    backgroundColor: COLORS.bgCard,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 12,
    gap: 10,
  },
  navLeft: { flexDirection: 'row', alignItems: 'center', gap: 8, flex: 1 },
  navLabel: { color: COLORS.textMuted, fontSize: 9, fontWeight: '700', letterSpacing: 1 },
  navInstruction: { color: COLORS.white, fontSize: 12, fontWeight: '600', maxWidth: 80 },
  sosButton: {
    backgroundColor: COLORS.red,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sosButtonActive: { backgroundColor: '#FF6B6B' },
  sosButtonText: { color: '#fff', fontWeight: '900', fontSize: 12, letterSpacing: 1 },
});
