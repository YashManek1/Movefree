import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Alert,
} from 'react-native';
import MapView, { Marker, Circle, UrlTile } from 'react-native-maps';
import Slider from '@react-native-community/slider';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '../src/context/AppContext';
import {
  subscribeToLocation,
  subscribeToGeofence,
  pushGeofenceZones,
} from '../src/firebase/database';
import { COLORS } from '../src/colors';

export default function MapScreen() {
  const { patientUid } = useApp();
  const [location, setLocation] = useState(null);
  const [zones, setZones] = useState([]);
  const [pendingZone, setPendingZone] = useState(null);
  const [radius, setRadius] = useState(100);
  const mapRef = useRef(null);

  useEffect(() => {
    if (!patientUid) return;
    const u1 = subscribeToLocation(patientUid, setLocation);
    const u2 = subscribeToGeofence(patientUid, (data) => {
      if (data?.zones) setZones(data.zones);
    });
    return () => { u1(); u2(); };
  }, [patientUid]);

  const onMapPress = (e) => {
    const { latitude, longitude } = e.nativeEvent.coordinate;
    setPendingZone({ lat: latitude, lng: longitude });
  };

  const confirmZone = async () => {
    if (!pendingZone) return;
    const updated = [...zones, { lat: pendingZone.lat, lng: pendingZone.lng, radius }];
    setZones(updated);
    setPendingZone(null);
    if (patientUid) await pushGeofenceZones(patientUid, updated);
  };

  const deleteZone = (index) => {
    Alert.alert('Remove Zone', 'Delete this geofence zone?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          const updated = zones.filter((_, i) => i !== index);
          setZones(updated);
          if (patientUid) await pushGeofenceZones(patientUid, updated);
        },
      },
    ]);
  };

  const region = location
    ? { latitude: location.latitude, longitude: location.longitude, latitudeDelta: 0.01, longitudeDelta: 0.01 }
    : { latitude: 19.076, longitude: 72.8777, latitudeDelta: 0.05, longitudeDelta: 0.05 };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name="map-outline" size={18} color={COLORS.cyan} />
          <Text style={styles.title}>GEOFENCE ZONES</Text>
        </View>
        <Text style={styles.subtitle}>Tap the map to add a safe zone boundary</Text>
      </View>

      <MapView
        ref={mapRef}
        style={styles.map}
        region={region}
        mapType="none"
        onPress={onMapPress}
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
            <View style={styles.patientMarker}>
              <View style={styles.patientMarkerInner} />
            </View>
          </Marker>
        )}
        {zones.map((zone, i) => (
          <React.Fragment key={i}>
            <Circle
              center={{ latitude: zone.lat, longitude: zone.lng }}
              radius={zone.radius}
              strokeColor="rgba(0,229,255,0.6)"
              fillColor="rgba(0,229,255,0.07)"
              strokeWidth={1.5}
            />
            <Marker
              coordinate={{ latitude: zone.lat, longitude: zone.lng }}
              onPress={() => deleteZone(i)}
            >
              <View style={styles.zoneMarker}>
                <Ionicons name="shield-checkmark-outline" size={14} color={COLORS.cyan} />
              </View>
            </Marker>
          </React.Fragment>
        ))}
        {pendingZone && (
          <>
            <Circle
              center={{ latitude: pendingZone.lat, longitude: pendingZone.lng }}
              radius={radius}
              strokeColor="rgba(255,159,10,0.8)"
              fillColor="rgba(255,159,10,0.1)"
              strokeWidth={2}
              strokeDash={[5, 5]}
            />
            <Marker coordinate={{ latitude: pendingZone.lat, longitude: pendingZone.lng }}>
              <View style={styles.pendingMarker}>
                <Ionicons name="add-circle-outline" size={18} color={COLORS.orange} />
              </View>
            </Marker>
          </>
        )}
      </MapView>

      {pendingZone && (
        <View style={styles.zonePanel}>
          <Text style={styles.zonePanelTitle}>NEW SAFE ZONE</Text>
          <View style={styles.sliderRow}>
            <Text style={styles.sliderLabel}>Radius</Text>
            <Text style={styles.sliderValue}>{radius}m</Text>
          </View>
          <Slider
            style={styles.slider}
            minimumValue={50}
            maximumValue={1000}
            step={10}
            value={radius}
            onValueChange={setRadius}
            minimumTrackTintColor={COLORS.cyan}
            maximumTrackTintColor="rgba(255,255,255,0.1)"
            thumbTintColor={COLORS.cyan}
          />
          <View style={styles.zoneBtns}>
            <TouchableOpacity style={styles.cancelZoneBtn} onPress={() => setPendingZone(null)}>
              <Text style={styles.cancelZoneBtnText}>CANCEL</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.confirmZoneBtn} onPress={confirmZone}>
              <Ionicons name="checkmark-outline" size={16} color="#000" />
              <Text style={styles.confirmZoneBtnText}>SAVE ZONE</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {!patientUid && (
        <View style={styles.noPatient}>
          <Text style={styles.noPatientText}>No patient connected</Text>
        </View>
      )}

      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: COLORS.cyan }]} />
          <Text style={styles.legendText}>{zones.length} zone{zones.length !== 1 ? 's' : ''} active</Text>
        </View>
        <Text style={styles.legendHint}>Tap a zone pin to delete</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg },
  header: { paddingHorizontal: 16, paddingTop: 56, paddingBottom: 10, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4 },
  title: { fontSize: 14, fontWeight: '900', color: COLORS.white, letterSpacing: 3 },
  subtitle: { color: COLORS.textSecondary, fontSize: 12 },
  map: { flex: 1 },
  patientMarker: {
    width: 20, height: 20, borderRadius: 10,
    backgroundColor: 'rgba(0,229,255,0.2)', borderWidth: 1.5, borderColor: COLORS.cyan,
    alignItems: 'center', justifyContent: 'center',
  },
  patientMarkerInner: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.cyan },
  zoneMarker: {
    backgroundColor: COLORS.bgCard, borderRadius: 8, padding: 4,
    borderWidth: 1, borderColor: COLORS.borderStrong,
  },
  pendingMarker: {
    backgroundColor: 'rgba(255,159,10,0.15)', borderRadius: 8, padding: 4,
    borderWidth: 1, borderColor: COLORS.orange,
  },
  zonePanel: {
    position: 'absolute', bottom: 80, left: 12, right: 12,
    backgroundColor: COLORS.bgCard, borderRadius: 16,
    borderWidth: 1, borderColor: COLORS.border, padding: 16,
  },
  zonePanelTitle: { color: COLORS.orange, fontSize: 11, fontWeight: '700', letterSpacing: 2, marginBottom: 12 },
  sliderRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  sliderLabel: { color: COLORS.textSecondary, fontSize: 12 },
  sliderValue: { color: COLORS.white, fontSize: 12, fontWeight: '700' },
  slider: { width: '100%', height: 32, marginBottom: 12 },
  zoneBtns: { flexDirection: 'row', gap: 10 },
  cancelZoneBtn: { flex: 1, borderWidth: 1, borderColor: COLORS.border, borderRadius: 10, paddingVertical: 12, alignItems: 'center' },
  cancelZoneBtnText: { color: COLORS.textSecondary, fontSize: 12, fontWeight: '700', letterSpacing: 1 },
  confirmZoneBtn: {
    flex: 2, backgroundColor: COLORS.cyan, borderRadius: 10,
    paddingVertical: 12, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
  },
  confirmZoneBtnText: { color: '#000', fontSize: 13, fontWeight: '900', letterSpacing: 1 },
  noPatient: { position: 'absolute', top: 110, alignSelf: 'center', backgroundColor: COLORS.bgCard, borderRadius: 10, padding: 12, borderWidth: 1, borderColor: COLORS.border },
  noPatientText: { color: COLORS.textSecondary, fontSize: 12 },
  legend: {
    position: 'absolute', bottom: 16, left: 16, right: 16,
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    backgroundColor: COLORS.bgCard, borderRadius: 10,
    borderWidth: 1, borderColor: COLORS.border, padding: 10,
  },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  legendDot: { width: 8, height: 8, borderRadius: 4 },
  legendText: { color: COLORS.white, fontSize: 12, fontWeight: '600' },
  legendHint: { color: COLORS.textMuted, fontSize: 11 },
});
