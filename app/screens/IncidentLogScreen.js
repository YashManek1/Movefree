import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TextInput,
  TouchableOpacity,
  StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '../src/context/AppContext';
import { subscribeToHazardLog } from '../src/firebase/database';
import { COLORS } from '../src/colors';

const HAZARD_ICONS = {
  person: { icon: 'person-outline', color: COLORS.orange },
  car: { icon: 'car-outline', color: COLORS.red },
  stairs: { icon: 'arrow-up-outline', color: COLORS.red },
  door: { icon: 'enter-outline', color: COLORS.cyan },
  chair: { icon: 'cube-outline', color: COLORS.orange },
  default: { icon: 'warning-outline', color: COLORS.orange },
};

function getHazardStyle(label) {
  return HAZARD_ICONS[label?.toLowerCase()] || HAZARD_ICONS.default;
}

function HazardItem({ item }) {
  const { icon, color } = getHazardStyle(item.label);
  const timeStr = item.timestamp
    ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';
  const dateStr = item.timestamp
    ? new Date(item.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' })
    : '';

  return (
    <View style={styles.item}>
      <View style={[styles.iconBox, { backgroundColor: `${color}18`, borderColor: `${color}50` }]}>
        <Ionicons name={icon} size={20} color={color} />
      </View>
      <View style={styles.itemBody}>
        <Text style={styles.itemLabel}>{item.label?.toUpperCase() || 'HAZARD'}</Text>
        <Text style={styles.itemSub}>
          {item.distance ? `${item.distance.toFixed(1)}m away` : ''}
          {item.zone ? ` · ${item.zone}` : ''}
          {item.clock_dir ? ` · ${item.clock_dir}` : ''}
        </Text>
        {item.location && (
          <Text style={styles.itemLocation}>
            {item.location.latitude?.toFixed(5)}, {item.location.longitude?.toFixed(5)}
          </Text>
        )}
      </View>
      <View style={styles.timeCol}>
        <Text style={styles.itemTime}>{timeStr}</Text>
        <Text style={styles.itemDate}>{dateStr}</Text>
      </View>
    </View>
  );
}

export default function IncidentLogScreen({ onClose, patientUid: propPatientUid }) {
  const { patientUid: ctxPatientUid } = useApp();
  const pid = propPatientUid || ctxPatientUid;
  const [incidents, setIncidents] = useState([]);
  const [query, setQuery] = useState('');

  useEffect(() => {
    if (!pid) return;
    const u = subscribeToHazardLog(pid, setIncidents);
    return u;
  }, [pid]);

  const filtered = query.trim()
    ? incidents.filter((i) => i.label?.toLowerCase().includes(query.toLowerCase()))
    : incidents;

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name="list-outline" size={18} color={COLORS.cyan} />
          <Text style={styles.title}>INCIDENT LOG</Text>
          <Text style={styles.count}>{incidents.length}</Text>
        </View>
        {onClose && (
          <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
            <Ionicons name="close" size={22} color={COLORS.textSecondary} />
          </TouchableOpacity>
        )}
      </View>

      <View style={styles.searchBar}>
        <Ionicons name="search-outline" size={16} color={COLORS.textMuted} />
        <TextInput
          style={styles.searchInput}
          placeholder="Search hazard type..."
          placeholderTextColor={COLORS.textMuted}
          value={query}
          onChangeText={setQuery}
        />
        {query.length > 0 && (
          <TouchableOpacity onPress={() => setQuery('')}>
            <Ionicons name="close-circle" size={16} color={COLORS.textMuted} />
          </TouchableOpacity>
        )}
      </View>

      {filtered.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="checkmark-circle-outline" size={48} color={COLORS.green} />
          <Text style={styles.emptyTitle}>All Clear</Text>
          <Text style={styles.emptyText}>
            {query ? 'No matching incidents.' : 'No hazard incidents recorded yet.'}
          </Text>
        </View>
      ) : (
        <FlatList
          data={filtered}
          keyExtractor={(item) => item.id || String(item.timestamp)}
          renderItem={({ item }) => <HazardItem item={item} />}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
        />
      )}
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
  count: {
    backgroundColor: COLORS.cyanBg, borderRadius: 8, paddingHorizontal: 7, paddingVertical: 2,
    color: COLORS.cyan, fontSize: 11, fontWeight: '700',
  },
  closeBtn: { padding: 4 },
  searchBar: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    margin: 12, backgroundColor: COLORS.bgCard,
    borderRadius: 10, borderWidth: 1, borderColor: COLORS.border,
    paddingHorizontal: 14, paddingVertical: 10,
  },
  searchInput: { flex: 1, color: COLORS.white, fontSize: 14 },
  list: { padding: 12, gap: 8 },
  item: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: COLORS.bgCard, borderRadius: 12,
    borderWidth: 1, borderColor: COLORS.border, padding: 12,
  },
  iconBox: { width: 42, height: 42, borderRadius: 10, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  itemBody: { flex: 1, gap: 2 },
  itemLabel: { color: COLORS.white, fontSize: 13, fontWeight: '700', letterSpacing: 1 },
  itemSub: { color: COLORS.textSecondary, fontSize: 12 },
  itemLocation: { color: COLORS.textMuted, fontSize: 10, fontFamily: 'monospace', marginTop: 2 },
  timeCol: { alignItems: 'flex-end', gap: 2 },
  itemTime: { color: COLORS.white, fontSize: 12, fontWeight: '600' },
  itemDate: { color: COLORS.textMuted, fontSize: 10 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12, padding: 40 },
  emptyTitle: { color: COLORS.green, fontSize: 18, fontWeight: '700' },
  emptyText: { color: COLORS.textSecondary, fontSize: 13, textAlign: 'center' },
});
