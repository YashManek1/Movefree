import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { findPatientByEmail, setCaretakerBinding } from '../src/firebase/auth';
import { useApp } from '../src/context/AppContext';
import { COLORS } from '../src/colors';

export default function ConnectPatientScreen({ navigation }) {
  const { user, setPatientUid, setPatientName } = useApp();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [found, setFound] = useState(null);

  const searchPatient = async () => {
    if (!email.trim()) return;
    setLoading(true);
    setFound(null);
    try {
      const patient = await findPatientByEmail(email.trim());
      if (!patient) {
        Alert.alert('Not Found', 'No visually impaired user with that email.');
      } else {
        setFound(patient);
      }
    } catch (err) {
      Alert.alert('Error', err.message);
    } finally {
      setLoading(false);
    }
  };

  const connectPatient = async () => {
    if (!found) return;
    setLoading(true);
    try {
      await setCaretakerBinding(user.uid, found.uid, found.displayName);
      setPatientUid(found.uid);
      setPatientName(found.displayName || '');
      navigation.replace('CaretakerHome');
    } catch (err) {
      Alert.alert('Connection Failed', err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      <View style={styles.header}>
        <View style={styles.logoDot} />
        <Text style={styles.logoText}>MOOVEFREE</Text>
      </View>

      <Text style={styles.title}>CONNECT PATIENT</Text>
      <Text style={styles.subtitle}>
        Enter the email of the visually impaired user you want to monitor.
      </Text>

      <View style={styles.card}>
        <View style={styles.inputGroup}>
          <Ionicons name="search-outline" size={18} color={COLORS.cyan} style={styles.inputIcon} />
          <TextInput
            style={styles.input}
            placeholder="Patient email address"
            placeholderTextColor={COLORS.textMuted}
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
          />
        </View>

        <TouchableOpacity
          style={[styles.searchBtn, loading && { opacity: 0.6 }]}
          onPress={searchPatient}
          disabled={loading}
          activeOpacity={0.85}
        >
          {loading ? (
            <ActivityIndicator color={COLORS.cyan} />
          ) : (
            <>
              <Ionicons name="person-search-outline" size={18} color={COLORS.cyan} />
              <Text style={styles.searchBtnText}>SEARCH</Text>
            </>
          )}
        </TouchableOpacity>
      </View>

      {found && (
        <View style={styles.resultCard}>
          <View style={styles.resultRow}>
            <View style={styles.resultAvatar}>
              <Ionicons name="accessibility" size={24} color={COLORS.cyan} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.resultName}>{found.displayName}</Text>
              <Text style={styles.resultEmail}>{found.email}</Text>
              <View style={styles.rolePill}>
                <Text style={styles.rolePillText}>VISUALLY IMPAIRED</Text>
              </View>
            </View>
          </View>

          <TouchableOpacity
            style={[styles.connectBtn, loading && { opacity: 0.6 }]}
            onPress={connectPatient}
            disabled={loading}
            activeOpacity={0.85}
          >
            {loading ? (
              <ActivityIndicator color="#000" />
            ) : (
              <>
                <Ionicons name="link-outline" size={18} color="#000" />
                <Text style={styles.connectBtnText}>CONNECT & MONITOR</Text>
              </>
            )}
          </TouchableOpacity>
        </View>
      )}
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg, padding: 24, paddingTop: 60 },
  header: { flexDirection: 'row', alignItems: 'center', marginBottom: 40 },
  logoDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.cyan, marginRight: 10 },
  logoText: { fontSize: 20, fontWeight: '900', color: COLORS.cyan, letterSpacing: 4 },
  title: { fontSize: 24, fontWeight: '900', color: COLORS.white, letterSpacing: 3, marginBottom: 8 },
  subtitle: { fontSize: 13, color: COLORS.textSecondary, marginBottom: 28, lineHeight: 20 },
  card: {
    backgroundColor: COLORS.bgCard,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 20,
    marginBottom: 20,
  },
  inputGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.bg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: 14,
    paddingHorizontal: 12,
  },
  inputIcon: { marginRight: 10 },
  input: { flex: 1, color: COLORS.white, fontSize: 14, paddingVertical: 14 },
  searchBtn: {
    borderWidth: 1.5,
    borderColor: COLORS.cyan,
    borderRadius: 10,
    paddingVertical: 13,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  searchBtnText: { color: COLORS.cyan, fontWeight: '700', fontSize: 13, letterSpacing: 2 },
  resultCard: {
    backgroundColor: COLORS.bgCard,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.borderStrong,
    padding: 20,
    gap: 16,
  },
  resultRow: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  resultAvatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: COLORS.cyanBg,
    borderWidth: 1,
    borderColor: COLORS.borderStrong,
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultName: { fontSize: 16, fontWeight: '700', color: COLORS.white, marginBottom: 2 },
  resultEmail: { fontSize: 12, color: COLORS.textSecondary, marginBottom: 6 },
  rolePill: {
    backgroundColor: COLORS.cyanBg,
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 3,
    alignSelf: 'flex-start',
  },
  rolePillText: { color: COLORS.cyan, fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  connectBtn: {
    backgroundColor: COLORS.cyan,
    borderRadius: 10,
    paddingVertical: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  connectBtnText: { color: '#000', fontWeight: '900', fontSize: 13, letterSpacing: 2 },
});
