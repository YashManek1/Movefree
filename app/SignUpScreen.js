import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StatusBar,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { signUp } from './src/firebase/auth';
import { useApp } from './src/context/AppContext';
import { COLORS } from './src/colors';

const ROLES = [
  { key: 'caretaker', label: 'Caretaker', icon: 'eye-outline', desc: 'Monitor and assist' },
  { key: 'visually_impaired', label: 'Visually Impaired', icon: 'accessibility-outline', desc: 'Navigate with AI' },
];

export default function SignUpScreen({ navigation }) {
  const { setUser } = useApp();
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [role, setRole] = useState('caretaker');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSignUp = async () => {
    if (!displayName.trim() || !email.trim() || !password || !confirmPassword) {
      Alert.alert('Missing Fields', 'Please fill in all fields.');
      return;
    }
    if (password !== confirmPassword) {
      Alert.alert('Password Mismatch', 'Passwords do not match.');
      return;
    }
    if (password.length < 6) {
      Alert.alert('Weak Password', 'Password must be at least 6 characters.');
      return;
    }
    setLoading(true);
    try {
      const profile = await signUp(email.trim(), password, role, displayName.trim());
      setUser(profile);
      if (role === 'visually_impaired') {
        navigation.replace('Transmitting');
      } else {
        navigation.replace('ConnectPatient');
      }
    } catch (err) {
      Alert.alert('Sign Up Failed', err.message || 'Could not create account.');
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
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="chevron-back" size={20} color={COLORS.cyan} />
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>

        <View style={styles.logoRow}>
          <View style={styles.logoDot} />
          <Text style={styles.logoText}>MOOVEFREE</Text>
        </View>

        <Text style={styles.title}>CREATE ACCOUNT</Text>
        <Text style={styles.subtitle}>Select your role to get started</Text>

        <View style={styles.roleRow}>
          {ROLES.map((r) => (
            <TouchableOpacity
              key={r.key}
              style={[styles.roleCard, role === r.key && styles.roleCardActive]}
              onPress={() => setRole(r.key)}
              activeOpacity={0.8}
            >
              <Ionicons
                name={r.icon}
                size={22}
                color={role === r.key ? '#000' : COLORS.textSecondary}
              />
              <Text style={[styles.roleLabel, role === r.key && styles.roleLabelActive]}>
                {r.label}
              </Text>
              <Text style={[styles.roleDesc, role === r.key && { color: '#000' }]}>{r.desc}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.card}>
          {[
            { icon: 'person-outline', placeholder: 'Display Name', value: displayName, onChange: setDisplayName, type: 'default', cap: 'words' },
            { icon: 'mail-outline', placeholder: 'Email address', value: email, onChange: setEmail, type: 'email-address', cap: 'none' },
          ].map((field) => (
            <View style={styles.inputGroup} key={field.placeholder}>
              <Ionicons name={field.icon} size={18} color={COLORS.cyan} style={styles.inputIcon} />
              <TextInput
                style={styles.input}
                placeholder={field.placeholder}
                placeholderTextColor={COLORS.textMuted}
                value={field.value}
                onChangeText={field.onChange}
                keyboardType={field.type}
                autoCapitalize={field.cap}
                autoCorrect={false}
              />
            </View>
          ))}

          <View style={styles.inputGroup}>
            <Ionicons name="lock-closed-outline" size={18} color={COLORS.cyan} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Password"
              placeholderTextColor={COLORS.textMuted}
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!showPassword}
            />
            <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.eyeBtn}>
              <Ionicons name={showPassword ? 'eye-outline' : 'eye-off-outline'} size={18} color={COLORS.textSecondary} />
            </TouchableOpacity>
          </View>

          <View style={styles.inputGroup}>
            <Ionicons name="lock-closed-outline" size={18} color={COLORS.cyan} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Confirm Password"
              placeholderTextColor={COLORS.textMuted}
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              secureTextEntry
            />
          </View>

          <TouchableOpacity
            style={[styles.registerBtn, loading && { opacity: 0.6 }]}
            onPress={handleSignUp}
            disabled={loading}
            activeOpacity={0.85}
          >
            {loading ? (
              <ActivityIndicator color="#000" />
            ) : (
              <>
                <Ionicons name="person-add-outline" size={18} color="#000" />
                <Text style={styles.registerBtnText}>CREATE ACCOUNT</Text>
              </>
            )}
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.loginLink} onPress={() => navigation.navigate('Login')}>
          <Text style={styles.loginLinkText}>
            Already have an account?{' '}
            <Text style={{ color: COLORS.cyan, fontWeight: '700' }}>Sign in →</Text>
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg },
  scroll: { flexGrow: 1, padding: 24, paddingTop: 60 },
  backBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, marginBottom: 32 },
  backText: { color: COLORS.cyan, fontSize: 14 },
  logoRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 24 },
  logoDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.cyan, marginRight: 10 },
  logoText: { fontSize: 20, fontWeight: '900', color: COLORS.cyan, letterSpacing: 4 },
  title: { fontSize: 24, fontWeight: '900', color: COLORS.white, letterSpacing: 3, marginBottom: 6 },
  subtitle: { fontSize: 13, color: COLORS.textSecondary, marginBottom: 24, letterSpacing: 0.5 },
  roleRow: { flexDirection: 'row', gap: 12, marginBottom: 20 },
  roleCard: {
    flex: 1,
    backgroundColor: COLORS.bgCard,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 14,
    alignItems: 'center',
    gap: 6,
  },
  roleCardActive: { backgroundColor: COLORS.cyan, borderColor: COLORS.cyan },
  roleLabel: { fontSize: 12, fontWeight: '700', color: COLORS.white, letterSpacing: 1 },
  roleLabelActive: { color: '#000' },
  roleDesc: { fontSize: 10, color: COLORS.textMuted, textAlign: 'center' },
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
  eyeBtn: { padding: 6 },
  registerBtn: {
    backgroundColor: COLORS.cyan,
    borderRadius: 10,
    paddingVertical: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 4,
  },
  registerBtnText: { color: '#000', fontWeight: '900', fontSize: 14, letterSpacing: 2 },
  loginLink: { alignItems: 'center', marginBottom: 32 },
  loginLinkText: { color: COLORS.textSecondary, fontSize: 14 },
});
