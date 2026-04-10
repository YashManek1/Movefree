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
  Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { signIn } from './src/firebase/auth';
import { getCaretakerBinding } from './src/firebase/auth';
import { useApp } from './src/context/AppContext';
import { COLORS } from './src/colors';

export default function LoginScreen({ navigation }) {
  const { setUser, setPatientUid, setPatientName } = useApp();
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading]   = useState(false);

  const handleLogin = async () => {
    if (!email.trim() || !password.trim()) {
      Alert.alert('Missing Fields', 'Please enter your email and password.');
      return;
    }
    setLoading(true);
    try {
      const profile = await signIn(email.trim(), password);
      setUser(profile);
      if (profile.role === 'visually_impaired') {
        navigation.replace('Transmitting');
      } else if (profile.role === 'caretaker') {
        const binding = await getCaretakerBinding(profile.uid);
        if (binding?.patientUid) {
          setPatientUid(binding.patientUid);
          setPatientName(binding.patientName || '');
          navigation.replace('CaretakerHome');
        } else {
          navigation.replace('ConnectPatient');
        }
      }
    } catch (err) {
      Alert.alert('Login Failed', err.message || 'Invalid credentials.');
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

        {}
        <View style={styles.logoBlock}>
          <Image
            source={require('./assets/icon.png')}
            style={styles.logoImage}
            resizeMode="contain"
          />
          <Text style={styles.tagline}>Navigate the world. With AI.</Text>
        </View>

        {}
        <View style={styles.dividerRow}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerText}>SIGN IN</Text>
          <View style={styles.dividerLine} />
        </View>

        {}
        <View style={styles.card}>
          <View style={styles.inputGroup}>
            <Ionicons name="mail-outline" size={18} color={COLORS.primary} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Email address"
              placeholderTextColor={COLORS.textMuted}
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>

          <View style={styles.inputGroup}>
            <Ionicons name="lock-closed-outline" size={18} color={COLORS.primary} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Password"
              placeholderTextColor={COLORS.textMuted}
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!showPass}
            />
            <TouchableOpacity onPress={() => setShowPass(!showPass)} style={styles.eyeBtn}>
              <Ionicons
                name={showPass ? 'eye-outline' : 'eye-off-outline'}
                size={18}
                color={COLORS.textSecondary}
              />
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            style={[styles.loginBtn, loading && styles.loginBtnDisabled]}
            onPress={handleLogin}
            disabled={loading}
            activeOpacity={0.85}
          >
            {loading ? (
              <ActivityIndicator color={COLORS.bg} />
            ) : (
              <>
                <Ionicons name="log-in-outline" size={18} color={COLORS.bg} />
                <Text style={styles.loginBtnText}>SIGN IN</Text>
              </>
            )}
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={styles.signupLink}
          onPress={() => navigation.navigate('SignUp')}
        >
          <Text style={styles.signupLinkText}>
            No account?{' '}
            <Text style={{ color: COLORS.gold, fontWeight: '700' }}>Create one →</Text>
          </Text>
        </TouchableOpacity>

        <View style={styles.footerRow}>
          <View style={styles.footerLine} />
          <Text style={styles.footerText}>Powered by Gemini AI</Text>
          <View style={styles.footerLine} />
        </View>

      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg },
  scroll: { flexGrow: 1, justifyContent: 'center', paddingHorizontal: 24, paddingVertical: 40 },

  logoBlock: { alignItems: 'center', marginBottom: 36 },
  logoImage: { width: 160, height: 160 },
  tagline: {
    marginTop: 8, color: COLORS.textSecondary,
    fontSize: 13, letterSpacing: 1, textAlign: 'center',
  },

  dividerRow: {
    flexDirection: 'row', alignItems: 'center',
    marginBottom: 24, gap: 12,
  },
  dividerLine: { flex: 1, height: 1, backgroundColor: COLORS.border },
  dividerText: {
    color: COLORS.primary, fontSize: 11,
    fontWeight: '700', letterSpacing: 3,
  },

  card: {
    backgroundColor: COLORS.bgCard,
    borderRadius: 16, borderWidth: 1,
    borderColor: COLORS.border, padding: 20, marginBottom: 20,
  },
  inputGroup: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: COLORS.bg, borderRadius: 10,
    borderWidth: 1, borderColor: COLORS.border,
    marginBottom: 14, paddingHorizontal: 12,
  },
  inputIcon: { marginRight: 10 },
  input: {
    flex: 1, color: COLORS.white, fontSize: 14,
    paddingVertical: 14,
  },
  eyeBtn: { padding: 6 },
  loginBtn: {
    backgroundColor: COLORS.primary, borderRadius: 10,
    paddingVertical: 14, flexDirection: 'row',
    alignItems: 'center', justifyContent: 'center',
    marginTop: 4, gap: 8,
  },
  loginBtnDisabled: { opacity: 0.6 },
  loginBtnText: {
    color: COLORS.bg, fontWeight: '900',
    fontSize: 14, letterSpacing: 2,
  },

  signupLink: { alignItems: 'center', marginBottom: 32 },
  signupLinkText: { color: COLORS.textSecondary, fontSize: 14 },
  footerRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  footerLine: { flex: 1, height: 1, backgroundColor: COLORS.border },
  footerText: { color: COLORS.textMuted, fontSize: 11, letterSpacing: 1 },
});
