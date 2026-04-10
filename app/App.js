import React, { useState, useEffect } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { onAuthChanged, getUserProfile, getCaretakerBinding } from './src/firebase/auth';
import { AppProvider, useApp } from './src/context/AppContext';
import { COLORS } from './src/colors';

import LoginScreen from './LoginScreen';
import SignUpScreen from './SignUpScreen';
import TransmittingScreen from './screens/TransmittingScreen';
import CaretakerNavigator from './screens/CaretakerNavigator';
import ConnectPatientScreen from './screens/ConnectPatientScreen';

const Stack = createNativeStackNavigator();

function RootNavigator() {
  const { user, setUser, setPatientUid, setPatientName } = useApp();
  const [initializing, setInitializing] = useState(true);
  const [initialRoute, setInitialRoute] = useState('Login');

  useEffect(() => {
    const unsubscribe = onAuthChanged(async (firebaseUser) => {
      if (firebaseUser) {
        try {
          const profile = await getUserProfile(firebaseUser.uid);
          if (!profile) {
            setUser(null);
            setInitialRoute('Login');
            setInitializing(false);
            return;
          }
          const fullUser = { uid: firebaseUser.uid, ...profile };
          setUser(fullUser);

          if (profile.role === 'visually_impaired') {
            setInitialRoute('Transmitting');
          } else if (profile.role === 'caretaker') {
            const binding = await getCaretakerBinding(firebaseUser.uid);
            if (binding && binding.patientUid) {
              setPatientUid(binding.patientUid);
              setPatientName(binding.patientName || '');
              setInitialRoute('CaretakerHome');
            } else {
              setInitialRoute('ConnectPatient');
            }
          }
        } catch {
          setUser(null);
          setInitialRoute('Login');
        }
      } else {
        setUser(null);
        setInitialRoute('Login');
      }
      setInitializing(false);
    });
    return unsubscribe;
  }, []);

  if (initializing) {
    return (
      <View style={styles.loader}>
        <ActivityIndicator size="large" color={COLORS.cyan} />
      </View>
    );
  }

  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName={initialRoute}
        screenOptions={{ headerShown: false, animation: 'fade' }}
      >
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="SignUp" component={SignUpScreen} />
        <Stack.Screen name="ConnectPatient" component={ConnectPatientScreen} />
        <Stack.Screen name="Transmitting" component={TransmittingScreen} />
        <Stack.Screen name="CaretakerHome" component={CaretakerNavigator} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <AppProvider>
      <RootNavigator />
    </AppProvider>
  );
}

const styles = StyleSheet.create({
  loader: {
    flex: 1,
    backgroundColor: COLORS.bg,
    justifyContent: 'center',
    alignItems: 'center',
  },
});