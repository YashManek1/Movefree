import React, { useState, useCallback } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Alert } from 'react-native';
import * as Location from 'expo-location';
import LoginScreen from './LoginScreen';
import VideoScreen from './VideoScreen';
import LocationScreen from './LocationScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  const [name, setName] = useState('');
  const [contact, setContact] = useState('');
  const [location, setLocation] = useState(null);
  const [loadingLocation, setLoadingLocation] = useState(false);

  const handleLogin = (navigation) => {
    if (!name.trim() || !contact.trim()) {
      Alert.alert('Please fill the fields', 'Enter both username and contact number.');
      return;
    }
    navigation.navigate('Video');
  };

  const requestLocation = useCallback(async () => {
    setLoadingLocation(true);

    try {
      // Check if GPS / location services are ON
      const servicesEnabled = await Location.hasServicesEnabledAsync();
      if (!servicesEnabled) {
        Alert.alert('Location services off', 'Please enable GPS / Location Services on your device.');
        setLocation(null);
        return;
      }

      // Ask for permission
      const { status } = await Location.requestForegroundPermissionsAsync();

      if (status !== 'granted') {
        Alert.alert('Permission required', 'Allow location access to see live location.');
        setLocation(null);
        return;
      }

      // Fetch current location
      const currentLocation = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
        mayShowUserSettingsDialog: true,
      });

      setLocation({
        latitude: currentLocation.coords.latitude,
        longitude: currentLocation.coords.longitude,
        accuracy: currentLocation.coords.accuracy,
      });
    } catch (error) {
      console.log('Location error:', error);
      Alert.alert('Location error', error.message || 'Unable to fetch location.');
      setLocation(null);
    } finally {
      setLoadingLocation(false);
    }
  }, []);

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Login">
        <Stack.Screen name="Login" options={{ title: 'Login' }}>
          {(props) => (
            <LoginScreen
              {...props}
              name={name}
              contact={contact}
              setName={setName}
              setContact={setContact}
              onLogin={() => handleLogin(props.navigation)}
            />
          )}
        </Stack.Screen>

        <Stack.Screen name="Video" options={{ title: 'Live Video' }}>
          {(props) => <VideoScreen {...props} />}
        </Stack.Screen>

        <Stack.Screen name="Location" options={{ title: 'Live Location' }}>
          {(props) => (
            <LocationScreen
              {...props}
              location={location}
              loadingLocation={loadingLocation}
              onRequestLocation={requestLocation}
            />
          )}
        </Stack.Screen>
      </Stack.Navigator>
    </NavigationContainer>
  );
}