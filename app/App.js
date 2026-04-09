import React, { useState } from 'react';
import { Alert, Vibration } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
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

  // Fetch current location for LocationScreen
  const requestLocation = async () => {
    try {
      setLoadingLocation(true);

      const { status } = await Location.requestForegroundPermissionsAsync();

      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permission is required.');
        setLoadingLocation(false);
        return;
      }

      const currentLocation = await Location.getCurrentPositionAsync({});
      setLocation({
        latitude: currentLocation.coords.latitude.toFixed(6),
        longitude: currentLocation.coords.longitude.toFixed(6),
        accuracy: currentLocation.coords.accuracy,
      });
    } catch (error) {
      Alert.alert('Location Error', 'Unable to fetch location.');
    } finally {
      setLoadingLocation(false);
    }
  };

  // Emergency alert when video is obstructed
  const handleSOSDetected = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();

      if (status !== 'granted') {
        Vibration.vibrate([500, 200, 500, 200, 500]);

        Alert.alert(
          'Emergency Alert',
          'Emergency, Video Obstructed Current Location : Permission Denied'
        );
        return;
      }

      const currentLocation = await Location.getCurrentPositionAsync({});
      const latitude = currentLocation.coords.latitude.toFixed(6);
      const longitude = currentLocation.coords.longitude.toFixed(6);

      Vibration.vibrate([500, 200, 500, 200, 500]);

      Alert.alert(
        'Emergency Alert',
        `Emergency, Video Obstructed Current Location : ${latitude}, ${longitude}`
      );
    } catch (error) {
      Vibration.vibrate([500, 200, 500, 200, 500]);

      Alert.alert(
        'Emergency Alert',
        'Emergency, Video Obstructed Current Location : Unable to fetch location'
      );
    }
  };

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Login">
        {/* LOGIN SCREEN */}
        <Stack.Screen name="Login" options={{ title: 'Login' }}>
          {(props) => (
            <LoginScreen
              {...props}
              name={name}
              contact={contact}
              setName={setName}
              setContact={setContact}
              onLogin={() => {
                if (!name.trim() || !contact.trim()) {
                  Alert.alert('Validation Error', 'Please enter both name and contact.');
                  return;
                }

                props.navigation.navigate('Video');
              }}
            />
          )}
        </Stack.Screen>

        {/* VIDEO SCREEN */}
        <Stack.Screen name="Video" options={{ title: 'Live Video Stream' }}>
          {(props) => (
            <VideoScreen
              {...props}
              onSOSDetected={handleSOSDetected}
            />
          )}
        </Stack.Screen>

        {/* LOCATION SCREEN */}
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