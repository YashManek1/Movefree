import React, { useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  Linking,
  Alert,
} from 'react-native';

export default function LocationScreen({
  location,
  loadingLocation,
  onRequestLocation,
  navigation,
}) {
  useEffect(() => {
    onRequestLocation();
  }, []);

  const openInGoogleMaps = async () => {
    if (!location) {
      Alert.alert('No location', 'Location is not available yet.');
      return;
    }

    const url = `https://www.google.com/maps/search/?api=1&query=${location.latitude},${location.longitude}`;

    const supported = await Linking.canOpenURL(url);
    if (supported) {
      await Linking.openURL(url);
    } else {
      Alert.alert('Error', 'Unable to open Google Maps.');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Live Location</Text>

      {loadingLocation ? (
        <ActivityIndicator size="large" color="#007AFF" />
      ) : location ? (
        <View style={styles.locationBox}>
          <Text style={styles.locationText}>
            Latitude: {location.latitude}
          </Text>
          <Text style={styles.locationText}>
            Longitude: {location.longitude}
          </Text>
          <Text style={styles.locationText}>
            Accuracy: {location.accuracy ? `${Math.round(location.accuracy)} meters` : 'N/A'}
          </Text>
        </View>
      ) : (
        <Text style={styles.infoText}>
          Grant location permission and enable device location, then tap Refresh.
        </Text>
      )}

      <TouchableOpacity style={styles.button} onPress={onRequestLocation}>
        <Text style={styles.buttonText}>Refresh Location</Text>
      </TouchableOpacity>

      {location && (
        <TouchableOpacity style={styles.mapButton} onPress={openInGoogleMaps}>
          <Text style={styles.buttonText}>Open in Google Maps</Text>
        </TouchableOpacity>
      )}

      <TouchableOpacity
        style={[styles.button, styles.secondaryButton]}
        onPress={() => navigation.navigate('Video')}
      >
        <Text style={[styles.buttonText, styles.secondaryButtonText]}>
          Back to Video
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F6F8FA',
    padding: 20,
    justifyContent: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 20,
    textAlign: 'center',
  },
  locationBox: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 20,
    borderWidth: 1,
    borderColor: '#D1D5DB',
    marginBottom: 16,
  },
  locationText: {
    fontSize: 16,
    marginBottom: 8,
  },
  infoText: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
    color: '#4B5563',
  },
  button: {
    backgroundColor: '#007AFF',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 10,
  },
  mapButton: {
    backgroundColor: '#34A853',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    backgroundColor: '#E5E7EB',
  },
  secondaryButtonText: {
    color: '#111827',
  },
});