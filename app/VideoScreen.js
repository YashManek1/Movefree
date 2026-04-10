import React, { useEffect, useMemo, useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { STREAM_URL } from './config';
import { WebView } from 'react-native-webview';

export default function VideoScreen({ navigation, onSOSDetected }) {
  const [loading, setLoading] = useState(true);
  const [sosTriggered, setSosTriggered] = useState(false);

  const screenWidth = Dimensions.get('window').width;
  const videoWidth = screenWidth - 40;
  const videoHeight = (videoWidth * 3) / 4;

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 4000);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    let sosTimer;

    if (loading && STREAM_URL && !sosTriggered) {
      sosTimer = setTimeout(async () => {
        await triggerSOSOnce();
      }, 8000);
    }

    return () => {
      if (sosTimer) clearTimeout(sosTimer);
    };
  }, [loading, sosTriggered]);

  const streamHtml = useMemo(
    () => `
      <!DOCTYPE html>
      <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
          <style>
            html, body {
              margin: 0;
              padding: 0;
              width: 100%;
              height: 100%;
              background: black;
              overflow: hidden;
            }
            .container {
              width: 100%;
              height: 100%;
              display: flex;
              align-items: center;
              justify-content: center;
              background: black;
              color: white;
              font-family: sans-serif;
            }
            img {
              width: 100%;
              height: 100%;
              object-fit: contain;
              background: black;
            }
            .message {
              text-align: center;
              padding: 16px;
              font-size: 16px;
            }
          </style>
        </head>
        <body>
          <div class="container">
            ${
              STREAM_URL
                ? `<img 
                    src="${STREAM_URL}" 
                    onload="window.ReactNativeWebView.postMessage('loaded')" 
                    onerror="window.ReactNativeWebView.postMessage('stream-error')" 
                  />`
                : `<div class="message">No stream URL configured.<br/>Set STREAM_URL locally before running.</div>`
            }
          </div>
        </body>
      </html>
    `,
    []
  );

  const triggerSOSOnce = async () => {
    if (!sosTriggered) {
      setSosTriggered(true);
      if (onSOSDetected) {
        await onSOSDetected();
      }
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Live Video Stream</Text>

      <View style={[styles.webViewContainer, { width: videoWidth, height: videoHeight }]}>
        {loading && STREAM_URL ? (
          <View style={styles.loaderOverlay}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>Loading stream...</Text>
          </View>
        ) : null}

        <WebView
          originWhitelist={['*']}
          source={{ html: streamHtml }}
          javaScriptEnabled
          domStorageEnabled
          scrollEnabled={false}
          onMessage={async (event) => {
            const message = event.nativeEvent.data;

            if (message === 'loaded') {
              setLoading(false);
            }

            if (message === 'stream-error') {
              setLoading(false);
              await triggerSOSOnce();
            }
          }}
          onError={async () => {
            setLoading(false);
            await triggerSOSOnce();
          }}
          style={styles.webview}
        />
      </View>

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Location')}
      >
        <Text style={styles.buttonText}>Show Live Location</Text>
      </TouchableOpacity>

      {sosTriggered && (
        <Text style={styles.sosText}>
          Emergency alert triggered due to video obstruction / stream failure.
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F6F8FA',
    padding: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 20,
    textAlign: 'center',
  },
  webViewContainer: {
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#D1D5DB',
    marginBottom: 16,
    backgroundColor: '#000',
    position: 'relative',
  },
  webview: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  loaderOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 10,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(246, 248, 250, 0.85)',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 14,
    color: '#374151',
  },
  button: {
    backgroundColor: '#007AFF',
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 20,
    alignItems: 'center',
    marginTop: 10,
    width: '100%',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  sosText: {
    marginTop: 14,
    color: '#DC2626',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
});