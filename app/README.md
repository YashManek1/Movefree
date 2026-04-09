# MooveFree App

A simple React Native Expo app with:

- Login screen
- Main screen showing a live video stream
- Location screen showing the device's live location

## Setup

1. Open a terminal in `/app` and install dependencies:

```bash
npm install
```

2. Start Expo:

```bash
npm start
```

## Run in Expo Go (Android)

1. Install the Expo Go app on your Android device.
2. Make sure your device is on the same Wi-Fi network as your development machine.
3. In the terminal, run:

```bash
npx expo start
```

4. Scan the QR code shown in the Expo DevTools with Expo Go.

If your phone cannot reach the local video URL directly, run Expo with tunnel mode:

```bash
npx expo start --tunnel
```

## Notes

- The video screen loads content from `http://{STREAM_URL}/video`.
- The location screen requests foreground location permission and shows latitude, longitude, and accuracy.
