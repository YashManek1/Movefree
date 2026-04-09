# MooveFree

MooveFree is an assistive navigation system designed to improve mobility for visually impaired users. It combines real-time sensing, edge AI, and intuitive feedback to help users move safely in dynamic environments.

MooveFree uses a hybrid sensing approach to detect obstacles and guide users.
The system delivers clear navigation feedback through audio and spatial cues.

![Moovefree System Flow](https://raw.githubusercontent.com/YashManek1/Movefree/refs/heads/main/assets/system_flow.png)

---

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

---

## Configuration

Create a config file at `app/config.js` for environment variables:

```js
export const STREAM_URL = "http://your-stream-url";
```

---

## Contributors

Made with ❤️
