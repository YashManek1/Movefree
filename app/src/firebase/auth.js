import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut as firebaseSignOut,
  onAuthStateChanged,
} from '@firebase/auth';
import { ref, set, get } from 'firebase/database';
import { auth, db } from './config';

export const signUp = async (email, password, role, displayName) => {
  const credential = await createUserWithEmailAndPassword(auth, email, password);
  const uid = credential.user.uid;
  await set(ref(db, `users/${uid}`), {
    email: email.toLowerCase().trim(),
    role,
    displayName,
    createdAt: Date.now(),
  });
  return { uid, role, displayName, email };
};

export const signIn = async (email, password) => {
  const credential = await signInWithEmailAndPassword(auth, email, password);
  const uid = credential.user.uid;
  const snap = await get(ref(db, `users/${uid}`));
  if (!snap.exists()) throw new Error('User profile not found.');
  const data = snap.val();
  return { uid, role: data.role, displayName: data.displayName, email: data.email };
};

export const signOut = () => firebaseSignOut(auth);

export const getUserProfile = async (uid) => {
  const snap = await get(ref(db, `users/${uid}`));
  return snap.exists() ? snap.val() : null;
};

export const findPatientByEmail = async (email) => {
  const snap = await get(ref(db, 'users'));
  if (!snap.exists()) return null;
  const users = snap.val();
  for (const [uid, data] of Object.entries(users)) {
    if (
      data.email === email.toLowerCase().trim() &&
      data.role === 'visually_impaired'
    ) {
      return { uid, ...data };
    }
  }
  return null;
};

export const setCaretakerBinding = (caretakerUid, patientUid, patientName) =>
  set(ref(db, `bindings/${caretakerUid}`), { patientUid, patientName });

export const getCaretakerBinding = async (caretakerUid) => {
  const snap = await get(ref(db, `bindings/${caretakerUid}`));
  return snap.exists() ? snap.val() : null;
};

export const onAuthChanged = (callback) => onAuthStateChanged(auth, callback);
