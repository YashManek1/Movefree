import React, { createContext, useContext, useState } from 'react';

const AppContext = createContext({
  user: null,
  setUser: () => {},
  patientUid: null,
  setPatientUid: () => {},
  patientName: '',
  setPatientName: () => {},
});

export const AppProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [patientUid, setPatientUid] = useState(null);
  const [patientName, setPatientName] = useState('');

  return (
    <AppContext.Provider
      value={{ user, setUser, patientUid, setPatientUid, patientName, setPatientName }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => useContext(AppContext);
