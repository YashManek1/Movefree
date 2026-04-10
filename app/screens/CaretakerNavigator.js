import React from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../src/colors';

import CaretakerDashboard from './CaretakerDashboard';
import MapScreen from './MapScreen';
import HardwareDashboard from './HardwareDashboard';
import SOSScreen from './SOSScreen';

const Tab = createBottomTabNavigator();

const TAB_CONFIG = [
  { name: 'HUD', component: CaretakerDashboard, icon: 'grid-outline', iconActive: 'grid' },
  { name: 'MAP', component: MapScreen, icon: 'map-outline', iconActive: 'map' },
  { name: 'VITALS', component: HardwareDashboard, icon: 'heart-outline', iconActive: 'heart' },
  { name: 'SOS', component: SOSScreen, icon: 'alert-circle-outline', iconActive: 'alert-circle' },
];

export default function CaretakerNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: styles.tabBar,
        tabBarActiveTintColor: COLORS.cyan,
        tabBarInactiveTintColor: COLORS.textMuted,
        tabBarLabelStyle: styles.tabLabel,
        tabBarItemStyle: styles.tabItem,
        tabBarIcon: ({ focused, color, size }) => {
          const tab = TAB_CONFIG.find((t) => t.name === route.name);
          const iconName = focused ? tab.iconActive : tab.icon;
          return (
            <View style={focused ? styles.activeIconWrapper : null}>
              <Ionicons name={iconName} size={20} color={color} />
            </View>
          );
        },
      })}
    >
      {TAB_CONFIG.map((tab) => (
        <Tab.Screen
          key={tab.name}
          name={tab.name}
          component={tab.component}
          options={{ title: tab.name }}
        />
      ))}
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: '#0D111E',
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingBottom: Platform.OS === 'ios' ? 20 : 6,
    paddingTop: 6,
    height: Platform.OS === 'ios' ? 80 : 60,
  },
  tabLabel: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1,
    marginTop: 2,
  },
  tabItem: { paddingTop: 4 },
  activeIconWrapper: {
    backgroundColor: COLORS.cyanBg,
    borderRadius: 8,
    padding: 4,
  },
});
