import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { AIProvider } from './context/AIContext';
import TaskSelectScreen from './screens/TaskSelectScreen';
import SettingsScreen from './screens/SettingsScreen';
import ChatScreen from './screens/ChatScreen';
import ASRScreen from './screens/ASRScreen';
import TTSScreen from './screens/TTSScreen';

export type RootStackParamList = {
  TaskSelect: undefined;
  Settings: undefined;
  Chat: undefined;
  ASR: undefined;
  TTS: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <AIProvider>
      <NavigationContainer>
        <Stack.Navigator initialRouteName="TaskSelect">
          <Stack.Screen
            name="TaskSelect"
            component={TaskSelectScreen}
            options={{ title: 'Select AI Task' }}
          />
          <Stack.Screen
            name="Settings"
            component={SettingsScreen}
            options={{ title: 'Settings' }}
          />
          <Stack.Screen
            name="Chat"
            component={ChatScreen}
            options={{ title: 'Chat' }}
          />
          <Stack.Screen
            name="ASR"
            component={ASRScreen}
            options={{ title: 'ASR' }}
          />
          <Stack.Screen
            name="TTS"
            component={TTSScreen}
            options={{ title: 'TTS' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </AIProvider>
  );
}
