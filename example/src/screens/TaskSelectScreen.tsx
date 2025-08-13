import { useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { useAI } from '../context/AIContext';

type RootStackParamList = {
  TaskSelect: undefined;
  Settings: undefined;
  Interact: undefined;
  Memory: undefined;
  PerformanceTest: undefined;
};

type Props = NativeStackScreenProps<RootStackParamList, 'TaskSelect'>;

export default function TaskSelectScreen({ navigation }: Props) {
  const { setTask, unloadPipeline } = useAI();

  useFocusEffect(
    useCallback(() => {
      // Dispose any loaded model when returning to home
      unloadPipeline();
      return () => {};
    }, [unloadPipeline])
  );
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Select AI Task</Text>
      <TouchableOpacity
        style={styles.button}
        onPress={() => {
          unloadPipeline();
          setTask('text-generation');
          navigation.navigate('Settings');
        }}
      >
        <Text style={styles.buttonText}>Text Generation (Chat)</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => {
          unloadPipeline();
          setTask('automatic-speech-recognition');
          navigation.navigate('Settings');
        }}
      >
        <Text style={styles.buttonText}>Automatic Speech Recognition</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => {
          unloadPipeline();
          setTask('text-to-audio');
          navigation.navigate('Settings');
        }}
      >
        <Text style={styles.buttonText}>Text to Audio (TTS)</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => {
          navigation.navigate('PerformanceTest');
        }}
      >
        <Text style={styles.buttonText}>Performance Test</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  title: { fontSize: 20, marginBottom: 16 },
  button: {
    backgroundColor: '#1E88E5',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 8,
    marginVertical: 6,
    width: '90%',
  },
  buttonText: { color: 'white', textAlign: 'center', fontSize: 16 },
});
