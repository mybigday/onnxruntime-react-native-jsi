import { useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useAI } from '../context/AIContext';

type RootStackParamList = {
  TaskSelect: undefined;
  Settings: undefined;
  Chat: undefined;
  ASR: undefined;
  TTS: undefined;
  Memory: undefined;
};

type Props = NativeStackScreenProps<RootStackParamList, 'Settings'>;

export default function SettingsScreen({ navigation }: Props) {
  const {
    task,
    modelId,
    setModelId,
    dtype,
    setDtype,
    isLoading,
    isLoaded,
    loadPipeline,
    unloadPipeline,
    error,
  } = useAI();
  const onContinue = useCallback(() => {
    if (task === 'text-generation') {
      navigation.navigate('Chat');
    } else if (task === 'automatic-speech-recognition') {
      navigation.navigate('ASR');
    } else if (task === 'text-to-audio') {
      navigation.navigate('TTS');
    }
  }, [task, navigation]);

  const onApply = useCallback(async () => {
    if (isLoaded) unloadPipeline();
    await loadPipeline();
    onContinue();
  }, [isLoaded, unloadPipeline, loadPipeline, onContinue]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Settings - {task}</Text>
      <Text>Model ID</Text>
      <TextInput
        style={styles.input}
        value={modelId}
        onChangeText={(text) => setModelId(text.trim())}
        autoCapitalize="none"
        autoCorrect={false}
      />
      <Text>Quantize dtype (transformers.js)</Text>
      <View style={styles.chipsRow}>
        {(
          [
            'auto',
            'fp32',
            'fp16',
            'q8',
            'q4',
            'q4f16',
            'int8',
            'uint8',
          ] as const
        ).map((opt) => (
          <TouchableOpacity
            key={opt}
            style={[styles.chip, dtype === opt && styles.chipSelected]}
            onPress={() => setDtype(opt)}
          >
            <Text
              style={[
                styles.chipText,
                dtype === opt && styles.chipTextSelected,
              ]}
            >
              {opt}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
      {Boolean(error) && <Text style={styles.errorText}>{String(error)}</Text>}
      <TouchableOpacity
        style={[styles.button, (isLoaded || isLoading) && styles.disabled]}
        onPress={onApply}
        disabled={isLoaded || isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>
            {isLoaded ? 'Reload' : 'Load'} Model
          </Text>
        )}
      </TouchableOpacity>
      <View style={{ height: 12 }} />
      <TouchableOpacity
        style={[styles.button, styles.secondary, !isLoaded && styles.disabled]}
        onPress={onContinue}
        disabled={!isLoaded}
      >
        <Text style={styles.buttonText}>Continue</Text>
      </TouchableOpacity>
      <View style={{ height: 12 }} />
      <TouchableOpacity
        style={[
          styles.button,
          styles.warning,
          (!isLoaded || isLoading) && styles.disabled,
        ]}
        onPress={() => {
          unloadPipeline();
        }}
        disabled={!isLoaded || isLoading}
      >
        <Text style={styles.buttonText}>Unload Model</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16 },
  title: { fontSize: 20, marginBottom: 12 },
  input: {
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 12,
  },
  button: {
    backgroundColor: '#1E88E5',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  warning: { backgroundColor: '#B71C1C' },
  secondary: { backgroundColor: '#43A047' },
  disabled: { opacity: 0.5 },
  buttonText: { color: 'white', fontWeight: '600' },
  errorText: { color: 'crimson', marginBottom: 12 },
  chipsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 12,
  },
  chip: {
    borderColor: '#999',
    borderWidth: 1,
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginRight: 8,
    marginTop: 8,
  },
  chipSelected: { backgroundColor: '#1E88E5', borderColor: '#1E88E5' },
  chipText: { color: '#333' },
  chipTextSelected: { color: 'white' },
});
